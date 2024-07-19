import os
import jax
import flax
from jax import random
import jax.numpy as jnp
from flax import linen as nn
from torch.utils.data import DataLoader
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state, checkpoints

from ..utils import cast_dim, update_ema
from .dataloader import reverse_transform
from ..components.consistency_utils import *

import torch
import wandb
from typing import Any
from tqdm import trange
from cleanfid import fid
from functools import partial
from dataclasses import dataclass


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    ema_params: Any


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(6, 7, 8))
def train_step(random_key: Any,
               state: train_state.TrainState,
               x: jax.Array,
               y: jax.Array,
               t1: jax.Array,
               t2: jax.Array,
               sigma_data: float,
               sigma_min: float,
               huber_const: float):

    noise_key, dropout_t1, dropout_t2 = random.split(random_key, 3)
    data_dim = jnp.prod(jnp.array(x.shape[1:]))
    c_data = huber_const * jnp.sqrt(data_dim)

    noise = random.normal(noise_key, x.shape)

    @jax.jit
    def loss_fn(params):
        t1_noise_dim = cast_dim(t1, noise.ndim)
        t2_noise_dim = cast_dim(t2, noise.ndim)

        xt1_raw = x + t1_noise_dim * noise
        xt2_raw = x + t2_noise_dim * noise

        _, xt1 = jax.lax.stop_gradient(consistency_fn(
            xt1_raw, y, t1, sigma_data, sigma_min, partial(state.apply_fn, rngs={"dropout": dropout_t1}), params))
        _, xt2 = consistency_fn(
            xt2_raw, y, t2, sigma_data, sigma_min, partial(state.apply_fn, rngs={"dropout": dropout_t2}), params)

        loss = pseudo_huber_loss(xt2, xt1, c_data)
        weight = cast_dim((1 / (t2 - t1)), loss.ndim)

        return jnp.mean(weight * loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    loss = jax.lax.pmean(loss, "batch")

    state = state.apply_gradients(grads=grads)
    return state, loss


class ConsistencyTrainer:

    def __init__(self,
                 random_key: Any,
                 model: nn.Module,
                 optimizer: Any,
                 dataloader: DataLoader,
                 img_shape,
                 num_devices: int,
                 config: dict,
                 consistency_config: dict):

        self.model = model
        self.config = config
        self.checkpoint_step = 0
        self.consistency_config = consistency_config
        self.dataloader = dataloader
        self.random_key, self.snapshot_key, init_key = random.split(
            random_key, 3)

        assert dataloader.batch_size % num_devices == 0, "Batch size must be divisible by the number of devices!"
        self.num_devices = num_devices
        device_batch_size = dataloader.batch_size // num_devices
        self.device_batch_shape = (device_batch_size, *img_shape)

        init_input = jnp.ones(self.device_batch_shape)
        init_labels = jnp.ones((device_batch_size,), dtype=jnp.int32)
        init_time = jnp.ones((device_batch_size,))
        model_params = model.init(init_key, init_input, init_labels, init_time, train=True)

        self.state = TrainState.create(
            apply_fn=model.apply, params=model_params, ema_params=model_params, tx=optimizer)

    def train(self, train_steps: int):
        parallel_state = replicate(self.state)

        with trange(self.checkpoint_step, train_steps, initial=self.checkpoint_step, total=train_steps) as steps:
            cumulative_loss = 0.0
            for step in steps:
                try:
                    batch = next(self.dataloader.__iter__())
                except StopIteration:
                    continue

                x_batch, y_batch = batch
                _, *data_dim = x_batch.shape
                _, *label_dim = y_batch.shape

                x_parallel = x_batch.reshape(self.num_devices, -1, *data_dim)
                y_parallel = y_batch.reshape(self.num_devices, -1, *label_dim)

                self.random_key, schedule_key, *device_keys = random.split(
                    self.random_key, self.num_devices + 2)

                device_keys = jnp.array(device_keys)

                config = self.consistency_config

                N = discretize(
                    step, config["s0"], config["s1"], train_steps)

                noise_levels = get_boundaries(
                    N, config["sigma_min"], config["sigma_max"], config["rho"])

                t1, t2 = sample_timesteps(
                    schedule_key, noise_levels, x_parallel.shape[:2], config["p_mean"], config["p_std"])

                parallel_state, parallel_loss = train_step(
                    device_keys,
                    parallel_state,
                    x_parallel,
                    y_parallel,
                    t1, t2,
                    config["sigma_data"],
                    config["sigma_min"],
                    config["huber_const"]
                )

                parallel_state = parallel_state.replace(
                    ema_params=update_ema(
                        parallel_state.ema_params,
                        parallel_state.params,
                        self.config["ema_decay"])
                )

                loss = unreplicate(parallel_loss)
                steps.set_postfix(loss=loss)
                cumulative_loss += loss
                log_freq = self.config["log_frequency"]

                if ((step + 1) % log_freq == 0) and self.config["log_wandb"]:
                    avg_loss = cumulative_loss / log_freq
                    cumulative_loss = 0
                    wandb.log({"step": step, "train_loss": avg_loss})

                save_checkpoint = (
                    step + 1) % self.config["checkpoint_frequency"] == 0
                save_snapshot = self.config["create_snapshots"] and (step == 0 or (
                    step + 1) % self.config["snapshot_frequency"] == 0)
                self._save(
                    parallel_state, step, save_checkpoint, save_snapshot)

                run_eval = self.config["run_evals"] and (
                    step + 1) % self.config["eval_frequency"] == 0
                if run_eval:
                    fid_score = self.run_eval(step, state)
                    wandb.log({"step": step, "fid_score": fid_score})

        self._save(
            parallel_state, train_steps, save_checkpoint=True, save_snapshot=True)

    def _save(self, parallel_state, step, save_checkpoint, save_snapshot):
        if not (save_checkpoint or save_snapshot):
            return

        self.state = unreplicate(parallel_state)
        if save_checkpoint:
            self.save_checkpoint(step)
        if save_snapshot:
            self.save_snapshot(step)

    def save_snapshot(self, step):
        outputs = sample_single_step(self.snapshot_key,
                                     self.state.apply_fn,
                                     self.state.ema_params,
                                     self.device_batch_shape,
                                     self.consistency_config["sigma_data"],
                                     self.consistency_config["sigma_min"],
                                     self.consistency_config["sigma_max"])

        pillow_outputs = [reverse_transform(output) for output in outputs]

        os.makedirs(self.config["snapshot_dir"], exist_ok=True)

        for idx, output in enumerate(pillow_outputs[:self.config["samples_to_keep"]]):
            fpath = f"{self.config['snapshot_dir']}/img_it{step+1}_n{idx + 1}.png"
            output.save(fpath)

    def save_checkpoint(self, step):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        checkpoints.save_checkpoint(self.config["checkpoint_dir"],
                                    target={"state": self.state, "step": step},
                                    step=step,
                                    overwrite=True,
                                    keep=self.config["checkpoints_to_keep"])

        self.checkpoint_step = step

    def load_checkpoint(self):
        target = {"state": self.state, "step": 0}
        try:
            restored = checkpoints.restore_checkpoint(
                ckpt_dir=self.config["checkpoint_dir"], target=target)
            self.checkpoint_step = int(restored["step"])
            self.state = restored["state"]
        except Exception as e:
            print(f"Unable to load checkpoint: {e}")

    def generate(self, key):
        return sample_single_step(key,
                                  self.state.apply_fn,
                                  self.state.params_ema,
                                  self.device_batch_shape,
                                  self.consistency_config["sigma_data"],
                                  self.consistency_config["sigma_min"],
                                  self.consistency_config["sigma_max"])

    def run_eval(self):
        from mincy.training.dataloader import reverse_transform

        eval_dir = f"../eval/synthetic"
        os.makedirs(eval_dir, exist_ok=True)
        num_synthetic_samples = 10_000

        sample_key = random.key(0)

        i = 0

        while i < num_synthetic_samples:
            sample_key, subkey = random.split(sample_key)
            samples = self.generate(sample_key)

            pillow_outputs = [reverse_transform(
                output) for output in samples[:min(num_synthetic_samples - i, len(samples))]]
            for idx, output in enumerate(pillow_outputs):
                fpath = f"{eval_dir}/{i + idx}.png"
                output.save(fpath)

            i += len(pillow_outputs)

        score = fid.compute_fid(eval_dir, dataset_name="cifar10",
                                dataset_res=32, dataset_split="test", device=torch.device('cpu'))
