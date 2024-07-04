import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
from torch.utils.data import DataLoader
from flax.jax_utils import replicate, unreplicate

from ..components.karras_utils import get_boundaries
from ..components.consistency_utils import *
from .dataloader import reverse_transform

from functools import partial
from tqdm import trange
from typing import Any


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

    data_dim = jnp.prod(jnp.array(x.shape[1:]))
    c_data = huber_const * jnp.sqrt(data_dim)

    noise = random.normal(random_key, x.shape)

    @jax.jit
    def loss_fn(params):
        t1_noise_dim = cast_dim(t1, noise.ndim)
        t2_noise_dim = cast_dim(t2, noise.ndim)

        xt1_raw = x + t1_noise_dim * noise
        xt2_raw = x + t2_noise_dim * noise

        _, xt1 = jax.lax.stop_gradient(consistency_fn(
            xt1_raw, t1, sigma_data, sigma_min, state.apply_fn, params))
        _, xt2 = consistency_fn(
            xt2_raw, t2, sigma_data, sigma_min, state.apply_fn, params)

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
        self.consistency_config = consistency_config
        self.dataloader = dataloader
        self.random_key, init_key = random.split(random_key)

        assert dataloader.batch_size % num_devices == 0, "Batch size must be divisible by the number of devices!"
        self.num_devices = num_devices
        device_batch_size = dataloader.batch_size // num_devices
        self.device_batch_shape = (device_batch_size, *img_shape)

        init_input = jnp.ones(self.device_batch_shape)
        init_time = jnp.ones((device_batch_size,))
        model_params = model.init(init_key, init_input, init_time, train=True)

        self.state = train_state.TrainState.create(
            apply_fn=model.apply, params=model_params, tx=optimizer)

    def train(self, train_steps: int):
        parallel_state = replicate(self.state)

        with trange(train_steps) as steps:
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

                parallel_state, loss = train_step(
                    device_keys,
                    parallel_state,
                    x_parallel,
                    y_parallel,
                    t1, t2,
                    config["sigma_data"],
                    config["sigma_min"],
                    config["huber_const"]
                )

                steps.set_postfix(loss=unreplicate(loss))

                if (step + 1) % self.config["checkpoint_granularity"] == 0:
                    self._unreplicate_and_save(
                        parallel_state, step, save_checkpoint=True)

                if (step + 1) % self.config["snapshot_granularity"] == 0:
                    self._unreplicate_and_save(
                        parallel_state, step, save_checkpoint=False)

    def _unreplicate_and_save(self, parallel_state, step, save_checkpoint=False):
        self.state = unreplicate(parallel_state)
        if save_checkpoint:
            self.save_checkpoint(step)
        else:
            self.save_snapshot(step)

    def save_snapshot(self, step):
        self.random_key, snapshot_key = random.split(self.random_key)
        outputs = sample_single_step(snapshot_key,
                                     self.state.apply_fn,
                                     self.device_batch_shape,
                                     self.consistency_config["sigma_data"],
                                     self.consistency_config["sigma_min"],
                                     self.consistency_config["sigma_max"])

        pillow_outputs = [reverse_transform(output) for output in outputs]

        os.makedirs(self.config["snapshot_dir"], exist_ok=True)

        for idx, output in enumerate(pillow_outputs[:self.config["samples_to_keep"]]):
            fpath = f"{self.config["snapshot_dir"]}/img_it{step}_n{idx + 1}.png"
            output.save(fpath)

    def save_checkpoint(self, step):
        checkpoints.save_checkpoint(self.config["checkpoint_dir"],
                                    target=self.state,
                                    step=step,
                                    overwrite=True,
                                    keep=self.config["checkpoints_to_keep"])
