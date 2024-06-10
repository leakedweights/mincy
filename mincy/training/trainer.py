import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
from torch.utils.data import DataLoader
from flax.jax_utils import replicate, unreplicate

from ..models.consistency import *
from ..components.schedule import *

from tqdm import trange
from typing import Any


class ConsistencyTrainer:

    def __init__(self,
                 random_key: Any,
                 model: nn.Module,
                 optimizer: Any,
                 dataloader: DataLoader,
                 img_shape,
                 batch_size,
                 num_devices: int,
                 config: dict,
                 consistency_config: dict):

        self.model = model
        self.config = config
        self.consistency_config = consistency_config
        self.dataloader = dataloader
        self.random_key, init_key = random.split(random_key)

        assert batch_size % num_devices == 0, "Batch size must be divisible by the number of devices!"
        self.num_devices = num_devices
        device_batch_size = batch_size // num_devices

        init_input = jnp.ones((device_batch_size, *img_shape))
        init_time = jnp.ones((device_batch_size,))
        model_params = model.init(init_key, init_input, init_time, train=True)

        self.state = train_state.TrainState.create(
            apply_fn=model, params=model_params, tx=optimizer)

    @staticmethod
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(5, 6))
    def train_step(random_key: Any, state: train_state.TrainState,
                   x: jax.Array, y: jax.Array, step: int, max_steps: int, config: dict):

        timestep_key, noise_key = random.split(random_key)

        data_dim = jnp.prod(jnp.array(x.shape[1:]))
        c_data = config["huber_const"] * jnp.sqrt(data_dim)

        N = discretize(step, config["s0"], config["s1"], max_steps)

        levels = [karras_levels(
            i, N, config["sigma_max"], config["rho"], config["sigma_max"], config["rho"]) for i in range(1, int)]

        t1, t2 = sample_timesteps(
            timestep_key, levels, x.shape[:1], config["p_mean"], config["p_std"])
        noise = random.normal(noise_key, x.shape)

        denoising_fn = jax.tree_util.Partial(
            state.apply_fn, params=state.params)

        xt1, xt2 = training_consistency(
            t1=t1,
            t2=t2,
            x0=x,
            noise=noise,
            denoising_fn=denoising_fn,
            sigma_data=config["sigma_data"],
            sigma_min=config["sigma_min"]
        )

        loss, grads = jax.value_and_grad(
            pseudo_huber_loss)(xt1, xt2, c_data=c_data)
        loss = jax.lax.pmean(loss, "batch")

        state = state.apply_gradients(grads=grads)
        return state, loss

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

                self.random_key, *device_keys = random.split(
                    self.random_key, self.num_devices + 1)

                consistency_config = FrozenDict(self.consistency_config)

                jax.debug.print('device_key_shape = {device_key_shape}', device_key_shape=jnp.array(
                    device_keys).shape)

                parallel_state, loss = self.train_step(jnp.array(
                    device_keys), parallel_state, x_parallel, y_parallel, replicate(step), train_steps, consistency_config)

                steps.set_postfix(loss=unreplicate(loss))
