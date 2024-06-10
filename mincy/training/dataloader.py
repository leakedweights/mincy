import torch
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import default_collate
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage


def numpy_collate(batch):
    batch = default_collate(batch)
    batch = tree_map(lambda x: np.asarray(x), batch)
    return batch


transform = Compose([
    ToTensor(),
    Lambda(lambda x: x.permute(1, 2, 0)),
    Lambda(lambda x: x * 2 - 1),
])

reverse_transform = Compose([
    Lambda(lambda x: torch.from_numpy(np.asarray(x))),
    Lambda(lambda x: x.permute(2, 0, 1)),
    Lambda(lambda x: 0.5 * (x - 1)),
    ToPILImage()
])
