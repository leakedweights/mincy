# Improved Techniques for Training Consistency Models

[![arXiv](https://img.shields.io/badge/arXiv-2310.14189-b31b1b.svg)](https://arxiv.org/abs/2310.14189)
<a target="_blank" href="https://colab.research.google.com/github/leakedweights/mincy/blob/main/notebooks/ict_cifar.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This repository contains the implementation for the paper *Improved Techniques for Training Consistency Models* by song et al. (https://arxiv.org/abs/2310.14189).

# Project Structure

```
mincy/
├── components/
│   ├── blocks.py
│   ├── embeddings.py
│   ├── schedule.py
│   └── solvers.py
├── configs/
│   ├── ict_cifar.py
│   └── ict.py
├── models/
│   ├── consistency.py
│   ├── unet.py
│   └── utils.py
├── training/
│   ├── dataloader.py
│   └── trainer.py
├── notebooks/
│   ├── ict_cifar.ipynb
│   └── ict_plots.ipynb
└── __init__.py
```