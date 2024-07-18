import flax.linen as nn

denoiser_config = {
    "dropout": 0.0,
    "nonlinearity": nn.swish
}

consistency_config = {
    "sigma_data": 0.5,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "huber_const": 0.00054,
    "rho": 7.0,
    "p_mean": 1.1,
    "p_std": 2.0,
    "s0": 10,
    "s1": 1280
}

trainer_config = {
    "optimizer": None,
    "log_wandb": True,
    "log_frequency": 100,

    "create_snapshots": True,
    "snapshot_frequency": 10_000,
    "samples_to_keep": 5,
    "snapshot_dir": None,

    "run_evals": True,
    "eval_frequency": 10_000,
    "ground_truth_dir": None,
    "eval_dir": None,

    "checkpoint_frequency": 40_000,
    "checkpoints_to_keep": 5,
    "checkpoint_dir": None,
}
