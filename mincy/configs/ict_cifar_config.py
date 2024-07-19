from .ict_config import denoiser_config, trainer_config

cifar_config = {
    **denoiser_config,
    "channel_mults": (1, 2, 4, 8),
    "attention_mults": (2,),
    "kernel_size": (3, 3),
    "num_init_channels": 16,
    "num_res_blocks": 4,
    "pos_emb_type": "fourier",
    "pos_emb_dim": 16,
    "rescale_skip_conns": True,
    "resblock_variant": "BigGAN++",
    "fourier_scale": 0.002,
    "dropout": 0.3,
    "num_classes": 10,
    "class_cond": True
}

cifar_trainer_config = {
    **trainer_config,
    "learning_rate": 1e-4,
    "use_ema": True,
    "ema_decay": 0.99990,
}
