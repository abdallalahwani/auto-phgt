"""Shared HGB-style training settings used by the published model utilities."""

BACKBONE_FIXED = {
    "d_model": 64,
    "heads": 8,
    "max_epochs": 300,
    "patience": 30,
    "schedule": "onecycle",
    "pct_start": 0.05,
    "val_ratio": 0.2,
    "early_stopping": "validation loss",
    "optimizer": "AdamW",
}

AUTO_PHGT = {
    "fusion_layers": 2,
    "fusion_heads": 4,
    "ffn_mult": 2,
    "pooling": "mean",
    "token_dropout": 0.1,
    "instances_per_path": 4,
}
