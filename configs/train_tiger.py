# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW

from ctdg.utils import LazyCall as L

from .datasets.forest import get_dataset
from .models.tiger import get_model

seed = 0

dataset, nodes_dim, events_dim = get_dataset("memetracker")

model = get_model(nodes_dim, events_dim)

data = {
    "dataset": dataset,
    "train_batch_size": 200,
    "val_batch_size": 200,
    "test_batch_size": 200,
}

optimizer = L(AdamW)(lr=1e-4)

trainer = L(Trainer)(
    accelerator="auto",
    devices=1,
    precision="bf16-mixed",
    logger=[L(WandbLogger)(project="ctdg-cascade")],
    callbacks=[
        L(ModelCheckpoint)(
            every_n_epochs=1,
            save_last=True,
            save_top_k=1,
            monitor="val/msle",
            mode="min",
        ),
    ],
    max_epochs=100,
    log_every_n_steps=10,
)
