# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW

from ctdg.utils import LazyCall as L

from .datasets.jodie import dataset
from .models.connect import model  # type: ignore

seed = 42

data = {
    "dataset": dataset,
    "train_batch_size": 200,
    "val_batch_size": 200,
    "test_batch_size": 200,
}

community_detection_p = 0.2
community_detection_every_n_steps = 10

optimizer = L(AdamW)(lr=1e-4)

trainer = L(Trainer)(
    accelerator="auto",
    devices=1,
    precision="bf16-mixed",
    logger=[L(WandbLogger)(project="ctdg")],
    callbacks=[
        L(ModelCheckpoint)(
            every_n_epochs=1,
            save_last=True,
            save_top_k=3,
            monitor="val/average_precision",
            mode="max",
        ),
    ],
    max_epochs=100,
    log_every_n_steps=10,
)
