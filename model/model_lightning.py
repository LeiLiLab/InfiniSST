import torch
from torch.utils.data import DataLoader

import lightning as L
from deepspeed.ops.adam import FusedAdam

from train.dataset import SpeechSampler

class SLlamaLightning(L.LightningModule):
    def __init__(
            self, model, 
            train_ds=None, dev_ds=None, train_bsz=None, dev_bsz=None, collate_fn=None,
            lr=2e-4, warmup_updates=4000, min_lr=0.
        ):
        super().__init__()
        self.model = model

        self.datasets = {
            "train_ds": train_ds,
            "dev_ds": dev_ds,
            "train_bsz": train_bsz,
            "dev_bsz": dev_bsz,
            "collate_fn": collate_fn
        }

        self.optimizer_params = {
            "lr": lr,
            "warmup_updates": warmup_updates,
            "min_lr": min_lr,
        }
    
    def train_dataloader(self):
        train_sampler = SpeechSampler(
            self.datasets["train_ds"], 
            shuffle=True, 
            batch_size=self.datasets["train_bsz"], 
            min_ms=320
        )
        train_dataloader = DataLoader(
            self.datasets["train_ds"], 
            batch_sampler=train_sampler, 
            collate_fn=self.datasets["collate_fn"]
        )
        return train_dataloader
    
    def val_dataloader(self):
        dev_sampler = SpeechSampler(
            self.datasets["dev_ds"], 
            shuffle=False, 
            batch_size=self.datasets["dev_bsz"], 
            min_ms=320
        )
        dev_dataloader = DataLoader(
            self.datasets["dev_ds"], 
            batch_sampler=dev_sampler, 
            collate_fn=self.datasets["collate_fn"]
        )
        return dev_dataloader

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("train/loss", loss, batch_size=batch["src_lengths"].sum() / 16000)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("eval/loss", loss, batch_size=batch["src_lengths"].sum() / 16000)

    def configure_optimizers(self):
        lr = self.optimizer_params["lr"]
        min_lr = self.optimizer_params["min_lr"]
        warmup_updates = self.optimizer_params["warmup_updates"]

        optimizer = FusedAdam(self.parameters(), lr=lr)        
        warmup_init_lr = 0 if warmup_updates > 0 else lr
        lr_step = (lr - warmup_init_lr) / warmup_updates
        decay_factor = lr * warmup_updates**0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda x: max(decay_factor * x**-0.5 if x >= warmup_updates \
                else warmup_init_lr + x * lr_step, min_lr) / lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def forward(self, batch):
        output = self.model(
            **batch,
            return_dict=True
        )
        return output.loss