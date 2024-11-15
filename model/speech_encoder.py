from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from x_transformers.x_transformers import (
    AttentionLayers, 
    LayerIntermediates
)

import lightning as L
from train.dataset import SpeechSampler

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]], # [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 4
        dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(
            n_in,
            n_out,
            k,
            stride,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias, padding=0)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(),
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    TransposeLast(),
                ),
                nn.GELU(),
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        
        # B*C*T

        return x

class SpeechEncoder(L.LightningModule):
    def __init__(
        self, 
        feature_extractor_cfg,
        n_attn_layers, n_dim, n_heads, dropout, block_size, max_cache_size,
        llm_embedding,
        train_ds, dev_ds, train_bsz, dev_bsz, collate_fn,
        lr, warmup_updates, min_lr=1e-6, temp=0.5, loss_fn='waco'
    ):
        super().__init__()

        self.feature_extractor_cfg = eval(feature_extractor_cfg)
        self.feature_extractor = ConvFeatureExtractionModel(self.feature_extractor_cfg)

        self.dropout = nn.Dropout(dropout)

        self.attn_layers = nn.ModuleList(
            [
                AttentionLayers(
                    n_dim, 
                    depth=1,
                    heads=n_heads,
                    rotary_pos_emb=True,
                    rotary_emb_dim=n_dim // n_heads,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                )
                for _ in range(n_attn_layers)
            ]
        )

        self.block_size = block_size
        self.max_cache_size = max_cache_size

        self.llm_embedding = llm_embedding
        self.llm_embedding.requires_grad_(False)
        self.proj = nn.Linear(n_dim, llm_embedding.embedding_dim)

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
        self.loss_fn = loss_fn
        self.temp = temp

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
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for cfg in self.feature_extractor_cfg:
            input_lengths = _conv_out_length(
                input_lengths, cfg[1], cfg[2]
            )

        return input_lengths.to(torch.long)
    
    def _get_attn_mask(self, bsz, seq_len, prefix_len, max_cache_size=None):
        max_len = seq_len + prefix_len

        blocksizes = [
            min(self.block_size, max_len - i * self.block_size) 
            for i in range((max_len + self.block_size - 1) // self.block_size)
        ]

        mask = torch.zeros(seq_len, max_len, device='cuda', dtype=torch.bool)
        start_idx = 0
        for block_size in blocksizes:
            end_idx = start_idx + block_size
            if end_idx > prefix_len:
                mask[max(0, start_idx - prefix_len) : end_idx - prefix_len, :end_idx] = 1
            start_idx = end_idx
        
        if max_cache_size is not None:
            for i in range(seq_len):
                mask[i, :max(0, i + max_len - seq_len - max_cache_size)] = 0
        
        return mask

    def encode_speech(self, src_tokens, src_lens, caches=None):
        x = self.feature_extractor(src_tokens)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        bsz, seq_len, _ = x.size()        
        
        # apply conv formula to get real output_lengths
        output_lengths = self._get_feat_extract_output_lengths(src_lens)
        padding_mask = torch.zeros(
            bsz, seq_len, dtype=x.dtype, device=x.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        padding_mask[
            (
                torch.arange(bsz, device=padding_mask.device),
                output_lengths - 1,
            )
        ] = 1
        padding_mask = padding_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        prefix_len = 0
        if caches is not None:
            prefix_len = caches[0].n_steps
            for cache in caches:
                k, v = cache.attn_intermediates.cached_kv
                cache.attn_intermediates.cached_kv = (
                    k[..., -self.max_cache_size:, :],
                    v[..., -self.max_cache_size:, :]
                )

            pos = torch.arange(
                max(0, prefix_len + seq_len - self.max_cache_size),
                prefix_len + seq_len
            )
            attn_mask = self._get_attn_mask(bsz, seq_len, prefix_len)
            attn_mask = attn_mask[:, -self.max_cache_size:]
        else:
            pos = torch.arange(seq_len)
            attn_mask = self._get_attn_mask(bsz, seq_len, 0, self.max_cache_size)
        
        updated_caches = []
        for i, layer in enumerate(self.attn_layers):
            x, cache = layer(
                x, 
                mask=padding_mask,
                attn_mask=attn_mask,
                cache=caches[i] if caches else None,
                cache_age=self.block_size,
                pos=pos,
                return_hiddens=True,
            )

            if hasattr(cache, "n_steps"):
                cache.n_steps += seq_len
            else:
                cache.n_steps = seq_len
                
            updated_caches.append(cache)
        
        x = self.proj(x)
        
        return x, updated_caches
    
    def forward(self, batch):
        src_speech = batch["src_speech"]
        src_speech_lengths = batch["src_speech_lengths"]

        src_text = batch["src_text"]
        src_text_lengths = batch["src_text_lengths"]

        speech_word = batch["speech_word"]
        text_word = batch["text_word"]
        
        src_speech_emb, _ = self.encode_speech(
            src_speech, 
            src_speech_lengths, 
        )
        src_speech_emb = src_speech_emb.float()

        if self.loss_fn == 'waco':
            src_text_emb = self.llm_embedding(src_text).float()
            speech_word_emb = []
            text_word_emb = []
            for i in range(len(text_word)):
                s_word, t_word = speech_word[i], text_word[i]
                if s_word is not None:
                    for j in range(s_word.size(0)):
                        s_l, s_r = s_word[j]
                        t_l, t_r = t_word[j]

                        # TODO: hard-coded here, one block equals 80ms
                        s_l = int((s_l / 0.08).floor())
                        s_r = min(int((s_r / 0.08).ceil()), src_speech_emb.size(1)) - 1

                        s_word_emb = src_speech_emb[i][s_l : s_r + 1].mean(dim=0)
                        t_word_emb = src_text_emb[i][t_l : t_r + 1].mean(dim=0)
                        speech_word_emb.append(s_word_emb)
                        text_word_emb.append(t_word_emb)
            speech_word_emb = torch.stack(speech_word_emb, dim=0)
            text_word_emb = torch.stack(text_word_emb, dim=0)

            st_sim = F.cosine_similarity(
                speech_word_emb.unsqueeze(1), 
                text_word_emb.unsqueeze(0), 
                dim=-1,
            )
            loss = F.cross_entropy(
                st_sim / self.temp,
                torch.arange(st_sim.size(0), device=st_sim.device)
            )
        else:
            raise NotImplementedError
        
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("train_loss", loss, batch_size=batch["src_speech_lengths"].sum() / 16000)
            self.log("train_nan", 0, batch_size=1)
        else:
            self.log("train_nan", 1, batch_size=1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("val_loss", loss, batch_size=batch["src_speech_lengths"].sum() / 16000)
            self.log("val_nan", 0, batch_size=1)
        else:
            self.log("val_nan", 1, batch_size=1)
    
    def configure_optimizers(self):
        lr = self.optimizer_params["lr"]
        min_lr = self.optimizer_params["min_lr"]
        warmup_updates = self.optimizer_params["warmup_updates"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)        
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