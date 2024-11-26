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

import fairseq
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.models.hubert import HubertEncoder
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
    Conv1dSubsampler,
)

from rotary_embedding_torch import RotaryEmbedding
from model.uni_wav2vec_monkey_patch_new import patch_w2v2

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
        in_d: int = 1,
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
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        
        # B*C*T

        return x

class Cache:
    src: torch.Tensor = None
    src_len: int = 0
    n_steps: int = 0
    layers: List[LayerIntermediates] = None

class SpeechEncoder(L.LightningModule):
    def __init__(
        self, 
        feature_extractor_cfg='[(1024, 10, 5)] + [(1024, 3, 2)] * 4 + [(1024,2,2)] * 4', 
        feature_extractor_state_dict_path=None, feature_extractor_freeze=False,
        length_shrink_cfg=None,
        n_attn_layers=12, n_dim=1024, n_heads=16, dropout=0.1, block_size=16, max_cache_size=125,
        llm_embedding=None,
        train_ds=None, dev_ds=None, train_bsz=None, dev_bsz=None, collate_fn=None,
        lr=1e-4, warmup_updates=0, min_lr=1e-6, temp=0.5, loss_fn='waco'
    ):
        super().__init__()

        self.feature_extractor_cfg = eval(feature_extractor_cfg)
        self.feature_extractor = ConvFeatureExtractionModel(self.feature_extractor_cfg)

        if feature_extractor_state_dict_path is not None:
            state_dict = torch.load(feature_extractor_state_dict_path)
            self.feature_extractor.load_state_dict(state_dict, strict=False)
            if feature_extractor_freeze:
                for name, param in self.feature_extractor.named_parameters():
                    if name in state_dict:
                        param.requires_grad_(False)
        
        self.post_feature_extractor_proj = None
        if self.feature_extractor_cfg[-1][0] != n_dim:
            self.post_feature_extractor_proj = nn.Linear(self.feature_extractor_cfg[-1][0], n_dim)

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

        self.length_shrink = None
        if length_shrink_cfg is not None:
            self.length_shrink_cfg = eval(length_shrink_cfg)
            self.length_shrink = ConvFeatureExtractionModel(self.length_shrink_cfg, in_d=n_dim)

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
        if self.length_shrink:
            for cfg in self.length_shrink_cfg:
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

    def encode_speech(self, src_tokens, src_lens, cache=None):
        if cache is None:
            cache = Cache()

        if cache.src_tokens is not None:
            src_tokens = torch.cat([cache.src_tokens, src_tokens], dim=1)
        cache.src_tokens = src_tokens

        x = self.feature_extractor(src_tokens)
        x = x.transpose(1, 2)

        if cache.src_len > 0:
            new_src_len = x.size(1)
            x = x[:, cache.src_len:]
            cache.src_len = new_src_len

            max_src_token_len = 79 + 1280 + 1280 * self.block_size
            if cache.src_tokens.size(1) > max_src_token_len:
                cache.src_tokens = cache.src_tokens[:, -max_src_token_len:]
                cache.src_len = self.block_size
        else:
            cache.src_len = x.size(1)

        if self.post_feature_extractor_proj is not None:
            x = self.post_feature_extractor_proj(x)

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
        if cache.n_steps > 0:
            prefix_len = cache.n_steps
            for cache_per_layer in cache.layers:
                k, v = cache_per_layer.attn_intermediates[0].cached_kv
                cache_per_layer.attn_intermediates[0].cached_kv = (
                    k[..., -self.max_cache_size:, :],
                    v[..., -self.max_cache_size:, :]
                )

            pos = torch.arange(
                max(0, prefix_len - self.max_cache_size),
                prefix_len + seq_len
            )
            attn_mask = self._get_attn_mask(bsz, seq_len, prefix_len)
            attn_mask = attn_mask[:, -self.max_cache_size-seq_len:]
        else:
            pos = torch.arange(seq_len)
            attn_mask = self._get_attn_mask(bsz, seq_len, 0, self.max_cache_size)
        cache.n_steps += seq_len
        
        if cache.layers is None:
            cache.layers = [None for _ in range(len(self.attn_layers))]
        for i, layer in enumerate(self.attn_layers):
            x, x_cache = layer(
                x, 
                mask=padding_mask if cache.layers[i] is None else None,
                attn_mask=attn_mask,
                cache=cache.layers[i],
                cache_age=self.block_size,
                pos=pos,
                return_hiddens=True,
            )
            cache.layers[i] = x_cache

        if self.length_shrink is not None:
            x = x.transpose(1, 2)
            x = self.length_shrink(x)
            x = x.transpose(1, 2)
        
        x = self.proj(x)
        
        return x, cache
    
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("val_loss", loss, batch_size=batch["src_speech_lengths"].sum() / 16000)
    
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
    
class ZeroPadConv1dSubsampler(Conv1dSubsampler):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=0,
            )
            for i, k in enumerate(kernel_sizes)
        )
    
    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 3) / 2 + 1).floor().long()
        return out 

class SpeechEncoderW2V2(L.LightningModule):
    def __init__(
        self, 
        speech_tower_path, ssl_finetuned, 
        len_adapter_channels, len_adapter_kernel_sizes, 
        llm_embedding, unidirectional, temp,
        lr, warmup_updates, min_lr=1e-6,
        loss_fn='waco'
    ):
        super().__init__()
        if not ssl_finetuned: # ssl model
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(speech_tower_path)
            w2v_args = state["args"]
            task = fairseq.tasks.setup_task(w2v_args)
            model = task.build_model(w2v_args)
            model.load_state_dict(state["model"], strict=True)
            speech_dimension = w2v_args.encoder_embed_dim
        else: # ctc finetune, w2v-ctc
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(speech_tower_path)
            model = Wav2VecEncoder(state['cfg']['model'], None)
            new = {}
            for key in state['model'].keys():
                new_key = key.replace('w2v_encoder.', '')
                if not new_key.startswith('proj'):
                    new[new_key] = state['model'][key]
            model.load_state_dict(new, strict=True)
            model = model.w2v_model
            speech_dimension = state['cfg']['model']['w2v_args']['model'].encoder_embed_dim
            
        self.speech_tower = model

        if unidirectional:
            self.mm_length_adapter = ZeroPadConv1dSubsampler(
                                    speech_dimension,
                                    len_adapter_channels,
                                    speech_dimension,
                                    [int(k) for k in len_adapter_kernel_sizes.split(',')]
                                )             
        else:
            self.mm_length_adapter = Conv1dSubsampler(
                                        speech_dimension,
                                        len_adapter_channels,
                                        speech_dimension,
                                        [int(k) for k in len_adapter_kernel_sizes.split(',')]
                                    ) 
            
        self.mm_mlp_adapter = nn.Linear(speech_dimension, llm_embedding.embedding_dim)
        self.length_after_ssl = self.speech_tower._get_feat_extract_output_lengths
        self.length_after_adp = self.mm_length_adapter.get_out_seq_lens_tensor

        self.llm_embedding = llm_embedding
        self.llm_embedding.requires_grad_(False)

        self.lr = lr
        self.warmup_updates = warmup_updates
        self.min_lr = min_lr
        self.temp = temp

        self.loss_fn = loss_fn

    def train_dataloader(self):
        train_sampler = SpeechSampler(self.train_ds, shuffle=False, batch_size=self.train_bsz, min_ms=320)
        train_dataloader = DataLoader(self.train_ds, batch_sampler=train_sampler, collate_fn=self.collate)
        return train_dataloader
    
    def val_dataloader(self):
        dev_sampler = SpeechSampler(self.dev_ds, shuffle=False, batch_size=self.dev_bsz, min_ms=320)
        dev_dataloader = DataLoader(self.dev_ds, batch_sampler=dev_sampler, collate_fn=self.collate)
        return dev_dataloader

    def get_ssl_feature_w2v(self, src_tokens, src_lengths, after_lens):
        padding_mask = lengths_to_padding_mask(src_lengths)
        res = self.speech_tower.extract_features(src_tokens, padding_mask)
        feature, padding_mask = res["x"], res["padding_mask"]
        if padding_mask is None:
        # Create a padding mask of shape [batch_size, seq_length] with all False values
            padding_mask = torch.zeros(feature.shape[:2], dtype=torch.bool, device=feature.device)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        feature, input_lengths = self.mm_length_adapter(feature, output_length)
        # assert after_lens.equal(input_lengths), "pre calculate length not match with the forward length"
        feature = self.mm_mlp_adapter(feature)       
        return feature
    
    def forward(self, batch):
        src_text = batch["src_text"]
        src_speech = batch["src_speech"]
        src_speech_lengths = batch["src_speech_lengths"]
        after_speech_lengths = batch["after_speech_lengths"]
        src_text_lengths = batch["src_text_lengths"]
        text_word = batch["text_word"]
        speech_word = batch["speech_word"]

        src_speech_emb = self.get_ssl_feature_w2v(src_speech, src_speech_lengths, after_speech_lengths).transpose(0, 1).float()

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
                        s_word_emb = src_speech_emb[i][s_l : s_r + 1].mean(dim=0)
                        t_word_emb = src_text_emb[i][t_l : t_r + 1].mean(dim=0)
                        speech_word_emb.append(s_word_emb)
                        text_word_emb.append(t_word_emb)
            speech_word_emb = torch.stack(speech_word_emb, dim=0)
            text_word_emb = torch.stack(text_word_emb, dim=0)

            st_sim = F.cosine_similarity(
                speech_word_emb.unsqueeze(1), 
                text_word_emb.unsqueeze(0), 
                dim=-1
            )
            loss = F.cross_entropy(
                st_sim / self.temp,
                torch.arange(st_sim.size(0), device=st_sim.device)
            )
        elif self.loss_fn == 'ctc':
            logits = torch.matmul(src_speech_emb, self.llm_embedding.weight.T)
            log_probs = F.log_softmax(logits / self.temp, dim=-1).transpose(0, 1)
            loss = F.ctc_loss(
                log_probs,
                src_text,
                after_speech_lengths,
                src_text_lengths,
                blank=2,
                zero_infinity=True
            )
        else:
            raise NotImplementedError

        if loss.isnan():
            return None
        
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if loss is not None:
            self.log("train_loss", loss, batch_size=batch["src_speech_lengths"].sum() / 16000)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if loss is not None:
            self.log("val_loss", loss, batch_size=batch["src_speech_lengths"].sum() / 16000)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        warmup_init_lr = 0 if self.warmup_updates > 0 else self.lr
        lr_step = (self.lr - warmup_init_lr) / self.warmup_updates
        decay_factor = self.lr * self.warmup_updates**0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda x: max(decay_factor * x**-0.5 if x >= self.warmup_updates \
                else warmup_init_lr + x * lr_step, self.min_lr) / self.lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
class LayerCache:
    k: torch.Tensor = None
    v: torch.Tensor = None
    key_padding_mask: torch.Tensor = None

class W2V2RoPECache:
    src: torch.Tensor = None
    src_len: int = 0
    n_steps: int = 0
    max_steps: int = 0
    layers: List[LayerCache] = None

    def __init__(self, src=None, src_len=0, n_steps=0, max_steps=0, layers=None):
        self.src = src
        self.src_len = src_len
        self.n_steps = n_steps
        self.max_steps = max_steps
        self.layers = layers

class SpeechEncoderW2V2RoPE(L.LightningModule):
    def __init__(
        self, 
        w2v2_path, w2v2_ctc_finetuned,
        length_shrink_cfg=None,
        block_size=16, max_cache_size=125,
        llm_embedding=None,
        train_ds=None, dev_ds=None, train_bsz=None, dev_bsz=None, collate_fn=None,
        lr=1e-4, warmup_updates=0, min_lr=1e-6, temp=0.5, loss_fn='waco'
    ):
        super().__init__()

        patch_w2v2(block_size)
        self.speech_encoder, s_dim, self.s_layer = self._load_w2v2(w2v2_path, w2v2_ctc_finetuned)
        self.max_cache_size = max_cache_size
        
        self.length_shrink = None
        if length_shrink_cfg is not None:
            self.length_shrink_cfg = eval(length_shrink_cfg)
            self.length_shrink = ConvFeatureExtractionModel(self.length_shrink_cfg, in_d=s_dim)
        self.proj = nn.Linear(s_dim, llm_embedding.embedding_dim)

        self.llm_embedding = llm_embedding
        self.llm_embedding.requires_grad_(False)

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

    def _load_w2v2(self, speech_tower_path, ssl_finetuned):
        if not ssl_finetuned: # ssl model
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(speech_tower_path)
            w2v_args = state["args"]
            task = fairseq.tasks.setup_task(w2v_args)
            model = task.build_model(w2v_args)
            model.load_state_dict(state["model"], strict=True)
            speech_dimension = w2v_args.encoder_embed_dim
            n_layer = w2v_args.encoder_layers
        else: # ctc finetune, w2v-ctc
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(speech_tower_path)
            model = Wav2VecEncoder(state['cfg']['model'], None)
            new = {}
            for key in state['model'].keys():
                new_key = key.replace('w2v_encoder.', '')
                if not new_key.startswith('proj'):
                    new[new_key] = state['model'][key]
            model.load_state_dict(new, strict=False)
            model = model.w2v_model
            speech_dimension = state['cfg']['model']['w2v_args']['model'].encoder_embed_dim
            n_layer = state['cfg']['model']['w2v_args']['model'].encoder_layers            
        return model, speech_dimension, n_layer

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

        input_lengths = self.speech_encoder._get_feat_extract_output_lengths(input_lengths)
        if self.length_shrink:
            for cfg in self.length_shrink_cfg:
                input_lengths = _conv_out_length(
                    input_lengths, cfg[1], cfg[2]
                )

        return input_lengths.to(torch.long)
    
    def encode_speech(self, src_tokens, src_lens, cache=None):
        if cache is None:
            cache = W2V2RoPECache(
                max_steps=self.max_cache_size,
                layers=[LayerCache() for _ in range(self.s_layer)]
            )

        padding_mask = lengths_to_padding_mask(src_lens)
        res = self.speech_encoder.extract_features(src_tokens, padding_mask, cache=cache)
        feature, padding_mask = res["x"], res["padding_mask"]

        feature = self.length_shrink(feature.transpose(1, 2)).transpose(1, 2)
        feature = self.proj(feature)
        
        return feature, cache
    
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("val_loss", loss, batch_size=batch["src_speech_lengths"].sum() / 16000)
    
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
    
class SpeechEncoderHuBERTRope(SpeechEncoderW2V2RoPE):
    def _load_w2v2(self, speech_tower_path, ssl_finetuned):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [speech_tower_path], strict=False
        )
        model = models[0].w2v_encoder.w2v_model
        speech_dimension = cfg.model.w2v_args.model.encoder_embed_dim
        n_layer = cfg.model.w2v_args.model.encoder_layers         
        return model, speech_dimension, n_layer

    def encode_speech(self, src_tokens, src_lens, cache=None):
        if cache is None:
            cache = W2V2RoPECache(
                max_steps=self.max_cache_size,
                layers=[LayerCache() for _ in range(self.s_layer)]
            )

        padding_mask = lengths_to_padding_mask(src_lens)
        feature, padding_mask = self.speech_encoder.extract_features(src_tokens, padding_mask, cache=cache)

        feature = self.length_shrink(feature.transpose(1, 2)).transpose(1, 2)
        feature = self.proj(feature)
        
        return feature, cache