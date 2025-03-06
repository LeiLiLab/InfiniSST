from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from train.dataset import SpeechSampler

import fairseq
from fairseq.modules import TransposeLast
from fairseq.models.wav2vec import Wav2VecEncoder, Wav2Vec2Model
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
)

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
        llm_embedding_dim=4096, llm_embedding=None, xpos=True, rope=True,
        train_ds=None, dev_ds=None, train_bsz=None, dev_bsz=None, collate_fn=None,
        lr=1e-4, warmup_updates=0, min_lr=1e-6, temp=0.5, loss_fn='waco' 
    ):
        super().__init__()

        self.speech_encoder, s_dim, self.s_layer = self._load_w2v2(
            w2v2_path, w2v2_ctc_finetuned
        )
        self.blocksize = block_size
        self.max_cache_size = max_cache_size
        
        self.length_shrink = None
        if length_shrink_cfg is not None:
            self.length_shrink_cfg = eval(length_shrink_cfg)
            self.length_shrink = ConvFeatureExtractionModel(self.length_shrink_cfg, in_d=s_dim)
        self.proj = nn.Linear(s_dim, llm_embedding_dim)

        self.llm_embedding = llm_embedding
        if llm_embedding:
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

    def set_blocksize(self, multiplier):
        self.speech_encoder.blocksize = self.blocksize * multiplier
        self.speech_encoder.encoder.blocksize = self.blocksize * multiplier

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

            cfg = state['cfg']['model']['w2v_args']['model']       
            speech_dimension = cfg.encoder_embed_dim
            n_layer = cfg.encoder_layers     

            model = Wav2VecEncoder(state['cfg']['model'], None)
            new = {}
            for key in state['model'].keys():
                new_key = key.replace('w2v_encoder.', '')
                if not new_key.startswith('proj'):
                    new[new_key] = state['model'][key]
            model.load_state_dict(new, strict=False)
            model = model.w2v_model
                   
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

        input_lengths = Wav2Vec2Model._get_feat_extract_output_lengths(self.speech_encoder, input_lengths)
        if self.length_shrink:
            for cfg in self.length_shrink_cfg:
                input_lengths = _conv_out_length(
                    input_lengths, cfg[1], cfg[2]
                )

        return input_lengths.to(torch.long)
    
    def encode_speech(self, src_tokens, src_lens=None, cache=None):
        if cache is None:
            cache = W2V2RoPECache(
                max_steps=self.max_cache_size,
                layers=[LayerCache() for _ in range(self.s_layer)]
            )

        if src_lens is None:
            res = self.speech_encoder.extract_features(src_tokens, cache=cache)
        else:
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