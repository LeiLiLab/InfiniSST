import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from train.dataset import SpeechSampler

import fairseq
from fairseq.modules import TransposeLast
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
)

from model.flashinfer.wav2vec2_asr import Wav2VecEncoder as Wav2VecEncoderFlash
from fairseq.models.wav2vec import Wav2VecEncoder

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
        llm_embedding_dim=4096, llm_embedding=None, rope=True,
        flash=False
    ):
        super().__init__()

        self.speech_encoder, s_dim, self.s_layer = self._load_w2v2(
            w2v2_path, w2v2_ctc_finetuned, flash
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

    def set_blocksize(self, multiplier):
        if type(multiplier) == int:
            self.speech_encoder.blocksize = self.blocksize * multiplier
            self.speech_encoder.encoder.blocksize = self.blocksize * multiplier
        elif type(multiplier) == list:
            self.speech_encoder.blocksize = [self.blocksize * m for m in multiplier]
            self.speech_encoder.encoder.blocksize = [self.blocksize * m for m in multiplier]
        else:
            raise ValueError(f"Invalid multiplier: {multiplier}")

    def _load_w2v2(self, speech_tower_path, ssl_finetuned, flash=False):
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

            if flash:
                print("Using flashinfer Wav2VecEncoder")
                model = Wav2VecEncoderFlash(state['cfg']['model'], None)
            else:
                print("Using fairseq Wav2VecEncoder")
                model = Wav2VecEncoder(state['cfg']['model'], None)
            new = {}
            for key in state['model'].keys():
                new_key = key.replace('w2v_encoder.', '')
                if not new_key.startswith('proj'):
                    new[new_key] = state['model'][key]
            model.load_state_dict(new, strict=False)
            model = model.w2v_model
                   
        return model, speech_dimension, n_layer
    
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
    
    def encode_speech_fast(self, requests, speech_pagetable):
        feature, requests, speech_pagetable, layer_results = self.speech_encoder(requests, speech_pagetable)
        feature = self.length_shrink(feature.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        feature = self.proj(feature)
        return feature, requests, speech_pagetable, layer_results