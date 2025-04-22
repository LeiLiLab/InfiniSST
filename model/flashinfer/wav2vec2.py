# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    RelPositionalEncoding,
    SamePad,
    TransposeLast,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor

from model.flashinfer.wav2vec2_mha import MultiheadAttention
from model.flashinfer.engine import (
    SpeechCache,
    allocate_paged_kv_cache, 
    PAGE_SIZE,
)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer"])


@dataclass
class Wav2Vec2Config(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})


@register_model("wav2vec2_flashinfer", dataclass=Wav2Vec2Config)
class Wav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        encoder_cls = TransformerEncoder

        self.encoder = encoder_cls(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        requests,
        pagetable,
    ):    
        sources = []
        for request in requests:
            if request['cache'] is None:
                request['cache'] = SpeechCache()
            cache = request['cache']
            if cache.src is not None:
                cache.src = torch.cat([cache.src, request['speech']], dim=0)
            else:
                cache.src = request['speech']
            sources.append(cache.src)
        source = torch.stack(sources, dim=0)

        with torch.no_grad():
            features = self.feature_extractor(source)

        feature_batch = []
        for i, request in enumerate(requests):
            cache = request['cache']
            blocksize = request['blocksize']
            if cache.src_len > 0:
                feature = features[i, :, blocksize : 2 * blocksize]
                max_src_token_len = 79 + 320 + 320 * blocksize
                cache.src = cache.src[-max_src_token_len:]
            else:
                feature = features[i, :, : blocksize]
            cache.src_len = blocksize
            feature_batch.append(feature)

        features = torch.cat(feature_batch, dim=-1).transpose(0, 1)        
        features = self.layer_norm(features)
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x = features

        # wrapper
        paged_kv_indices = []
        paged_kv_indptr = [0]
        paged_kv_last_page_len = []
        qo_indptr = [0]
        for request in requests:
            pagetable.paged_queue, request['cache'].paged_kv_indices, request['cache'].paged_kv_last_page_len = \
                allocate_paged_kv_cache(
                    pagetable.paged_queue,
                    request['cache'].paged_kv_indices,
                    request['cache'].paged_kv_last_page_len,
                    request['blocksize']
                )
            paged_kv_indices.extend(request['cache'].paged_kv_indices)
            paged_kv_indptr.append(len(request['cache'].paged_kv_indices))
            paged_kv_last_page_len.append(request['cache'].paged_kv_last_page_len)
            qo_indptr.append(request['blocksize'])

        paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32, device=x.device)

        paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32, device=x.device)
        paged_kv_indptr = paged_kv_indptr.cumsum(dim=0, dtype=torch.int32)

        paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device=x.device)

        qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=x.device)
        qo_indptr = qo_indptr.cumsum(dim=0, dtype=torch.int32)

        pagetable.wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.encoder.layers[0].self_attn.num_heads,
            self.encoder.layers[0].self_attn.num_heads,
            self.encoder.layers[0].self_attn.head_dim,
            PAGE_SIZE,
            pos_encoding_mode='ROPE_LLAMA',
            rope_scale=1.0,
            rope_theta=1e4,
            q_data_type=x.dtype,
            kv_data_type=x.dtype,
        )

        x, pagetable, layer_results = self.encoder(
            x,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            pagetable,
        )

        return x, requests, pagetable, layer_results

    def remove_pretraining_modules(self, last_layer=None):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

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
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args: Wav2Vec2Config):
        assert args.layer_type == "transformer"
        layer = TransformerSentenceEncoderLayer(
            embedding_dim=self.embedding_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=self.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            activation_fn=args.activation_fn,
            layer_norm_first=args.layer_norm_first,
        )
        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args: Wav2Vec2Config):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(
        self, 
        x, 
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        pagetable,
    ):
        x, pagetable, layer_results = self.extract_features(
            x, 
            qo_indptr,
            paged_kv_indptr, 
            paged_kv_indices, 
            paged_kv_last_page_len,
            pagetable,
        )

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, pagetable, layer_results

    def extract_features(
        self,
        x, # B x T x C
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        pagetable,
    ):
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = [(x, None, None)]
        r = None
        for i, layer in enumerate(self.layers):
            x, (z, lr), pagetable = layer(
                x,
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                pagetable,
                i,
            )
            layer_results.append((x, z, lr))

        return x, pagetable, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        pagetable,
        layer_idx,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pagetable = self.self_attn(
                query=x,
                key=x,
                value=x,
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                pagetable=pagetable,
                layer_idx=layer_idx,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pagetable = self.self_attn(
                query=x,
                key=x,
                value=x,
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                pagetable=pagetable,
                layer_idx=layer_idx,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result), pagetable
