import os
import math
import torch
import torch.nn as nn
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Callable, Optional, Tuple, Union
import fairseq
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from fairseq.models.hubert.hubert import HubertModel
from fairseq.models.wav2vec import (
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
    Wav2Vec2Model,
    Wav2VecEncoder    
)
from fairseq.models.wav2vec.utils import pad_to_multiple
from fairseq.modules import GradMultiply
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.utils import index_put, is_xla_tensor
from fairseq import utils
from rotary_embedding_torch import RotaryEmbedding

from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention,
    apply_rotary_pos_emb, repeat_kv
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward
from transformers.cache_utils import StaticCache, DynamicCache, EncoderDecoderCache
from transformers.generation.beam_search import BeamSearchScorer, BeamScorer, BeamHypotheses

XPOS = True

def get_attn_mask_training(seq_len, max_cache_size=None, blocksize=1):
    blocksizes = [
        min(blocksize, seq_len - i * blocksize) 
        for i in range((seq_len + blocksize - 1) // blocksize)
    ]

    mask = torch.zeros(seq_len, seq_len, device='cuda', dtype=torch.bool)
    start_idx = 0
    for block_size in blocksizes:
        end_idx = start_idx + block_size
        mask[start_idx : end_idx, :end_idx] = 1
        start_idx = end_idx
    
    if max_cache_size is not None:
        for i in range(seq_len):
            mask[i, : max(0, i - max_cache_size)] = 0

    mask_num = torch.zeros_like(mask, dtype=torch.float)
    mask_num.masked_fill_(~mask, float('-inf'))
    
    return mask_num

def get_attn_mask_inference(seq_len, prefix_len, max_cache_size, blocksize=1):
    max_len = seq_len + min(prefix_len, max_cache_size)

    blocksizes = [
        min(blocksize, seq_len + prefix_len - i * blocksize) 
        for i in range((seq_len + prefix_len + blocksize - 1) // blocksize)
    ]

    mask = torch.zeros(seq_len, max_len, device='cuda', dtype=torch.bool)
    start_idx = 0
    for block_size in blocksizes:
        end_idx = start_idx + block_size
        if end_idx > prefix_len:
            mask[
                max(0, start_idx - prefix_len) : end_idx - prefix_len,
                : end_idx - max(0, prefix_len - max_cache_size)
            ] = 1
        start_idx = end_idx
    
    for i in range(seq_len):
        mask[i, : max(0, i + prefix_len - max_cache_size) - max(0, prefix_len - max_cache_size)] = 0

    mask_num = torch.zeros_like(mask, dtype=torch.float)
    mask_num.masked_fill_(~mask, float('-inf'))
    
    return mask_num


def uni_hubert_extract_features(
    self,
    source: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    mask: bool = False,
    ret_conv: bool = False,
    output_layer: Optional[int] = None,
    cache=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    res = self.forward(
        source,
        padding_mask=padding_mask,
        mask=mask,
        features_only=True,
        output_layer=output_layer,
        cache=cache,
    )
    feature = res["features"] if ret_conv else res["x"]
    return feature, res["padding_mask"]


def uni_hubert_forward(
    self,
    source: torch.Tensor,
    target_list: Optional[List[torch.Tensor]] = None,
    padding_mask: Optional[torch.Tensor] = None,
    mask: bool = True,
    features_only: bool = False,
    output_layer: Optional[int] = None,
    cache=None,
) -> Dict[str, torch.Tensor]:
    """output layer is 1-based"""

    if cache.src is not None:
        source = torch.cat([cache.src, source], dim=1)
    cache.src = source

    features = self.forward_features(source)
    if target_list is not None:
        features, target_list = self.forward_targets(features, target_list)

    if cache.src_len > 0:
        new_src_len = features.size(-1)
        features = features[..., cache.src_len:]
        cache.src_len = new_src_len

        max_src_token_len = 79 + 320 + 320 * self.blocksize
        if cache.src.size(1) > max_src_token_len:
            cache.src = cache.src[:, -max_src_token_len:]
            cache.src_len = self.blocksize
    else:
        cache.src_len = features.size(-1)

    features_pen = features.float().pow(2).mean()

    features = features.transpose(1, 2)
    features = self.layer_norm(features)
    unmasked_features = features.clone()

    if padding_mask is not None:
        padding_mask = self.forward_padding_mask(features, padding_mask)

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    features = self.dropout_input(features)
    unmasked_features = self.dropout_features(unmasked_features)

    if mask:
        x, mask_indices = self.apply_mask(features, padding_mask, target_list)
    else:
        x = features
        mask_indices = None

    # feature: (B, T, D), float
    # target: (B, T), long
    # x: (B, T, D), float
    # padding_mask: (B, T), bool
    # mask_indices: (B, T), bool
    x, _ = self.encoder(
        x,
        padding_mask=padding_mask,
        layer=None if output_layer is None else output_layer - 1,
        cache=cache,
    )

    if features_only:
        return {"x": x, "padding_mask": padding_mask, "features": features}

    def compute_pred(proj_x, target, label_embs):
        # compute logits for the i-th label set
        y = torch.index_select(label_embs, 0, target.long())
        negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)
        # proj_x: (S, D)
        # y: (S, D)
        # negs: (Neg, S, D)
        return self.compute_nce(proj_x, y, negs)

    label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

    if not self.skip_masked:
        masked_indices = torch.logical_and(~padding_mask, mask_indices)
        proj_x_m = self.final_proj(x[masked_indices])
        if self.untie_final_proj:
            proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
        else:
            proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
        logit_m_list = [
            compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
            for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
        ]
    else:
        logit_m_list = [None for _ in target_list]

    if not self.skip_nomask:
        nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
        proj_x_u = self.final_proj(x[nomask_indices])
        if self.untie_final_proj:
            proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
        else:
            proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

        logit_u_list = [
            compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
            for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
        ]
    else:
        logit_u_list = [None for _ in target_list]

    result = {
        "logit_m_list": logit_m_list,
        "logit_u_list": logit_u_list,
        "padding_mask": padding_mask,
        "features_pen": features_pen,
    }
    return result



def uni_w2v2_extract_features(self, source, padding_mask=None, mask=False, layer=None, cache=None):
    res = self.forward(
        source, padding_mask, mask=mask, features_only=True, layer=layer, cache=cache,
    )
    return res

def uni_w2v2_forward(
    self,
    source,
    padding_mask=None,
    mask=True,
    features_only=False,
    layer=None,
    mask_indices=None,
    mask_channel_indices=None,
    padding_count=None,
    cache=None,
):
    
    if cache.src is not None:
        source = torch.cat([cache.src, source], dim=1)
    cache.src = source

    if self.feature_grad_mult > 0:
        features = self.feature_extractor(source)
        if self.feature_grad_mult != 1.0:
            features = GradMultiply.apply(features, self.feature_grad_mult)
    else:
        with torch.no_grad():
            features = self.feature_extractor(source)
    
    # logger.info(f"w2v2 forward: device {features.device}, blocksize {self.blocksize}")
    if cache.src_len > 0:
        new_src_len = features.size(-1)
        features = features[..., cache.src_len:]
        cache.src_len = new_src_len

        max_src_token_len = 79 + 320 + 320 * self.blocksize
        if cache.src.size(1) > max_src_token_len:
            cache.src = cache.src[:, -max_src_token_len:]
            cache.src_len = self.blocksize
    else:
        cache.src_len = features.size(-1)

    features_pen = features.float().pow(2).mean()

    features = features.transpose(1, 2)
    features = self.layer_norm(features)
    unmasked_features = features.clone()

    if padding_mask is not None and padding_mask.any():
        input_lengths = (1 - padding_mask.long()).sum(-1)
        # apply conv formula to get real output_lengths
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)

        padding_mask = torch.zeros(
            features.shape[:2], dtype=features.dtype, device=features.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        padding_mask[
            (
                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,
            )
        ] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
    else:
        padding_mask = None

    time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
    if time_steps_to_drop != 0:
        features = features[:, :-time_steps_to_drop]
        unmasked_features = unmasked_features[:, :-time_steps_to_drop]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :-time_steps_to_drop]

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    features = self.dropout_input(features)
    unmasked_features = self.dropout_features(unmasked_features)

    num_vars = None
    code_ppl = None
    prob_ppl = None
    curr_temp = None

    if self.input_quantizer:
        q = self.input_quantizer(features, produce_targets=False)
        features = q["x"]
        num_vars = q["num_vars"]
        code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]
        features = self.project_inp(features)

    if mask:
        x, mask_indices = self.apply_mask(
            features,
            padding_mask,
            mask_indices=mask_indices,
            mask_channel_indices=mask_channel_indices,
        )
        if not is_xla_tensor(x) and mask_indices is not None:
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            y = unmasked_features[mask_indices].view(
                unmasked_features.size(0), -1, unmasked_features.size(-1)
            )
        else:
            y = unmasked_features
    else:
        x = features
        y = unmasked_features
        mask_indices = None

    x, layer_results = self.encoder(
        x, 
        padding_mask=padding_mask, 
        layer=layer, 
        cache=cache,
    )

    if features_only:
        return {
            "x": x,
            "padding_mask": padding_mask,
            "features": unmasked_features,
            "layer_results": layer_results,
        }

    if self.quantizer:
        if self.negatives_from_everywhere:
            q = self.quantizer(unmasked_features, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            y = self.project_q(y)

            negs, _ = self.sample_negatives(
                y,
                mask_indices[0].sum(),
                padding_count=padding_count,
            )
            y = y[mask_indices].view(y.size(0), -1, y.size(-1))

        else:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            negs, _ = self.sample_negatives(
                y,
                y.size(1),
                padding_count=padding_count,
            )

        if self.codebook_negatives > 0:
            cb_negs = self.quantizer.sample_from_codebook(
                y.size(0) * y.size(1), self.codebook_negatives
            )
            cb_negs = cb_negs.view(
                self.codebook_negatives, y.size(0), y.size(1), -1
            )  # order doesnt matter
            cb_negs = self.project_q(cb_negs)
            negs = torch.cat([negs, cb_negs], dim=0)
    else:
        y = self.project_q(y)

        if self.negatives_from_everywhere:
            negs, _ = self.sample_negatives(
                unmasked_features,
                y.size(1),
                padding_count=padding_count,
            )
            negs = self.project_q(negs)
        else:
            negs, _ = self.sample_negatives(
                y,
                y.size(1),
                padding_count=padding_count,
            )

    if not is_xla_tensor(x):
        # tpu-comment: reducing the size in a dynamic way causes
        # too many recompilations on xla.
        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

    if self.target_glu:
        y = self.target_glu(y)
        negs = self.target_glu(negs)

    x = self.final_proj(x)
    x = self.compute_preds(x, y, negs)

    result = {
        "x": x,
        "padding_mask": padding_mask,
        "features_pen": features_pen,
    }

    if prob_ppl is not None:
        result["prob_perplexity"] = prob_ppl
        result["code_perplexity"] = code_ppl
        result["num_vars"] = num_vars
        result["temp"] = curr_temp

    return result

def uni_transformer_encoder_forward(self, x, padding_mask=None, layer=None, cache=None):
    x, layer_results = self.extract_features(x, padding_mask, layer, cache=cache)

    if self.layer_norm_first and layer is None:
        x = self.layer_norm(x)

    return x, layer_results

def sinusoidal_positional_embedding(offset, length, d_model, device):
    half_dim = d_model // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.bfloat16, device=device) * -emb)
    emb = torch.arange(offset, offset + length, dtype=torch.bfloat16, device=device).unsqueeze(
        1
    ) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
        length, -1
    )
    if d_model % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(length, 1)], dim=1)
    return emb


def uni_transformer_encoder_extract_features(
    self,
    x,
    padding_mask=None,
    tgt_layer=None,
    min_layer=0,
    cache=None
):
    if padding_mask is not None:
        x = index_put(x, padding_mask, 0)

    # pad to the sequence length dimension
    x, pad_length = pad_to_multiple(
        x, self.required_seq_len_multiple, dim=-2, value=0
    )

    if pad_length > 0 and padding_mask is None:
        padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
        padding_mask[:, -pad_length:] = True
    else:
        padding_mask, _ = pad_to_multiple(
            padding_mask, self.required_seq_len_multiple, dim=-1, value=True
        )

    if not ROPE:
        pos_emb = sinusoidal_positional_embedding(
            cache.n_steps, x.size(1), x.size(2), x.device
        )
        # logger.info(f"pos_emb: {pos_emb.shape}, x: {x.shape}")
        x = x + pos_emb

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    prefix_len = cache.n_steps
    seq_len = x.size(0)
    # logger.info(f"w2v2 enc forward: device {x.device}, blocksize {self.blocksize}")
    if prefix_len > 0:
        attn_mask = get_attn_mask_inference(seq_len, prefix_len, cache.max_steps, self.blocksize)
    else:
        attn_mask = get_attn_mask_training(seq_len, cache.max_steps, self.blocksize)

    layer_results = []
    r = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random() if self.layerdrop > 0 else 1
        if not self.training or (dropout_probability > self.layerdrop):
            if cache.layers[i].k is not None:
                cache.layers[i].k = cache.layers[i].k[:, -cache.max_steps:]
                cache.layers[i].v = cache.layers[i].v[:, -cache.max_steps:]
                if cache.layers[i].key_padding_mask is not None:
                    cache.layers[i].key_padding_mask = cache.layers[i].key_padding_mask[:, -cache.max_steps:]
            x, (z, lr) = layer(
                x, 
                self_attn_mask=attn_mask,
                self_attn_padding_mask=padding_mask, 
                need_weights=False, cache=cache.layers[i],
            )
            if i >= min_layer:
                layer_results.append((x, z, lr))
        if i == tgt_layer:
            r = x
            break

    cache.n_steps += seq_len

    if r is not None:
        x = r

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    # undo paddding
    if pad_length > 0:
        x = x[:, :-pad_length]

        def undo_pad(a, b, c):
            return (
                a[:-pad_length],
                b[:-pad_length] if b is not None else b,
                c[:-pad_length],
            )

        layer_results = [undo_pad(*u) for u in layer_results]

    return x, layer_results

def uni_self_attn_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor = None,
    self_attn_padding_mask: torch.Tensor = None,
    need_weights: bool = False,
    att_args=None,
    cache=None,
):
    """
    LayerNorm is applied either before or after the self-attention/ffn
    modules similar to the original Transformer imlementation.
    """
    residual = x

    assert self.layer_norm_first
    x = self.self_attn_layer_norm(x)

    x, attn = self.self_attn(
        query=x,
        key=x,
        value=x,
        key_padding_mask=self_attn_padding_mask,
        attn_mask=self_attn_mask,
        cache=cache,
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

    return x, (attn, layer_result)


def uni_mha_init(
    self,
    embed_dim,
    num_heads,
    kdim=None,
    vdim=None,
    dropout=0.0,
    bias=True,
    add_bias_kv=False,
    add_zero_attn=False,
    self_attention=False,
    encoder_decoder_attention=False,
    q_noise=0.0,
    qn_block_size=8,
    # TODO: pass in config rather than string.
    # config defined in xformers.components.attention.AttentionConfig
    xformers_att_config: Optional[str] = None,
    xformers_blocksparse_layout: Optional[
        torch.Tensor
    ] = None,  # This should be part of the config
    xformers_blocksparse_blocksize: Optional[
        int
    ] = 16,  # This should be part of the config
    max_batch_size: Optional[
        int
    ] = 8,
    max_seq_len: Optional[
        int
    ] = 1024,
):
    super(MultiheadAttention, self).__init__()
    
    self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, use_xpos=XPOS)

    xformers_att_config = utils.eval_str_dict(xformers_att_config)
    self.use_xformers = xformers_att_config is not None
    self.embed_dim = embed_dim
    self.kdim = kdim if kdim is not None else embed_dim
    self.vdim = vdim if vdim is not None else embed_dim
    self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

    self.num_heads = num_heads
    self.dropout_module = FairseqDropout(
        dropout, module_name=self.__class__.__name__
    )

    self.head_dim = embed_dim // num_heads
    assert (
        self.head_dim * num_heads == self.embed_dim
    ), "embed_dim must be divisible by num_heads"
    self.scaling = self.head_dim**-0.5

    self.self_attention = self_attention
    self.encoder_decoder_attention = encoder_decoder_attention

    assert not self.self_attention or self.qkv_same_dim, (
        "Self-attention requires query, key and " "value to be of the same size"
    )

    self.k_proj = quant_noise(
        nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
    )
    self.v_proj = quant_noise(
        nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
    )
    self.q_proj = quant_noise(
        nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
    )

    self.out_proj = quant_noise(
        nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
    )

    if add_bias_kv:
        self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
        self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
    else:
        self.bias_k = self.bias_v = None

    self.add_zero_attn = add_zero_attn
    self.beam_size = 1
    self.reset_parameters()

    if self.use_xformers:
        raise NotImplementedError

    self.onnx_trace = False
    self.skip_embed_dim_check = False

    self.max_batch_size = max_batch_size
    self.max_seq_len = max_seq_len


def uni_mha_forward(
    self,
    query,
    key: Optional[Tensor],
    value: Optional[Tensor],
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    static_kv: bool = False,
    attn_mask: Optional[Tensor] = None,
    before_softmax: bool = False,
    need_head_weights: bool = False,
    cache=None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Input shape: Time x Batch x Channel

    Args:
        key_padding_mask (ByteTensor, optional): mask to exclude
            keys that are pads, of shape `(batch, src_len)`, where
            padding elements are indicated by 1s.
        need_weights (bool, optional): return the attention weights,
            averaged over heads (default: False).
        attn_mask (ByteTensor, optional): typically used to
            implement causal attention, where the mask prevents the
            attention from looking forward in time (default: None).
        before_softmax (bool, optional): return the raw attention
            weights and values before the attention softmax.
        need_head_weights (bool, optional): return the attention
            weights for each head. Implies *need_weights*. Default:
            return the average attention weights over all heads.
    """
    if need_head_weights:
        need_weights = True

    is_tpu = query.device.type == "xla"

    tgt_len, bsz, embed_dim = query.size()
    src_len = tgt_len
    if not self.skip_embed_dim_check:
        assert (
            embed_dim == self.embed_dim
        ), f"query dim {embed_dim} != {self.embed_dim}"
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    if key is not None:
        src_len, key_bsz, _ = key.size()
        if not torch.jit.is_scripting():
            assert value is not None
            assert src_len, key_bsz == value.shape[:2]

    if self.self_attention:
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
    elif self.encoder_decoder_attention:
        # encoder-decoder attention
        q = self.q_proj(query)
        if key is None:
            assert value is None
            k = v = None
        else:
            if self.beam_size > 1 and bsz == key.size(1):
                # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                    :, :, 0, :
                ]
                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask.view(
                        -1, self.beam_size, key_padding_mask.size(1)
                    )[:, 0, :]
            k = self.k_proj(key)
            v = self.v_proj(key)

    else:
        assert key is not None and value is not None
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
    q *= self.scaling

    if self.bias_k is not None:
        assert self.bias_v is not None
        k, v, attn_mask, key_padding_mask = self._add_bias(
            k, v, attn_mask, key_padding_mask, bsz
        )

    q = (
        q.contiguous()
        .view(tgt_len, bsz * self.num_heads, self.head_dim)
        .transpose(0, 1)
    )

    kv_bsz = bsz  # need default value for scripting
    if k is not None:
        kv_bsz = k.size(1)
        k = (
            k.contiguous()
            .view(-1, kv_bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
    if v is not None:
        v = (
            v.contiguous()
            .view(-1, kv_bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )    

    if cache.k is not None:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)

        cache.k = cache.k.to(q)
        cache.v = cache.v.to(q)

        cache.k = torch.cat([cache.k, k], dim=1)
        cache.v = torch.cat([cache.v, v], dim=1)

        k, v = cache.k, cache.v

        src_len = k.size(1)

        assert k is not None and v is not None

        key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
            key_padding_mask=key_padding_mask,
            prev_key_padding_mask=cache.key_padding_mask,
            batch_size=kv_bsz,
            src_len=k.size(1),
            static_kv=static_kv,
        )
        cache.key_padding_mask = key_padding_mask
    else:
        cache.k, cache.v = k, v
    
    if ROPE:
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

    assert k is not None
    assert k.size(1) == src_len

    # This is part of a workaround to get around fork/join parallelism
    # not supporting Optional types.
    if key_padding_mask is not None and key_padding_mask.dim() == 0:
        key_padding_mask = None

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == kv_bsz
        assert key_padding_mask.size(1) == src_len

    if self.add_zero_attn:
        assert v is not None
        src_len += 1
        k, v, key_padding_mask, attn_mask = self._append_zero_attn(
            k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )

    if self.encoder_decoder_attention and bsz != kv_bsz:
        attn_weights = torch.einsum(
            "bxhtd,bhsd->bxhts",
            q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]),
            k.view((kv_bsz, self.num_heads) + k.size()[1:]),
        )
        attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
    else:
        attn_weights = torch.bmm(q, k.transpose(1, 2))
    attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

    assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        if self.onnx_trace:
            attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
        attn_weights += attn_mask

    if key_padding_mask is not None:
        # don't attend to padding symbols
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if not is_tpu:
            attn_weights = attn_weights.view(
                kv_bsz, -1, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .to(torch.bool),
                float("-inf"),
            )
        else:
            attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
            attn_weights = attn_weights.transpose(0, 2)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if before_softmax:
        return attn_weights, v

    attn_weights_float = utils.softmax(
        attn_weights, dim=-1, onnx_trace=self.onnx_trace
    )
    attn_weights = attn_weights_float.type_as(attn_weights)
    attn_probs = self.dropout_module(attn_weights)

    assert v is not None
    if self.encoder_decoder_attention and bsz != kv_bsz:
        attn = torch.einsum(
            "bxhts,bhsd->bxhtd",
            attn_probs.view(
                (
                    kv_bsz,
                    -1,
                    self.num_heads,
                )
                + attn_probs.size()[1:]
            ),
            v.view(
                (
                    kv_bsz,
                    self.num_heads,
                )
                + v.size()[1:]
            ),
        )
        attn = attn.reshape((-1,) + attn.size()[-2:])
    else:
        attn = torch.bmm(attn_probs, v)
    assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
    if self.onnx_trace and attn.size(1) == 1:
        # when ONNX tracing a single decoder step (sequence length == 1)
        # the transpose is a no-op copy before view, thus unnecessary
        attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
    else:
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
    attn = self.out_proj(attn)
    attn_weights: Optional[Tensor] = None
    if need_weights:
        attn_weights = attn_weights_float.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)
        if not need_head_weights:
            # average attention weights over heads
            attn_weights = attn_weights.mean(dim=0)

    return attn, attn_weights


def llama_attention_new_forward(self, *args, **kwargs):
    """
    Modified LlamaAttention forward method that stores unrotated key/value states in the cache.
    The rotation is applied after retrieving from cache instead of before storing.
    
    This modification changes the caching behavior to:
    1. Store unrotated key/value states in the cache
    2. Apply rotation after retrieving from cache, using the correct positional 
       embeddings for both new and cached keys
    
    This allows for more flexible position-based rotations during inference
    since the original unrotated states are preserved.
    """
    # Extract relevant arguments
    hidden_states = kwargs.get('hidden_states', args[0] if args else None)
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None) 
    past_key_value = kwargs.get('past_key_value', None)
    output_attentions = kwargs.get('output_attentions', False)
    use_cache = kwargs.get('use_cache', False)
    cache_position = kwargs.get('cache_position', None)
    position_embeddings = kwargs.get('position_embeddings', None)
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    # First update cache with unrotated key/value states
    unrotated_key_states = key_states.clone()
    if past_key_value is not None:
        # Store unrotated keys in cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(unrotated_key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Get the total sequence length including cached tokens
        total_seq_len = key_states.size(-2)  # Use actual size after cache update
        past_seq_len = total_seq_len - q_len
        
        key_position_ids = torch.arange(total_seq_len, device=cos.device)
        query_position_ids = torch.arange(past_seq_len, total_seq_len, device=cos.device)
        
        # Get rotary embeddings for queries and keys separately
        key_cos, key_sin = self.rotary_emb(value_states, key_position_ids.unsqueeze(0))
        query_cos, query_sin = self.rotary_emb(value_states, query_position_ids.unsqueeze(0))
        
        # Apply rotation with appropriate position embeddings
        query_states = apply_rotary_pos_emb(query_states, query_states, query_cos, query_sin)[0]
        key_states = apply_rotary_pos_emb(key_states, key_states, key_cos, key_sin)[1]
    else:
        # For the first token, just apply rotation normally
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_flash_attention_2_new_forward(self, *args, **kwargs):
    hidden_states = kwargs.pop('hidden_states', args[0] if args else None)
    attention_mask = kwargs.pop('attention_mask', None)
    position_ids = kwargs.pop('position_ids', None) 
    past_key_value = kwargs.pop('past_key_value', None)
    output_attentions = kwargs.pop('output_attentions', False)
    use_cache = kwargs.pop('use_cache', False)
    cache_position = kwargs.pop('cache_position', None)
    position_embeddings = kwargs.pop('position_embeddings', None)

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    # First update cache with unrotated key/value states
    unrotated_key_states = key_states.clone()
    if past_key_value is not None:
        # Store unrotated keys in cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(unrotated_key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Get the total sequence length including cached tokens
        total_seq_len = key_states.size(-2)  # Use actual size after cache update
        past_seq_len = total_seq_len - q_len
        
        key_position_ids = torch.arange(total_seq_len, device=cos.device)
        query_position_ids = torch.arange(past_seq_len, total_seq_len, device=cos.device)
        
        # Get rotary embeddings for queries and keys separately
        key_cos, key_sin = self.rotary_emb(value_states, key_position_ids.unsqueeze(0))
        query_cos, query_sin = self.rotary_emb(value_states, query_position_ids.unsqueeze(0))
        
        # Apply rotation with appropriate position embeddings
        query_states = apply_rotary_pos_emb(query_states, query_states, query_cos, query_sin)[0]
        key_states = apply_rotary_pos_emb(key_states, key_states, key_cos, key_sin)[1]
    else:
        # For the first token, just apply rotation normally
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_sdpa_attention_new_forward(self, *args, **kwargs):
    hidden_states = kwargs.get('hidden_states', args[0] if args else None)
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None) 
    past_key_value = kwargs.get('past_key_value', None)
    output_attentions = kwargs.get('output_attentions', False)
    use_cache = kwargs.get('use_cache', False)
    cache_position = kwargs.get('cache_position', None)
    position_embeddings = kwargs.get('position_embeddings', None)

    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super(LlamaSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    # First update cache with unrotated key/value states
    unrotated_key_states = key_states.clone()
    if past_key_value is not None:
        # Store unrotated keys in cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(unrotated_key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Get the total sequence length including cached tokens
        total_seq_len = key_states.size(-2)  # Use actual size after cache update
        past_seq_len = total_seq_len - q_len
        
        key_position_ids = torch.arange(total_seq_len, device=cos.device)
        query_position_ids = torch.arange(past_seq_len, total_seq_len, device=cos.device)
        
        # Get rotary embeddings for queries and keys separately
        key_cos, key_sin = self.rotary_emb(value_states, key_position_ids.unsqueeze(0))
        query_cos, query_sin = self.rotary_emb(value_states, query_position_ids.unsqueeze(0))
        
        # Apply rotation with appropriate position embeddings
        query_states = apply_rotary_pos_emb(query_states, query_states, query_cos, query_sin)[0]
        key_states = apply_rotary_pos_emb(key_states, key_states, key_cos, key_sin)[1]
    else:
        # For the first token, just apply rotation normally
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value



def beam_search_process(
    self,
    input_ids: torch.LongTensor,
    next_scores: torch.FloatTensor,
    next_tokens: torch.LongTensor,
    next_indices: torch.LongTensor,
    pad_token_id: Optional[Union[int, torch.Tensor]] = None,
    eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
    beam_indices: Optional[torch.LongTensor] = None,
    group_index: Optional[int] = 0,
    decoder_prompt_len: Optional[int] = 0,
    past_key_values: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # add up to the length which the next_scores is calculated on (including decoder prompt)
    cur_len = input_ids.shape[-1] + 1
    batch_size = len(self._beam_hyps) // self.num_beam_groups

    if not (batch_size == (input_ids.shape[0] // self.group_size)):
        if self.num_beam_groups > 1:
            raise ValueError(
                f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                f"size of {self.group_size} is expected by the beam scorer."
            )
        else:
            raise ValueError(
                f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                f"{self.group_size} is expected by the beam scorer."
            )

    device = input_ids.device
    next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
    next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
    next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

    if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id = torch.tensor(eos_token_id)

    for batch_idx in range(batch_size):
        batch_group_idx = batch_idx * self.num_beam_groups + group_index
        if self._done[batch_group_idx]:
            if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
            if eos_token_id is None or pad_token_id is None:
                raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
            # pad the batch
            next_beam_scores[batch_idx, :] = 0
            next_beam_tokens[batch_idx, :] = pad_token_id
            next_beam_indices[batch_idx, :] = 0
            continue

        # next tokens for this sentence
        beam_idx = 0
        for beam_token_rank, (next_token, next_score, next_index) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
        ):
            batch_beam_idx = batch_idx * self.group_size + next_index
            # add to generated hypotheses if end of sentence
            if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                if is_beam_token_worse_than_top_num_beams:
                    continue
                if beam_indices is not None:
                    beam_index = beam_indices[batch_beam_idx]
                    beam_index = beam_index + (batch_beam_idx,)
                else:
                    beam_index = None

                if past_key_values is not None:
                    cur_past_key_values = DynamicCache(len(past_key_values))
                    for i, (k, v) in enumerate(past_key_values):
                        cur_past_key_values.update(
                            k[batch_beam_idx : batch_beam_idx + 1], 
                            v[batch_beam_idx : batch_beam_idx + 1], 
                            i
                        )

                self._beam_hyps[batch_group_idx].add(
                    input_ids[batch_beam_idx].clone(),
                    next_score.item(),
                    beam_indices=beam_index,
                    generated_len=cur_len - decoder_prompt_len,
                    past_key_values=cur_past_key_values
                )
            else:
                # add next predicted token since it is not eos_token
                next_beam_scores[batch_idx, beam_idx] = next_score
                next_beam_tokens[batch_idx, beam_idx] = next_token
                next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                beam_idx += 1

            # once the beam for next step is full, don't add more tokens to it.
            if beam_idx == self.group_size:
                break

        if beam_idx < self.group_size:
            raise ValueError(
                f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
            )

        # Check if we are done so that we can save a pad step if all(done)
        self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
        )

    return UserDict(
        {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
        }
    )

def beam_search_finalize(
    self,
    input_ids: torch.LongTensor,
    final_beam_scores: torch.FloatTensor,
    final_beam_tokens: torch.LongTensor,
    final_beam_indices: torch.LongTensor,
    max_length: int,
    pad_token_id: Optional[Union[int, torch.Tensor]] = None,
    eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
    beam_indices: Optional[torch.LongTensor] = None,
    decoder_prompt_len: Optional[int] = 0,
    past_key_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.LongTensor]:
    batch_size = len(self._beam_hyps) // self.num_beam_groups

    if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id = torch.tensor(eos_token_id)

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
        if self._done[batch_group_idx]:
            continue

        # all open beam hypotheses are added to the beam hypothesis
        # beam hypothesis class automatically keeps the best beams
        for index_per_group in range(self.group_size):
            batch_beam_idx = batch_group_idx * self.group_size + index_per_group
            final_score = final_beam_scores[batch_beam_idx].item()
            final_tokens = input_ids[batch_beam_idx]
            beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
            generated_len = final_tokens.shape[-1] - decoder_prompt_len

            if past_key_values is not None:
                cur_past_key_values = DynamicCache(len(past_key_values))
                for i, (k, v) in enumerate(past_key_values):
                    cur_past_key_values.update(
                        k[batch_beam_idx : batch_beam_idx + 1], 
                        v[batch_beam_idx : batch_beam_idx + 1], 
                        i
                    )

            beam_hyp.add(
                final_tokens, 
                final_score, 
                beam_indices=beam_index, 
                generated_len=generated_len, 
                past_key_values=cur_past_key_values
            )

    # select the best hypotheses
    sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
    best = []
    best_indices = []
    best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
    best_kv_caches = []
    # retrieve best hypotheses
    for i in range(batch_size):
        beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
        candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
        sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
        for j in range(self.num_beam_hyps_to_keep):
            best_hyp_tuple = sorted_hyps.pop()
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            best_index = best_hyp_tuple[2]
            kv_cache = best_hyp_tuple[3]
            sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

            # append hyp to lists
            best.append(best_hyp)

            # append indices to list
            best_indices.append(best_index)

            best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
            best_kv_caches.append(kv_cache)
    
    # prepare for adding eos
    sent_lengths_max = sent_lengths.max().item() + 1
    sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
    decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

    if len(best_indices) > 0 and best_indices[0] is not None:
        indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
    else:
        indices = None

    # shorter batches are padded if needed
    if sent_lengths.min().item() != sent_lengths.max().item():
        if pad_token_id is None:
            raise ValueError("`pad_token_id` has to be defined")
        decoded.fill_(pad_token_id)

    if indices is not None:
        indices.fill_(-1)

    # fill with hypotheses and eos_token_id if the latter fits in
    for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
        decoded[i, : sent_lengths[i]] = hypo

        if indices is not None:
            indices[i, : len(best_idx)] = torch.tensor(best_idx)

        if sent_lengths[i] < sent_max_len:
            # inserting only the first eos_token_id
            decoded[i, sent_lengths[i]] = eos_token_id[0]

    return UserDict(
        {
            "sequences": decoded,
            "sequence_scores": best_scores,
            "beam_indices": indices,
            "past_key_values": best_kv_caches,
        }
    )


def beam_hypotheses_add(
    self,
    hyp: torch.LongTensor,
    sum_logprobs: float,
    beam_indices: Optional[torch.LongTensor] = None,
    generated_len: Optional[int] = None,
    past_key_values: Optional[torch.Tensor] = None,
):
    """
    Add a new hypothesis to the list.
    """
    if generated_len is not None:
        score = sum_logprobs / (generated_len**self.length_penalty)
    # This 'else' case exists for retrocompatibility
    else:
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

    if len(self) < self.num_beams or score > self.worst_score:
        self.beams.append((score, hyp, beam_indices, past_key_values))
        if len(self) > self.num_beams:
            sorted_next_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.beams)])
            del self.beams[sorted_next_scores[0][1]]
            self.worst_score = sorted_next_scores[1][0]
        else:
            self.worst_score = min(score, self.worst_score)


@staticmethod
def generation_mixin_expand_inputs_for_generation(
    expand_size: int = 1,
    is_encoder_decoder: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
    # Do not call torch.repeat_interleave if expand_size is 1 because it clones
    # the input tensor and thus requires more memory although no change is applied
    if expand_size == 1:
        return input_ids, model_kwargs

    def _expand_dict_for_generation(dict_to_expand):
        for key in dict_to_expand:
            if (
                key != "cache_position"
                and dict_to_expand[key] is not None
                and isinstance(dict_to_expand[key], torch.Tensor)
            ):
                dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
        return dict_to_expand

    if input_ids is not None:
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)

    model_kwargs = _expand_dict_for_generation(model_kwargs)
    if "past_key_values" in model_kwargs:
        for i, (k, v) in enumerate(model_kwargs["past_key_values"]):
            model_kwargs["past_key_values"].key_cache[i] = k.repeat_interleave(expand_size, dim=0)
            model_kwargs["past_key_values"].value_cache[i] = v.repeat_interleave(expand_size, dim=0)

    if is_encoder_decoder:
        if model_kwargs.get("encoder_outputs") is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

    return input_ids, model_kwargs

from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationMixin,
    GenerateOutput,
    GenerateBeamOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
    _split_model_inputs,
    stack_model_outputs,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)
import torch.distributed as dist
import inspect
import warnings

logger = logging.get_logger(__name__)

@torch.no_grad()
def generation_mixin_generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    encoder_input_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
            to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
            deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    # self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        model_kwargs["num_logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
    user_defined_cache = model_kwargs.get(cache_name)
    max_cache_length = generation_config.max_length
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=encoder_input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache


    if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and hasattr(result.past_key_values, "to_legacy_cache")
        and result.past_key_values.to_legacy_cache is not None
    ):
        # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
        should_convert_cache = generation_config.return_legacy_cache
        is_user_defined_cache = user_defined_cache is not None
        is_default_cache_type = (
            type(result.past_key_values) == DynamicCache  # noqa E721
            or (
                isinstance(result.past_key_values, EncoderDecoderCache)
                and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
                and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
            )
        )
        if not is_user_defined_cache and is_default_cache_type:
            logger.warning_once(
                "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                "keep returning the legacy format, please set `return_legacy_cache=True`."
            )
            should_convert_cache = True
        if should_convert_cache:
            result.past_key_values = result.past_key_values.to_legacy_cache()
    return result

def generation_mixin_beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`:
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    sequential = generation_config.low_memory
    do_sample = generation_config.do_sample

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # if sequential is True, split the input to batches of batch_size and run sequentially
        if sequential:
            if any(
                model_name in self.__class__.__name__.lower()
                for model_name in [
                    "fsmt",
                    "reformer",
                    "ctrl",
                    "gpt_bigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            inputs_per_sub_batches = _split_model_inputs(
                model_inputs,
                split_size=batch_size,
                full_batch_size=batch_beam_size,
                config=self.config.get_text_config(),
            )
            outputs_per_sub_batch = [
                self(**inputs_per_sub_batch, return_dict=True) for inputs_per_sub_batch in inputs_per_sub_batches
            ]

            outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())

        else:  # Unchanged original behavior
            outputs = self(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # .float() is needed to retain precision for later logits manipulations
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
        # non eos token per beam.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)
        else:
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
            )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
            past_key_values=model_kwargs.get("past_key_values", None),
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
        # (that way the memory peak does not include outputs.logits)
        del outputs

        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
        past_key_values=model_kwargs.get("past_key_values", None),
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=sequence_outputs["past_key_values"],
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=sequence_outputs["past_key_values"],
            )
    else:
        return sequence_outputs["sequences"]


def patch_w2v2(xpos=True, rope=True):
    global XPOS, ROPE
    print("Patching with xpos {}, rope {}".format(xpos, rope))
    XPOS = xpos
    ROPE = rope
    Wav2Vec2Model.extract_features = uni_w2v2_extract_features
    Wav2Vec2Model.forward = uni_w2v2_forward
    HubertModel.extract_features = uni_hubert_extract_features
    HubertModel.forward = uni_hubert_forward
    TransformerEncoder.forward = uni_transformer_encoder_forward
    TransformerEncoder.extract_features = uni_transformer_encoder_extract_features
    TransformerSentenceEncoderLayer.forward = uni_self_attn_forward
    MultiheadAttention.__init__ = uni_mha_init
    MultiheadAttention.forward = uni_mha_forward
    
    # Patch LLaMA attention to store unrotated key/value states in the cache
    LlamaAttention.forward = llama_attention_new_forward
    LlamaFlashAttention2.forward = llama_flash_attention_2_new_forward
    LlamaSdpaAttention.forward = llama_sdpa_attention_new_forward

    # Patch beam search
    BeamSearchScorer.process = beam_search_process
    BeamSearchScorer.finalize = beam_search_finalize
    BeamHypotheses.add = beam_hypotheses_add
    # Patch generation mixin
    GenerationMixin.generate = generation_mixin_generate
    GenerationMixin._expand_inputs_for_generation = generation_mixin_expand_inputs_for_generation
    GenerationMixin._beam_search = generation_mixin_beam_search