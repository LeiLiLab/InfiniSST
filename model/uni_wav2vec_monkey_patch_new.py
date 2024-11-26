import os
from typing import Dict, List, Optional, Tuple, Union
import fairseq
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
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

BLOCKSIZE = 1

def get_attn_mask_training(seq_len, max_cache_size=None):
    blocksizes = [
        min(BLOCKSIZE, seq_len - i * BLOCKSIZE) 
        for i in range((seq_len + BLOCKSIZE - 1) // BLOCKSIZE)
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
    
    return mask

def get_attn_mask_inference(seq_len, prefix_len, max_cache_size):
    max_len = seq_len + min(prefix_len, max_cache_size)

    blocksizes = [
        min(BLOCKSIZE, seq_len + prefix_len - i * BLOCKSIZE) 
        for i in range((seq_len + prefix_len + BLOCKSIZE - 1) // BLOCKSIZE)
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
    
    return mask


def uni_w2v2_extract_features(self, source, padding_mask, mask=False, layer=None, cache=None):
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
    
    if cache.src_len > 0:
        new_src_len = features.size(-1)
        features = features[..., cache.src_len:]
        cache.src_len = new_src_len

        max_src_token_len = 79 + 320 + 320 * BLOCKSIZE
        if cache.src.size(1) > max_src_token_len:
            cache.src = cache.src[:, -max_src_token_len:]
            cache.src_len = BLOCKSIZE
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

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    prefix_len = cache.n_steps
    seq_len = x.size(0)
    if prefix_len > 0:
        attn_mask = get_attn_mask_inference(seq_len, prefix_len, cache.max_steps)
    else:
        attn_mask = get_attn_mask_training(seq_len, cache.max_steps)

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
    
    self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, use_xpos=True)

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


def patch_w2v2(blocksize=1):
    global BLOCKSIZE
    BLOCKSIZE = blocksize
    Wav2Vec2Model.extract_features = uni_w2v2_extract_features
    Wav2Vec2Model.forward = uni_w2v2_forward
    TransformerEncoder.forward = uni_transformer_encoder_forward
    TransformerEncoder.extract_features = uni_transformer_encoder_extract_features
    TransformerSentenceEncoderLayer.forward = uni_self_attn_forward
    MultiheadAttention.__init__ = uni_mha_init
    MultiheadAttention.forward = uni_mha_forward