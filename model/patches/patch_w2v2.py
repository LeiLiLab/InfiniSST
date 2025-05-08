from typing import Dict, List, Optional, Tuple

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.models.hubert.hubert import HubertModel
from fairseq.models.wav2vec import (
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
    Wav2Vec2Model,  
)
from fairseq.models.wav2vec.utils import pad_to_multiple
from fairseq.modules import GradMultiply
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.utils import index_put, is_xla_tensor
from fairseq import utils

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)

def get_attn_mask_training(seq_len, max_cache_size=None, blocksize=1, device='cuda'):
    blocksizes = [
        min(blocksize, seq_len - i * blocksize) 
        for i in range((seq_len + blocksize - 1) // blocksize)
    ]

    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
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

def get_attn_mask_training_opt(seq_len, max_cache_size=None, blocksize=1, device='cuda', dtype=torch.float32):
    """Generate block causal attention mask more efficiently using vectorized operations.
    
    Args:
        seq_len: Length of sequence
        max_cache_size: Maximum size of the cache window (optional)
        blocksize: Size of each block for block-wise attention
        device: Device to create tensors on
        dtype: Data type of the mask
    Returns:
        mask_num: Float tensor containing -inf for masked positions and 0 for attended positions
    """
    # Create position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    
    # Calculate block indices for each position
    row_block = row_idx // blocksize  # [seq_len, 1]
    col_block = col_idx // blocksize  # [1, seq_len]
    
    # Create block causal mask
    # Allow attention within same block or to previous blocks
    mask = (row_block >= col_block)
    
    # Apply cache size limit if specified
    if max_cache_size is not None:
        # Only attend to at most max_cache_size previous positions
        cache_mask = (col_idx >= (row_idx - max_cache_size))
        mask = mask & cache_mask
    
    # Convert to float mask with -inf for masked positions
    mask_num = torch.where(mask, 0.0, float('-inf')).to(dtype)
    
    return mask_num

def get_attn_mask_inference(seq_len, prefix_len, max_cache_size, blocksize=1, device='cuda'):
    max_len = seq_len + min(prefix_len, max_cache_size)

    blocksizes = [
        min(blocksize, seq_len + prefix_len - i * blocksize) 
        for i in range((seq_len + prefix_len + blocksize - 1) // blocksize)
    ]

    mask = torch.zeros(seq_len, max_len, device=device, dtype=torch.bool)
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

def get_attn_mask_inference_opt(seq_len, prefix_len, max_cache_size, blocksize=1, device='cuda', dtype=torch.float32):
    """Generate block causal attention mask for inference more efficiently using vectorized operations.
    
    Args:
        seq_len: Length of new sequence to generate
        prefix_len: Length of prefix/context
        max_cache_size: Maximum size of the cache window
        blocksize: Size of each block for block-wise attention
        device: Device to create tensors on
        dtype: Data type of the mask
    Returns:
        mask_num: Float tensor containing -inf for masked positions and 0 for attended positions
    """
    max_len = seq_len + min(prefix_len, max_cache_size)
    
    # Create position indices
    # [seq_len, 1]
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1) + prefix_len
    # [1, max_len]
    col_idx = torch.arange(max_len, device=device).unsqueeze(0) + max(0, prefix_len - max_cache_size)
    
    
    # Calculate block indices
    # For rows: offset by prefix_len since we're only generating seq_len new tokens
    row_block = row_idx // blocksize  # [seq_len, 1]
    col_block = col_idx // blocksize  # [1, max_len]
    
    # Create block causal mask
    mask = (row_block >= col_block)
    
    # Apply cache size limitation
    # Each position can only attend to itself and max_cache_size previous positions
    cache_mask = (col_idx >= (row_idx - max_cache_size))
    mask = mask & cache_mask
    
    # Convert to float mask with -inf for masked positions
    mask_num = torch.where(mask, 0.0, float('-inf')).to(dtype)
    
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
    # TODO: optimize
    
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
    # x = x.transpose(0, 1)

    prefix_len = cache.n_steps
    seq_len = x.size(1)
    # logger.info(f"w2v2 enc forward: device {x.device}, blocksize {self.blocksize}")
    if prefix_len > 0:
        attn_mask = get_attn_mask_inference_opt(
            seq_len, prefix_len, cache.max_steps, self.blocksize, 
            x.device, x.dtype
        )
    else:
        attn_mask = get_attn_mask_training_opt(
            seq_len, cache.max_steps, self.blocksize, 
            x.device, x.dtype
        )


    layer_results = [(x, None, None)]
    r = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random() if self.layerdrop > 0 else 1
        if not self.training or (dropout_probability > self.layerdrop):
            if cache.layers[i].k is not None:
                cache.layers[i].k = cache.layers[i].k[:, :, -cache.max_steps:]
                cache.layers[i].v = cache.layers[i].v[:, :, -cache.max_steps:]
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
    # x = x.transpose(0, 1)

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
    
    self.rotary_emb = LlamaRotaryEmbedding(embed_dim // num_heads)

    xformers_att_config = utils.eval_str_dict(xformers_att_config)
    self.use_xformers = xformers_att_config is not None
    self.embed_dim = embed_dim
    self.kdim = kdim if kdim is not None else embed_dim
    self.vdim = vdim if vdim is not None else embed_dim
    self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

    self.num_heads = num_heads
    self.dropout = dropout
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

    bsz, q_len, _ = query.size()
    _, k_len, _ = key.size()

    q = self.q_proj(query)
    k = self.k_proj(key)
    v = self.v_proj(value)

    q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

    if cache.k is not None:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)

        cache.k = cache.k.to(q)
        cache.v = cache.v.to(q)

        cache.k = torch.cat([cache.k, k], dim=2)
        cache.v = torch.cat([cache.v, v], dim=2)

        k, v = cache.k, cache.v
        k_len = k.size(2)
    else:
        cache.k, cache.v = k, v
    
    if ROPE:
        position_ids = torch.arange(k_len, device=v.device).repeat(bsz, 1)
        cos, sin = self.rotary_emb(k, position_ids)
        if q_len == k_len: # training
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else: # inference
            k, _ = apply_rotary_pos_emb(k, k, cos, sin)
            q_cos, q_sin = self.rotary_emb(q, position_ids[:, -q_len:])
            q, _ = apply_rotary_pos_emb(q, q, q_cos, q_sin)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.out_proj(attn_output)

    return attn_output, None

def patch_w2v2(rope=True):
    global ROPE
    print("Patching with rope {}".format(rope))
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