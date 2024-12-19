import os
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
    Wav2Vec2BertModel,
    Wav2Vec2BertEncoder,
    Wav2Vec2BertEncoderLayer,
    Wav2Vec2BertSelfAttention,
    Wav2Vec2BertConvolutionModule
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Wav2Vec2BaseModelOutput
)

BLOCKSIZE = 1

def get_attn_mask_training(seq_len, max_cache_size=None):
    blocksizes = [
        min(BLOCKSIZE, seq_len - i * BLOCKSIZE) 
        for i in range((seq_len + BLOCKSIZE - 1) // BLOCKSIZE)
    ]

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    start_idx = 0
    for block_size in blocksizes:
        end_idx = start_idx + block_size
        mask[start_idx : end_idx, :end_idx] = 1
        start_idx = end_idx
    
    if max_cache_size is not None:
        for i in range(seq_len):
            mask[i, : max(0, i - max_cache_size)] = 0

    mask_num = torch.zeros_like(mask, dtype=torch.float32)
    mask_num.masked_fill_(~mask, float('-inf'))
    
    return mask_num

def get_attn_mask_inference(seq_len, prefix_len, max_cache_size):
    max_len = seq_len + min(prefix_len, max_cache_size)

    blocksizes = [
        min(BLOCKSIZE, seq_len + prefix_len - i * BLOCKSIZE) 
        for i in range((seq_len + prefix_len + BLOCKSIZE - 1) // BLOCKSIZE)
    ]

    mask = torch.zeros(seq_len, max_len, dtype=torch.bool)
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

    mask_num = torch.zeros_like(mask, dtype=torch.float32)
    mask_num.masked_fill_(~mask, float('-inf'))
    
    return mask_num

def conv_forward(
    self, hidden_states, attention_mask=None, cache=None,
):
    hidden_states = self.layer_norm(hidden_states)

    # Ensure that we do not leak padded positions in depthwise convolution if attention mask is passed.
    # Put 0 where necessary
    if attention_mask is not None:
        hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

    # exchange the temporal dimension and the feature dimension
    hidden_states = hidden_states.transpose(1, 2)

    # GLU mechanism
    # => (batch, 2*channel, dim)
    hidden_states = self.pointwise_conv1(hidden_states)
    # => (batch, channel, dim)
    hidden_states = self.glu(hidden_states)

    _, _, seq_len = hidden_states.size()
    if cache.hidden_states_conv is not None:
        hidden_states = torch.cat([cache.hidden_states_conv, hidden_states], dim=2)
    cache.hidden_states_conv = hidden_states

    # Pad the sequence entirely on the left because of causal convolution.
    hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

    # 1D Depthwise Conv
    hidden_states = self.depthwise_conv(hidden_states)[:, :, -seq_len:]
    cache.hidden_states_conv = cache.hidden_states_conv[:, :, -self.depthwise_conv.kernel_size[0]:]

    hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    hidden_states = self.activation(hidden_states)

    hidden_states = self.pointwise_conv2(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = hidden_states.transpose(1, 2)
    return hidden_states

def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    relative_position_embeddings: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    block_causal_mask=None,
    cache=None,
):
    # self-attention mechanism
    batch_size, sequence_length, hidden_size = hidden_states.size()

    if cache.hidden_states_attn is not None:
        cache.hidden_states_attn = cache.hidden_states_attn.to(hidden_states)
        cache.hidden_states_attn = torch.cat([cache.hidden_states_attn, hidden_states], dim=1)
        hidden_states = cache.hidden_states_attn
    cache.hidden_states_attn = hidden_states
    
    # make sure query/key states can be != value states
    query_key_states = hidden_states
    value_states = hidden_states

    # project query_key_states and value_states
    query = self.linear_q(query_key_states)
    key = self.linear_k(query_key_states)
    value = self.linear_v(value_states)

    if self.position_embeddings_type == "rotary":
        if relative_position_embeddings is None:
            raise ValueError(
                "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
            )
        query = self._apply_rotary_embedding(query, relative_position_embeddings)
        key = self._apply_rotary_embedding(key, relative_position_embeddings)

    query = query.view(batch_size, -1, self.num_heads, self.head_size)[:, -sequence_length:]
    key = key.view(batch_size, -1, self.num_heads, self.head_size)
    value = value.view(batch_size, -1, self.num_heads, self.head_size)
    
    # => (batch, head, time1, d_k)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if self.position_embeddings_type == "relative":
        if relative_position_embeddings is None:
            raise ValueError(
                "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                " 'relative'"
            )
        # apply relative_position_embeddings to qk scores
        # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
        scores = self._apply_relative_embeddings(
            query=query, key=key, relative_position_embeddings=relative_position_embeddings
        )
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

    if self.position_embeddings_type == "relative_key":
        query_length, key_length = query.shape[2], key.shape[2]

        position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_r - position_ids_l
        distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

        positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
        positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

        relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
        scores = scores + (relative_position_attn_weights / math.sqrt(self.head_size))

    # apply attention_mask if necessary
    if attention_mask is not None:
        scores = scores + attention_mask

    if block_causal_mask is not None:
        scores = scores + block_causal_mask

    # => (batch, head, time1, time2)
    probs = torch.softmax(scores, dim=-1)
    probs = self.dropout(probs)

    # => (batch, head, time1, d_k)
    hidden_states = torch.matmul(probs, value)

    # => (batch, time1, hidden_size)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
    hidden_states = self.linear_out(hidden_states)

    return hidden_states, probs


def encoder_layer_forward(
    self,
    hidden_states,
    attention_mask: Optional[torch.Tensor] = None,
    relative_position_embeddings: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    conv_attention_mask: Optional[torch.Tensor] = None,
    block_causal_mask=None,
    cache=None,
):
    hidden_states = hidden_states

    # 1. Feed-Forward 1 layer
    residual = hidden_states
    hidden_states = self.ffn1_layer_norm(hidden_states)
    hidden_states = self.ffn1(hidden_states)
    hidden_states = hidden_states * 0.5 + residual
    residual = hidden_states

    # 2. Self-Attention layer
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_weigts = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        relative_position_embeddings=relative_position_embeddings,
        output_attentions=output_attentions,
        block_causal_mask=block_causal_mask,
        cache=cache,
    )
    hidden_states = self.self_attn_dropout(hidden_states)
    hidden_states = hidden_states + residual

    # 3. Convolutional Layer
    residual = hidden_states
    hidden_states = self.conv_module(
        hidden_states, 
        attention_mask=conv_attention_mask,
        cache=cache,
    )
    hidden_states = residual + hidden_states

    # 4. Feed-Forward 2 Layer
    residual = hidden_states
    hidden_states = self.ffn2_layer_norm(hidden_states)
    hidden_states = self.ffn2(hidden_states)
    hidden_states = hidden_states * 0.5 + residual
    hidden_states = self.final_layer_norm(hidden_states)

    return hidden_states, attn_weigts


def encoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
    cache=None,
):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    conv_attention_mask = attention_mask
    if attention_mask is not None:
        # make sure padded tokens output 0
        hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # extend attention_mask
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )

    hidden_states = self.dropout(hidden_states)

    if self.embed_positions is not None:
        hidden_states_rope = hidden_states
        if cache.layers[0].hidden_states_attn is not None:
            hidden_states_rope = torch.cat(
                [cache.layers[0].hidden_states_attn, hidden_states], 
                dim=1
            )
        relative_position_embeddings = self.embed_positions(hidden_states_rope)
    else:
        relative_position_embeddings = None

    synced_gpus = False # is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

    seq_len = hidden_states.size(1)
    if cache.n_steps > 0:
        block_causal_mask = get_attn_mask_inference(seq_len, cache.n_steps, cache.max_steps)
    else:
        block_causal_mask = get_attn_mask_training(seq_len, cache.max_steps)
    block_causal_mask = block_causal_mask.to(hidden_states)

    for i, layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = torch.rand([])

        if cache.layers[i].hidden_states_attn is not None:
            cache.layers[i].hidden_states_attn = cache.layers[i].hidden_states_attn[:, -cache.max_steps:]

        skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
        if not skip_the_layer or synced_gpus:
            # under fsdp or deepspeed zero3 all gpus must run in sync
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    relative_position_embeddings,
                    output_attentions,
                    conv_attention_mask,
                    block_causal_mask,
                    cache.layers[i],
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    relative_position_embeddings=relative_position_embeddings,
                    output_attentions=output_attentions,
                    conv_attention_mask=conv_attention_mask,
                    block_causal_mask=block_causal_mask,
                    cache=cache.layers[i],
                )
            hidden_states = layer_outputs[0]

        if skip_the_layer:
            layer_outputs = (None, None)

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
    cache.n_steps += seq_len

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def model_forward(
    self,
    input_features: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    mask_time_indices: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache=None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states, extract_features = self.feature_projection(input_features)
    hidden_states = self._mask_hidden_states(
        hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
    )

    encoder_outputs = self.encoder(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache=cache
    )

    hidden_states = encoder_outputs[0]

    if self.intermediate_ffn:
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

    if self.adapter is not None:
        hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

    if not return_dict:
        return (hidden_states, extract_features) + encoder_outputs[1:]

    return Wav2Vec2BaseModelOutput(
        last_hidden_state=hidden_states,
        extract_features=extract_features,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def patch_w2vbert2(blocksize=1):
    global BLOCKSIZE, XPOS
    print("Patching with block size {}".format(blocksize))
    BLOCKSIZE = blocksize

    Wav2Vec2BertModel.forward = model_forward
    Wav2Vec2BertEncoder.forward = encoder_forward
    Wav2Vec2BertEncoderLayer.forward = encoder_layer_forward
    Wav2Vec2BertSelfAttention.forward = attention_forward
    Wav2Vec2BertConvolutionModule.forward = conv_forward