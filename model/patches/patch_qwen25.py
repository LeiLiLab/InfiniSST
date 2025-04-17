import math
import torch
import torch.nn as nn

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention,
    apply_rotary_pos_emb, repeat_kv
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import StaticCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

def qwen2_flash_attention_2_new_forward(self, *args, **kwargs):
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


def patch_qwen25():
    Qwen2FlashAttention2.forward = qwen2_flash_attention_2_new_forward