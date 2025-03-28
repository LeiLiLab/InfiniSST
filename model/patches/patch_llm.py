import math
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention,
    apply_rotary_pos_emb, repeat_kv
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import StaticCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

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

def patch_llm():    
    # Patch LLaMA attention to store unrotated key/value states in the cache
    LlamaAttention.forward = llama_attention_new_forward
    LlamaFlashAttention2.forward = llama_flash_attention_2_new_forward
    LlamaSdpaAttention.forward = llama_sdpa_attention_new_forward