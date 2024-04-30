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
from fairseq.models.speech_to_text import lengths_to_padding_mask, Conv1dSubsampler
from fairseq.models.wav2vec.utils import pad_to_multiple
from fairseq.modules import GradMultiply
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.utils import index_put, is_xla_tensor
from fairseq import utils
from model.model import SpeechLlamaModel, ZeroPadConv1dSubsampler
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel

original_forward = TransformerSentenceEncoderLayer.forward
BLOCKSIZE = 1

# def generate_2d_causal_mask(seq_len, device='cpu'):
#     """
#     Generates a 2D causal mask for multi-head attention.
    
#     Args:
#         seq_len (int): The length of the sequence.
#         device (str): The device on which to create the mask.
    
#     Returns:
#         torch.Tensor: A 2D causal attention mask.
#     """
#     mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)
#     mask = mask.masked_fill(mask == 1, float('-inf'))
#     return mask

def generate_2d_causal_mask(seq_len, prefix_len, dtype, device='gpu'):
    """
    Generates a 2D causal mask for multi-head attention.
    
    Args:
        seq_len (int): The length of the sequence.
        dtype (torch.dtype): The data type for the mask.
        device (str): The device on which to create the mask.
    
    Returns:
        torch.Tensor: A 2D causal attention mask.
    """
    global BLOCKSIZE
    blocksizes = [min(BLOCKSIZE, seq_len - i * BLOCKSIZE) for i in range((seq_len + BLOCKSIZE - 1) // BLOCKSIZE)]

    mask = torch.full((seq_len - prefix_len, seq_len), float('-inf'), device=device, dtype=dtype)
    start_idx = 0
    for block_size in blocksizes:
        end_idx = start_idx + block_size
        if end_idx > prefix_len:
            mask[max(0, start_idx - prefix_len) : end_idx - prefix_len, :end_idx] = 1
        start_idx = end_idx

    return mask
    # mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=dtype), diagonal=1)
    # mask = mask.masked_fill(mask == 1, float('-inf'))
    # return mask


# def causal_forward(
#     self,
#     x: torch.Tensor,
#     self_attn_mask: torch.Tensor = None,
#     self_attn_padding_mask: torch.Tensor = None,
#     need_weights: bool = False,
#     att_args=None,
# ):
#     # Generate the causal mask
#     # print(x)
#     # print(x.size(2))
#     # print(self_attn_mask)
#     causal_mask = generate_2d_causal_mask(x.size(0), dtype=x.dtype,device=x.device)
    
#     if self_attn_mask is not None:
#         self_attn_mask = self_attn_mask + causal_mask
#     else:
#         self_attn_mask = causal_mask

#     return original_forward(
#         self, x, 
#         self_attn_mask=self_attn_mask, 
#         self_attn_padding_mask=self_attn_padding_mask, 
#         need_weights=need_weights,
#         att_args=att_args)

def uni_w2v2_extract_features(self, source, padding_mask, mask=False, layer=None, 
                              past_features=None, start_pos=-1):
    res = self.forward(
        source, padding_mask, mask=mask, features_only=True, layer=layer, 
        past_features=past_features, start_pos=start_pos,
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
    past_features=None,
    start_pos=-1,
):

    if self.feature_grad_mult > 0:
        features = self.feature_extractor(source)
        if self.feature_grad_mult != 1.0:
            features = GradMultiply.apply(features, self.feature_grad_mult)
    else:
        with torch.no_grad():
            features = self.feature_extractor(source)

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
        past_features=past_features,
        start_pos=start_pos,
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

def uni_transformer_encoder_forward(self, x, padding_mask=None, layer=None, past_features=None, start_pos=-1):
    x, layer_results = self.extract_features(x, padding_mask, layer, past_features=past_features, start_pos=start_pos)

    if self.layer_norm_first and layer is None:
        x = self.layer_norm(x)

    if past_features is not None:
        x = torch.cat([past_features, x], dim=1)

    return x, layer_results

def uni_transformer_encoder_extract_features(
    self,
    x,
    padding_mask=None,
    tgt_layer=None,
    min_layer=0,
    past_features=None,
    start_pos=-1,
):
    if padding_mask is not None:
        x = index_put(x, padding_mask, 0)

    x_pad = F.pad(
        x.transpose(1, 2), 
        (self.pos_conv[0].padding[0] - 1, 0), 
        'constant', 
        0
    )

    x_conv = self.pos_conv(x_pad)
    x_conv = x_conv[:, :, :-(self.pos_conv[0].padding[0] - 1)]
    x_conv = x_conv.transpose(1, 2)
    x = x + x_conv

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

    prefix_length = 0
    if past_features is not None:
        prefix_length = past_features.size(1)
    x = x[:, prefix_length:, :]

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    total_length = x.size(0) + prefix_length
    causal_mask = generate_2d_causal_mask(
        total_length, 
        prefix_length,
        dtype=x.dtype, 
        device=x.device
    )

    layer_results = []
    r = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random() if self.layerdrop > 0 else 1
        if not self.training or (dropout_probability > self.layerdrop):
            x, (z, lr) = layer(
                x, 
                self_attn_mask=causal_mask,
                self_attn_padding_mask=padding_mask[:, prefix_length:] if padding_mask is not None else None, 
                need_weights=False, start_pos=start_pos,
            )
            if i >= min_layer:
                layer_results.append((x, z, lr))
        if i == tgt_layer:
            r = x
            break

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
    start_pos=-1,
):
    """
    LayerNorm is applied either before or after the self-attention/ffn
    modules similar to the original Transformer imlementation.
    """
    residual = x

    if self.layer_norm_first:
        x = self.self_attn_layer_norm(x)
        if start_pos == -1:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
                start_pos=start_pos,
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
        raise NotImplementedError
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
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

    return x, (attn, layer_result)


def uni_get_ssl_feature_w2v(self, src_tokens, src_lengths, after_lens, states=None):
    padding_mask = lengths_to_padding_mask(src_lengths)
    res = self.speech_tower.extract_features(
        src_tokens, 
        padding_mask, 
        past_features=states.w2v2_past_features,
        start_pos=states.w2v2_start_pos,
    )
    feature, padding_mask = res["x"], res["padding_mask"]
    states.w2v2_start_pos = feature.size(1)
    states.w2v2_past_features = feature
    if padding_mask is None:
    # Create a padding mask of shape [batch_size, seq_length] with all False values
        padding_mask = torch.zeros(feature.shape[:2], dtype=torch.bool, device=feature.device)
    output_length = (1 - padding_mask.int()).sum(dim=1)
    feature, input_lengths = self.mm_length_adapter(feature, output_length)
    assert after_lens.equal(input_lengths), "pre calculate length not match with the forward length"
    feature = self.mm_mlp_adapter(feature)    
    res = feature[states.speech_past_length:]
    states.speech_past_length = feature.size(0)
    return res
    # return feature


def uni_initialize_speech_modules(
    self, speech_tower_path, speech_tower_type=None,
    len_adapter_channels=None, len_adapter_kernel_sizes=None,
    stage1_complete=False, ssl_fintuned=False
):
    # loading pretrained ssl model
    # wav2vec 2.0
    if not ssl_fintuned: # ssl model
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
    self.mm_length_adapter = ZeroPadConv1dSubsampler(
                                    speech_dimension,
                                    len_adapter_channels,
                                    speech_dimension,
                                    [int(k) for k in len_adapter_kernel_sizes.split(',')]
                                ) 
    self.mm_mlp_adapter = nn.Linear(speech_dimension, self.config.hidden_size)
    length_after_ssl = self.speech_tower._get_feat_extract_output_lengths
    length_after_adp = self.mm_length_adapter.get_out_seq_lens_tensor
    
    if not stage1_complete:
        self.config.speech_tower_path = speech_tower_path
        self.config.len_adapter_channels = len_adapter_channels
        self.config.len_adapter_kernel_sizes = len_adapter_kernel_sizes
        self.config.stage1_complete = True
        self.config.ssl_fintuned = ssl_fintuned
                
    return (length_after_ssl, length_after_adp)



def uni_llama_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        if attention_mask is None or len(attention_mask.size()) == 2:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            # )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

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
    ] = 32,
    max_seq_len: Optional[
        int
    ] = 3000,
):
    super(MultiheadAttention, self).__init__()

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

    self.cache_k = torch.zeros(
        (
            self.max_batch_size * self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )
    ).cuda()
    self.cache_v = torch.zeros(
        (
            self.max_batch_size * self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )
    ).cuda()
    self.prev_key_padding_mask = None


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
    start_pos: int = -1,
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

    if (
        not self.onnx_trace
        and not is_tpu  # don't use PyTorch version on TPUs
        and start_pos < 0
        and not static_kv
        # A workaround for quantization to work. Otherwise JIT compilation
        # treats bias in linear module as method.
        and not torch.jit.is_scripting()
        # The Multihead attention implemented in pytorch forces strong dimension check
        # for input embedding dimention and K,Q,V projection dimension.
        # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
        # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
        and not self.skip_embed_dim_check
    ):
        assert key is not None and value is not None

        if self.use_xformers:
            return self._xformers_attn_forward(
                query, key, value, key_padding_mask, need_weights, attn_mask
            )

        else:
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

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

    if start_pos >= 0:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)

        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(q)

        self.cache_k[:kv_bsz * self.num_heads, start_pos : start_pos + src_len] = k
        self.cache_v[:kv_bsz * self.num_heads, start_pos : start_pos + src_len] = v

        k = self.cache_k[:kv_bsz * self.num_heads, : start_pos + src_len]
        v = self.cache_v[:kv_bsz * self.num_heads, : start_pos + src_len]

        src_len += start_pos

        assert k is not None and v is not None
        if start_pos == 0:
            self.prev_key_padding_mask = None
        else:
            self.prev_key_padding_mask = self.prev_key_padding_mask[:, :start_pos]
        key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
            key_padding_mask=key_padding_mask,
            prev_key_padding_mask=self.prev_key_padding_mask,
            batch_size=kv_bsz,
            src_len=k.size(1),
            static_kv=static_kv,
        )
        self.prev_key_padding_mask = key_padding_mask

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


def replace_uni_train(blocksize=1):
    global BLOCKSIZE
    BLOCKSIZE = blocksize
    TransformerEncoder.extract_features = uni_transformer_encoder_extract_features
    TransformerSentenceEncoderLayer.forward = uni_self_attn_forward
    SpeechLlamaModel.initialize_speech_modules = uni_initialize_speech_modules
    LlamaModel.forward = uni_llama_forward
    

def replace_uni_decode(blocksize=1):
    global BLOCKSIZE
    BLOCKSIZE = blocksize
    Wav2Vec2Model.extract_features = uni_w2v2_extract_features
    Wav2Vec2Model.forward = uni_w2v2_forward
    TransformerEncoder.forward = uni_transformer_encoder_forward
    TransformerEncoder.extract_features = uni_transformer_encoder_extract_features
    TransformerSentenceEncoderLayer.forward = uni_self_attn_forward
    SpeechLlamaModel.forward = SpeechLlamaModel.forward_incremental_v2
    SpeechLlamaModel.get_ssl_feature_w2v = uni_get_ssl_feature_w2v
    SpeechLlamaModel.initialize_speech_modules = uni_initialize_speech_modules
    MultiheadAttention.__init__ = uni_mha_init
    MultiheadAttention.forward = uni_mha_forward
    LlamaModel.forward = uni_llama_forward