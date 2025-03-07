# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SeamlessM4Tv2 model."""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from rotary_embedding_torch import RotaryEmbedding

from fairseq.models.speech_to_text import lengths_to_padding_mask

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Wav2Vec2BaseModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.seamless_m4t_v2.configuration_seamless_m4t_v2 import SeamlessM4Tv2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = ""
_CONFIG_FOR_DOC = "SeamlessM4Tv2Config"


SEAMLESS_M4T_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEAMLESS_M4T_V2_MULTIMODAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
    """

M4T_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

M4T_SPEECH_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
        """

SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING = r"""
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

M4T_MODEL_INPUTS_DOCSTRING = SEAMLESS_M4T_V2_MULTIMODAL_INPUTS_DOCSTRING + SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING

M4T_TEXT_INPUTS_DOCSTRING = M4T_TEXT_INPUTS_DOCSTRING + SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING

M4T_SPEECH_INPUTS_DOCSTRING = M4T_SPEECH_INPUTS_DOCSTRING + SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING

M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        char_input_ids (`torch.LongTensor` of shape `(batch_size, char_sequence_length)`):
            Character indices. The correspondence between characters and indices can be found in `char_to_id`, a
            dictionary in the generation configuration.
        char_count_per_id (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


############ UTILS ################


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.format_speech_generation_kwargs with SeamlessM4T->SeamlessM4Tv2
def format_speech_generation_kwargs(kwargs):
    """
    Format kwargs for SeamlessM4Tv2 models that generate speech, attribute kwargs to either the text generation or the
    speech generation models.

    Args:
        kwargs (`dict`)`:
             Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for one generation but not for the
                other.
    """
    # attribute kwargs to models
    kwargs_text = {}
    kwargs_speech = {}
    for key, value in kwargs.items():
        if key.startswith("text_"):
            key = key[len("text_") :]
            kwargs_text[key] = value
        elif key.startswith("speech_"):
            key = key[len("speech_") :]
            kwargs_speech[key] = value
        elif key == "generation_config":
            kwargs_text[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return kwargs_text, kwargs_speech


############ SPEECH ENCODER related code ################

class SeamlessM4Tv2SpeechEncoderLayerCache:
    key: torch.Tensor = None
    value: torch.Tensor = None
    conv_hidden_states: torch.Tensor = None

class SeamlessM4Tv2SpeechEncoderCache:
    cache_size: int = 0
    layers: List[SeamlessM4Tv2SpeechEncoderLayerCache] = None

    def __init__(self, cache_size=0, layers=None):
        self.cache_size = cache_size
        self.layers = layers


class SeamlessM4Tv2ConformerFeatureProjection(nn.Module):
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeatureProjection.__init__
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states.to(self.layer_norm.weight.dtype))
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeedForward with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        self.intermediate_dropout = nn.Dropout(dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block. Uses a causal depthwise convolution similar to that
    described in Section 2.1 of `https://doi.org/10.48550/arxiv.1609.03499"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        cache=None,
    ):
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution.
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

        seq_len = hidden_states.size(-1)
        if cache.conv_hidden_states is not None:
            hidden_states = torch.cat([cache.conv_hidden_states, hidden_states], dim=2)
        cache.conv_hidden_states = hidden_states[:, :, -self.depthwise_conv.kernel_size[0]:]

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = hidden_states[:, :, -seq_len:]
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SeamlessM4Tv2ConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4Tv2ConformerSelfAttention object.
    Can be enhanced with relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        super().__init__()

        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        self.num_heads = config.speech_encoder_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)
        elif self.position_embeddings_type == "rope":
            self.rotary_embedding = RotaryEmbedding(self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache: Optional[SeamlessM4Tv2SpeechEncoderLayerCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if cache.key is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            key = torch.cat([cache.key, key], dim=2)
            value = torch.cat([cache.value, value], dim=2)
        cache.key, cache.value = key, value

        if self.position_embeddings_type == "rope":
            query, key = self.rotary_embedding.rotate_queries_with_cached_keys(query, key)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = torch.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class SeamlessM4Tv2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4Tv2, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[SeamlessM4Tv2SpeechEncoderLayerCache] = None,
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
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
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

        return hidden_states, attn_weights


class SeamlessM4Tv2ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.speech_encoder_left_chunk_num = config.speech_encoder_left_chunk_num
        self.speech_encoder_chunk_size = config.speech_encoder_chunk_size

        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList(
            [SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def _apply_multiplier(self, multiplier):
        self.speech_encoder_left_chunk_num = self.config.speech_encoder_left_chunk_num / multiplier
        self.speech_encoder_chunk_size = self.config.speech_encoder_chunk_size * multiplier

    def _apply_chunk_attention(self, attention_mask, hidden_states, cache):
        """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
        sequence_len = hidden_states.shape[1] + cache.cache_size

        chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
        chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()

        start_indices = torch.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(min=0)
            start_indices = start_indices * self.config.speech_encoder_chunk_size
            start_indices = start_indices
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)

        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(max=sequence_len)

        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)

        indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)

        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask[cache.cache_size:]
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)

        # TODO: might need to modify later for batched inference
        bsz = hidden_states.shape[0]
        chunk_mask = chunk_mask.repeat(bsz, 1, 1, 1)

        if attention_mask is not None:
            chunk_mask[:, :, :, cache.cache_size:] |= attention_mask.bool()
        attention_mask = chunk_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    def forward(
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
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        if self.config.speech_encoder_chunk_size is not None:
            # sliding window
            max_cache_size = self.config.speech_encoder_chunk_size * self.config.speech_encoder_left_chunk_num
            for layer in cache.layers:
                if layer.key is not None:
                    layer.key = layer.key[:, :, -max_cache_size:]
                    layer.value = layer.value[:, :, -max_cache_size:]
                    cache.cache_size = layer.key.size(2)

            attention_mask = self._apply_chunk_attention(attention_mask, hidden_states, cache)

        if attention_mask is not None:
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = (
                True if self.training and (dropout_probability < self.config.speech_encoder_layerdrop) else False
            )
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                        conv_attention_mask,
                        cache.layers[i],
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                        cache=cache.layers[i],
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapterLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout

        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=0,
        )
        self.activation = nn.GLU(dim=1)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = 0
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache: Optional[SeamlessM4Tv2SpeechEncoderLayerCache] = None,
    ):
        hidden_states = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.residual_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapter with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            SeamlessM4Tv2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)
        )

    def forward(self, hidden_states, attention_mask):
        # down project hidden_states if necessary

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ SUB-MODELS related code ################


class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SeamlessM4Tv2Config
    base_model_prefix = "seamless_m4t_v2"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "SeamlessM4Tv2EncoderLayer",
        "SeamlessM4Tv2DecoderLayer",
        "SeamlessM4Tv2ConformerEncoderLayer",
        "SeamlessM4Tv2TextToUnitDecoderLayer",
    ]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SeamlessM4Tv2ConformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, SeamlessM4Tv2ConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TPreTrainedModel._compute_sub_sample_lengths_from_attention_mask
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        pad = 0
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.floor()


@dataclass
class SeamlessM4Tv2SpeechEncoderBaseModelOutput(Wav2Vec2BaseModelOutput):
    """
    Base class for models that have been trained with the Wav2Vec2 loss objective.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    cache: Optional[SeamlessM4Tv2SpeechEncoderCache] = None

@add_start_docstrings(
    """Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.
    Each layer is a [`SeamlessM4Tv2ConformerEncoderLayer`].""",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TSpeechEncoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):
    main_input_name = "input_features"

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=0.0)
        self.adapter = SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        self.final_proj = nn.Linear(config.hidden_size, config.llm_embedding_dim)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_blocksize(self, multiplier):
        self.encoder._apply_multiplier(multiplier)

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache: Optional[SeamlessM4Tv2SpeechEncoderCache] = None,
        multiplier: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple, SeamlessM4Tv2SpeechEncoderBaseModelOutput]:
        
        if cache is None:
            cache = SeamlessM4Tv2SpeechEncoderCache(
                layers=[SeamlessM4Tv2SpeechEncoderLayerCache() for _ in range(self.config.speech_encoder_layers)],
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4Tv2SpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        hidden_states = self.feature_projection(input_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache=cache,
        )

        hidden_states = encoder_outputs[0]

        # print(hidden_states.size())

        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # print(hidden_states.size())

        hidden_states = self.inner_layer_norm(hidden_states)

        hidden_states = self.final_proj(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:] + (cache,)

        return SeamlessM4Tv2SpeechEncoderBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cache=cache,
        )

    def encode_speech(self, speech_batch, src_lengths, cache=None):
        attention_mask = (~lengths_to_padding_mask(src_lengths)).int()
        output = self(speech_batch, attention_mask, cache=cache)
        return output.last_hidden_state, output.cache