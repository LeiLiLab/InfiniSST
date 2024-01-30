from typing import Dict, List, Optional, Tuple
import fairseq
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
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
from fairseq.utils import index_put, is_xla_tensor
from model.model import SpeechLlamaModel

class ZeroPadConv1dSubsampler(Conv1dSubsampler):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=0,
            )
            for i, k in enumerate(kernel_sizes)
        )
    
    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 3) / 2 + 1).floor().long()
        return out 

original_forward = TransformerSentenceEncoderLayer.forward

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

def generate_2d_causal_mask(seq_len, dtype, device='gpu'):
    """
    Generates a 2D causal mask for multi-head attention.
    
    Args:
        seq_len (int): The length of the sequence.
        dtype (torch.dtype): The data type for the mask.
        device (str): The device on which to create the mask.
    
    Returns:
        torch.Tensor: A 2D causal attention mask.
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


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
                              past_key_values=None, past_features=None):
    res = self.forward(
        source, padding_mask, mask=mask, features_only=True, layer=layer, 
        past_key_values=past_key_values, past_features=past_features
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
    past_key_values=None,
    past_features=None,
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
        past_key_values=past_key_values, 
        past_features=past_features,
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

def uni_transformer_encoder_forward(self, x, padding_mask=None, layer=None, past_key_values=None, past_features=None):
    x, layer_results = self.extract_features(x, padding_mask, layer, past_key_values=past_key_values, past_features=past_features)

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
    past_key_values=None,
    past_features=None,
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
    if past_key_values is not None and len(past_key_values[0]) > 0:
        saved_states = self.layers[0].self_attn._get_input_buffer(past_key_values[0])
        prefix_length = saved_states["prev_key"].size(2)
    x = x[:, prefix_length:, :]

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    layer_results = []
    r = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random() if self.layerdrop > 0 else 1
        if not self.training or (dropout_probability > self.layerdrop):
            x, (z, lr) = layer(
                x, self_attn_padding_mask=padding_mask[:, prefix_length:], need_weights=False,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
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

        if past_key_values is not None:
            for past_key_value in past_key_values:
                for _, p in past_key_value.items():
                    p['prev_key'] = p['prev_key'][:, :, :-pad_length, :]
                    p['prev_value'] = p['prev_value'][:, :, :-pad_length, :]
                    p['prev_key_padding_mask'] = p['prev_key_padding_mask'][:, :-pad_length]

    return x, layer_results

def uni_self_attn_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor = None,
    self_attn_padding_mask: torch.Tensor = None,
    need_weights: bool = False,
    att_args=None,
    past_key_value=None,
):
    """
    LayerNorm is applied either before or after the self-attention/ffn
    modules similar to the original Transformer imlementation.
    """

    saved_states = None
    total_length = x.size(0)
    prefix_length = 0
    if past_key_value is not None and len(past_key_value) > 0:
        saved_states = self.self_attn._get_input_buffer(past_key_value)
        prefix_length = saved_states["prev_key"].size(2)
        total_length += prefix_length

    causal_mask = generate_2d_causal_mask(total_length, dtype=x.dtype,device=x.device)[prefix_length:]

    if self_attn_mask is not None:
        self_attn_mask = self_attn_mask + causal_mask
    else:
        self_attn_mask = causal_mask
        
    residual = x

    if self.layer_norm_first:
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            need_weights=False,
            incremental_state=past_key_value,
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


def uni_get_ssl_feature_w2v(self, src_tokens, src_lengths, after_lens, states):
    padding_mask = lengths_to_padding_mask(src_lengths)
    res = self.speech_tower.extract_features(
        src_tokens, 
        padding_mask, 
        past_key_values=states.w2v2_past_key_values,
        past_features=states.w2v2_past_features
    )
    feature, padding_mask = res["x"], res["padding_mask"]
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


def replace_forward():
    TransformerEncoder.extract_features = uni_transformer_encoder_extract_features
    TransformerSentenceEncoderLayer.forward = uni_self_attn_forward
    SpeechLlamaModel.initialize_speech_modules = uni_initialize_speech_modules
    

def replace_forward_incremental():
    Wav2Vec2Model.extract_features = uni_w2v2_extract_features
    Wav2Vec2Model.forward = uni_w2v2_forward
    TransformerEncoder.forward = uni_transformer_encoder_forward
    TransformerEncoder.extract_features = uni_transformer_encoder_extract_features
    TransformerSentenceEncoderLayer.forward = uni_self_attn_forward
    SpeechLlamaModel.forward = SpeechLlamaModel.forward_incremental
    SpeechLlamaModel.get_ssl_feature_w2v = uni_get_ssl_feature_w2v
    SpeechLlamaModel.initialize_speech_modules = uni_initialize_speech_modules