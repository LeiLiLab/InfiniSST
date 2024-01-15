import fairseq
import torch 
import torch.nn as nn
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from fairseq.models.wav2vec import (
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
    Wav2Vec2Model,
    Wav2VecEncoder    
)

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

def causal_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor = None,
    self_attn_padding_mask: torch.Tensor = None,
    need_weights: bool = False,
    att_args=None,
):
    """
    LayerNorm is applied either before or after the self-attention/ffn
    modules similar to the original Transformer imlementation.
    """

    causal_mask = generate_2d_causal_mask(x.size(0), dtype=x.dtype,device=x.device)
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

def replace_forward():
    TransformerSentenceEncoderLayer.forward = causal_forward