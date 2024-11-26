import sys
sys.path.append('/home/siqiouyang/work/projects/sllama')

import copy
import torch
import transformers
from model.speech_encoder import SpeechEncoder
from model.model import SpeechLlamaForCausalLM

llm = SpeechLlamaForCausalLM.from_pretrained(
    "/mnt/taurus/data/siqiouyang/download/llama3.1-8b-hf",
    low_cpu_mem_usage=True,
    load_in_8bit=False,
    device_map='cpu',
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/mnt/taurus/data/siqiouyang/download/llama3.1-8b-hf",
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = "<|finetune_right_pad_id|>"
model = SpeechEncoder(
    '[(1024, 10, 5)] + [(1024, 3, 2)] * 4 + [(1024,2,2)] * 4',
    None,
    False,
    None,    
    n_attn_layers=12,
    n_dim=1024,
    n_heads=16,
    dropout=0.1,
    block_size=12,
    max_cache_size=125,      
    llm_embedding=copy.deepcopy(llm.model.embed_tokens)
).to('cuda')
model.eval()
torch.set_grad_enabled(False)

src_tokens = torch.rand(1, 79 + 15360 * 4).to('cuda')

len = 79 + 1280 + 15360 * 3
src_lens = torch.Tensor([len]).to('cuda')
full_output, _ = model.encode_speech(src_tokens[:, : len], src_lens)

len = 79 + 1280 + 15360
src_lens = torch.Tensor([len]).to('cuda')
partial_output, cache = model.encode_speech(src_tokens[:, : len], src_lens)
full_output.size(), partial_output.size()
((partial_output - full_output[:, :12]) / partial_output).abs().mean()

len1 = 79 + 1280 + 15360
len2 = 79 + 1280 + 15360 * 2
src_lens = torch.Tensor([len2 - len1]).to('cuda')
partial_output_2, cache = model.encode_speech(src_tokens[:, len1 : len2], 
src_lens, cache=cache)

len1 = 79 + 1280 + 15360 * 2
len2 = 79 + 1280 + 15360 * 3
src_lens = torch.Tensor([len2 - len1]).to('cuda')
partial_output_3, cache = model.encode_speech(src_tokens[:, len1 : len2], 
src_lens, cache=cache)

inc_output = torch.cat([partial_output, partial_output_2, partial_output_3], dim=1)
print((inc_output - full_output).abs() / inc_output.abs())