import sys
sys.path.append('/home/siqiouyang/work/projects/sllama')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
from tqdm.notebook import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from model.speech_encoder import SpeechEncoder, SpeechEncoderW2V2RoPE
from model.model import SpeechLlamaForCausalLM
from fairseq.examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv

import numpy as np
import matplotlib.pyplot as plt
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
model = SpeechEncoderW2V2RoPE(
    "/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt",
    True,
    "[(1024,2,2)] * 2",    
    48,
    500,      
    copy.deepcopy(llm.model.embed_tokens),
).to('cuda')
del llm
ckpt = torch.load("/mnt/taurus/data/siqiouyang/runs/sllama/en-de/crtl-stage0-w2v2-cache10s/epoch=57-step=75021.ckpt")
model.load_state_dict(ckpt['state_dict'])
del ckpt

model.eval()
torch.set_grad_enabled(False)
df = load_df_from_tsv("/mnt/aries/data/siqiouyang/datasets/must-c-v1.0/dev_st_de_full_mfa_llama3.tsv")
df_tst = load_df_from_tsv("/mnt/aries/data/siqiouyang/datasets/must-c-v1.0/tst-COMMON_st_de_full_mfa_llama3.tsv")

df_seg = load_df_from_tsv("/mnt/aries/data/siqiouyang/datasets/must-c-v1.0/dev_st_de_mfa_llama3.tsv")
df_tst_seg = load_df_from_tsv("/mnt/aries/data/siqiouyang/datasets/must-c-v1.0/tst-COMMON_st_de_mfa_llama3.tsv")
df = pd.concat([df, df_tst], ignore_index=True)
df_seg = pd.concat([df_seg, df_tst_seg], ignore_index=True)
d = df.iloc[0]

path, offset, duration = d['audio'].split(':')
offset = int(offset)
duration = int(duration)
wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=duration)

ted_id = int(d['id'].split('_')[1])

cache = None
outputs = []
last_frame = 0
for i in range(79 + 320, wav.size(1), 320 * 48):
    x = wav[:, last_frame : i + 320 * 48].to('cuda')
    x_len = torch.LongTensor([x.size(1)]).to('cuda')

    output, cache = model.encode_speech(x, x_len, cache=cache)
    outputs.append(output)
    last_frame = i + 320 * 48