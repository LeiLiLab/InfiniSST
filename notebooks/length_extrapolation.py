import sys
sys.path.append('/home/siqiouyang/work/projects/sllama')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from model.speech_encoder import SpeechEncoder
from model.model import SpeechLlamaForCausalLM
from fairseq.examples.speech_to_text.data_utils import load_df_from_tsv

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
del llm

# ckpt = torch.load("/mnt/taurus/data/siqiouyang/runs/sllama/en-de/crtl-stage0-cache125/epoch=74-step=96672.ckpt")
ckpt = torch.load("/mnt/taurus/data/siqiouyang/runs/sllama/en-de/crtl-stage0-cacheinf/epoch=46-step=60688.ckpt")
model.load_state_dict(ckpt['state_dict'])
del ckpt

model.eval()
torch.set_grad_enabled(False)

n_minute = 30

df = load_df_from_tsv("/mnt/aries/data/siqiouyang/datasets/must-c-v1.0/dev_st_de_full_mfa_llama3.tsv".format(n_minute))
all_sims = []
for df_idx in tqdm(range(len(df))):
    d = df.iloc[df_idx]

    path, offset, duration = d['audio'].split(':')
    offset = int(offset)
    duration = int(duration)
    wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=duration)

    cache = None
    cache_begin = None
    outputs = []
    last_frame = 0
    for i in range(79 + 1280, wav.size(1), 1280 * 12):
        x = wav[:, last_frame : i + 1280 * 12].to('cuda')
        x_len = torch.LongTensor([x.size(1)]).to('cuda')


        if cache_begin is not None and cache.n_steps >= model.max_cache_size:
            for c, cb in zip(cache.layers, cache_begin):
                k, v = c.attn_intermediates[0].cached_kv
                k_b, v_b = cb.attn_intermediates[0].cached_kv
                
                c.attn_intermediates[0].cached_kv = (
                    torch.cat([k_b, k[..., k.size(-2) - model.max_cache_size + k_b.size(-2):, :]], dim=-2),
                    torch.cat([v_b, v[..., v.size(-2) - model.max_cache_size + v_b.size(-2):, :]], dim=-2)
                )       

        output, cache = model.encode_speech(x, x_len, cache=cache)
        outputs.append(output)
        last_frame = i + 1280 * 12

        if i == 79 + 1280:
            cache_begin = copy.deepcopy(cache.layers)
    full_output = torch.cat(outputs, dim=1)

    speech_words = torch.tensor(eval(d['speech_word']))
    text_words = torch.tensor(eval(d['text_word']))
    d['src_text']
    src_text = tokenizer.encode(
        d['src_text'],
        return_tensors="pt",
        padding="longest",
        truncation=False,
        add_special_tokens=False
    ).to('cuda')
    src_text_emb = model.llm_embedding(src_text)
    for i in range(speech_words.size(0)):
        s_l, s_r = speech_words[i]
        t_l, t_r = text_words[i]

        s_l = int((s_l / 0.08).floor())
        s_r = min(int((s_r / 0.08).ceil()), full_output.size(1)) - 1

        s_word_emb = full_output[0][s_l : s_r + 1].mean(dim=0)
        t_word_emb = src_text_emb[0][t_l : t_r + 1].mean(dim=0)

        sim = F.cosine_similarity(s_word_emb, t_word_emb, dim=0)
        all_sims.append((speech_words[i][1].item(), sim.item()))  
all_sims = np.array(all_sims)
mean_sims = []
for i in range(n_minute * 6):
    mean_sims.append(all_sims[(all_sims[:, 0] >= i * 10) & (all_sims[:, 0] < (i + 1) * 10), 1].mean())
plt.plot(mean_sims)
plt.xlabel("Input Length (unit: 10 seconds)")
plt.ylabel("Cosine Similarity of Speech and Text Embeddings")
plt.savefig("/home/siqiouyang/work/projects/sllama/notebooks/length_extrapolation_dev_llm_inf_cacheinfmodel.png".format(n_minute))