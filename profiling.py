import os
import time
start_time = time.time()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch, transformers
import torch.nn as nn
from eval.utils import disable_torch_init
from model.model import SpeechLlamaForCausalLM, SpeechLlamaModel, SpeechLlamaConfig
from model.utils import KeywordsStoppingCriteria
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
import conversation as conversation_lib
from conversation import SeparatorStyle

import requests

import torch.nn.functional as F

import importlib
import numpy as np
from fairseq.models.speech_to_text import lengths_to_padding_mask
from train.uni_wav2vec_monkey_patch import replace_uni_train, replace_uni_decode, uni_self_attn_forward, uni_w2v2_extract_features

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()

device = args.device

# loading model

transformers.set_seed(998244353)
# torch.use_deterministic_algorithms(True)

args = argparse.Namespace()
args.model_name = '/data/user_data/siqiouya/runs/stage2-uni-waco'
args.length_adapter_path = os.path.join(args.model_name, 'length_adapter.bin')
args.mlp_adapter_path = os.path.join(args.model_name, 'mlp_adapter.bin')
args.speech_tower_path = os.path.join(args.model_name, 'speech_tower.bin')

load_type = torch.float32
disable_torch_init()
model_name = os.path.expanduser(args.model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_name,
    padding_side="right",
    use_fast=False,
)
config = json.load(open(os.path.join(args.model_name, 'config.json')))
config['large_model'] = True
update_config = os.path.join(args.model_name, 'config_large.json')
json.dump(config, open(update_config, 'w'), indent=2)
# replace_llama_attn_with_flash_attn()

replace_uni_train()

model = SpeechLlamaForCausalLM.from_pretrained(args.model_name,
                                                torch_dtype=load_type,
                                                low_cpu_mem_usage=True,
                                                device_map=device,
                                                config=update_config,).eval()

device_input = device_output = device

length_after_ssl, length_after_adp = model.model.initialize_speech_modules(
    speech_tower_path='/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt',
    speech_tower_type=None,
    len_adapter_channels=model.config.len_adapter_channels,
    len_adapter_kernel_sizes=model.config.len_adapter_kernel_sizes,
    ssl_fintuned=model.config.ssl_fintuned,
)
model.model.speech_tower.to(dtype=load_type, device=device_input)

length_adapter_weights = torch.load(args.length_adapter_path, map_location='cpu')
mlp_adapter_weights = torch.load(args.mlp_adapter_path, map_location='cpu')
speech_tower_weights = torch.load(args.speech_tower_path, map_location='cpu')


model.model.mm_length_adapter.load_state_dict(length_adapter_weights)
model.model.mm_mlp_adapter.load_state_dict(mlp_adapter_weights)
model.model.speech_tower.load_state_dict(speech_tower_weights)

model.model.mm_length_adapter.to(dtype=load_type, device=device_input).eval()
model.model.mm_mlp_adapter.to(dtype=load_type, device=device_input).eval()
model.model.speech_tower.to(dtype=load_type, device=device_input).eval()


import sys
sys.path.append('/home/siqiouya/work/SimulEval')
from eval.agents.tt_waitk_sllama_word import S2TAgentStates
from eval.agents.tt_waitk_sllama_word_incremental import IncrementalS2TAgentStates

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

f = open("profiling.log", "w")
load_time = time.time()
print('loading time', load_time - start_time, file=f)

# normal run

def process(states):
    source = torch.tensor(states.source).to(
        device=model.device, dtype=model.dtype
    )
    speech_batch = _collate_frames([source], is_audio_input=True)
    n_frames = torch.tensor([source.size(0)], dtype=torch.long)
    # source = F.layer_norm(source, source.size())
    speech_lens = length_after_adp(length_after_ssl(n_frames))

    to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
    to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]

    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], to_adds[0])
    conv.append_message(conv.roles[1], None)
    prompt_inputs = conv.get_prompt()

    inputs = tokenizer([prompt_inputs])
    input_ids = inputs.input_ids[0] + states.target_ids
    input_ids_tensor = torch.as_tensor([input_ids]).to(device)
    model.model.speech_features_extracted = False

    with torch.inference_mode():
        output = model.model(
            attention_mask=None, # input_ids_tensor.ne(tokenizer.pad_token_id),
            input_ids=input_ids_tensor,
            speech_batch=speech_batch,
            src_lengths=n_frames.to(device=model.device),
            after_lens=speech_lens.to(device=model.device),
        )
        # output = model.model.speech_tower.extract_features(speech_batch, None)
        
    return output

# states3 = S2TAgentStates([])
# states3.source_finished = False
# states3.source_sample_rate = 16000
# states3.source = states2.source + np.random.rand(5120).tolist()

for i in range(10):
    states1 = S2TAgentStates([])
    states1.source_finished = False
    states1.source_sample_rate = 16000
    states1.source = np.random.rand(25600).tolist()

    states2 = S2TAgentStates([])
    states2.source_finished = False
    states2.source_sample_rate = 16000
    states2.source = states1.source + np.random.rand(5120).tolist()
    states2.target_ids = [0]

    states3 = S2TAgentStates([])
    states3.source_finished = False
    states3.source_sample_rate = 16000
    states3.source = states2.source + np.random.rand(5120).tolist()
    states3.target_ids = [0, 1]

    o1 = process(states1)
    normal_time_1 = time.time()
    print('normal_time_1', normal_time_1 - start_time, file=f)
    o2 = process(states2)
    normal_time_2 = time.time()
    print('normal_time_2', normal_time_2 - start_time, file=f)
    o3 = process(states3)
    normal_time_3 = time.time()
    print('normal_time_3', normal_time_3 - start_time, file=f)

# uni run

def incremental_process(states):
    source = torch.tensor(states.source).to(
        device=model.device, dtype=model.dtype
    )
    speech_batch = _collate_frames([source], is_audio_input=True)
    n_frames = torch.tensor([source.size(0)], dtype=torch.long)
    speech_lens = length_after_adp(length_after_ssl(n_frames))

    to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
    to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]

    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], to_adds[0])
    conv.append_message(conv.roles[1], None)
    prompt_inputs = conv.get_prompt()

    inputs = tokenizer([prompt_inputs])
    input_ids = inputs.input_ids[0] + states.target_ids
    input_ids_tensor = torch.as_tensor([input_ids]).to(device)
    model.model.speech_features_extracted = False

    with torch.inference_mode():
        output = model.model(
            attention_mask=input_ids_tensor.ne(tokenizer.pad_token_id),
            input_ids=input_ids_tensor,
            speech_batch=speech_batch,
            src_lengths=n_frames.to(device=model.device),
            after_lens=speech_lens.to(device=model.device),
            states=states,
            use_cache=True
        )
        # output = uni_w2v2_extract_features(
        #     model.model.speech_tower,
        #     speech_batch, 
        #     None,
        #     past_key_values=states.w2v2_past_key_values,
        #     past_features=states.w2v2_past_features,
        # )
        # states.w2v2_past_features = output["x"]
        
    # states.num_frames_read = len(states.source)

    return output

replace_uni_decode()
for i in range(10):

    states1 = S2TAgentStates([])
    states1.source_finished = False
    states1.source_sample_rate = 16000
    states1.source = np.random.rand(25600).tolist()

    states2 = S2TAgentStates([])
    states2.source_finished = False
    states2.source_sample_rate = 16000
    states2.source = states1.source + np.random.rand(5120).tolist()

    states3 = S2TAgentStates([])
    states3.source_finished = False
    states3.source_sample_rate = 16000
    states3.source = states2.source + np.random.rand(5120).tolist()

    inc_states = IncrementalS2TAgentStates([], [], None, None, -1, 0)
    inc_states.source_sample_rate = 16000
    inc_states.source_finished = False
    inc_states.w2v2_past_key_values = [
        {} for _ in range(model.model.speech_tower.cfg.encoder_layers)
    ]
    inc_states.source = states1.source
    io1 = incremental_process(inc_states)
    uni_time_1 = time.time()
    print('uni_time_1', uni_time_1 - start_time, file=f)
    inc_states.source = states2.source
    inc_states.target_ids = [0]
    io2 = incremental_process(inc_states)
    uni_time_2 = time.time()
    print('uni_time_2', uni_time_2 - start_time, file=f)
    inc_states.source = states3.source
    inc_states.target_ids = [0, 1]
    io3 = incremental_process(inc_states)
    uni_time_3 = time.time()
    print('uni_time_3', uni_time_3 - start_time, file=f)

f.close()