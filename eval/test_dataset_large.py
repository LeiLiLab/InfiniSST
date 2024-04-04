import argparse, sys, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch, transformers
from eval.utils import disable_torch_init
from model.model import SpeechLlamaForCausalLM, SpeechLlamaConfig
from model.utils import KeywordsStoppingCriteria
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
import conversation as conversation_lib
from conversation import SeparatorStyle
from train.uni_wav2vec_monkey_patch import replace_uni_train

import os
import requests

import torch.nn.functional as F

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

prompt_list = ['<speech_here>']

def eval_model(args):
    load_type = torch.float16 # torch.float32
    # Model
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
    if not os.path.exists(update_config):
        json.dump(config, open(update_config, 'w'), indent=2)
    # replace_llama_attn_with_flash_attn()
    # change wav2vec to uni-directional
    replace_uni_train(args.blocksize)
    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name,
                                                   torch_dtype=load_type,
                                                   low_cpu_mem_usage=True,
                                                   device_map='auto',
                                                   config=update_config,).eval()
    if 'model.embed_tokens' in model.hf_device_map.keys():
        device_input = 'cuda:' + str(model.hf_device_map['model.embed_tokens'])    
        device_output = 'cuda:' + str(model.hf_device_map['lm_head'])    
    else:
        device_input = 'cuda'  
        device_input = 'cuda'
    length_after_ssl, length_after_adp = model.model.initialize_speech_modules(
        speech_tower_path=model.config.speech_tower_path,
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
    
    model.model.mm_length_adapter.to(dtype=load_type, device=device_input)
    model.model.mm_mlp_adapter.to(dtype=load_type, device=device_input)
    model.model.speech_tower.to(dtype=load_type, device=device_input)

    model.model.config.inference = True
        
    test_dataset = PromptSpeechToTextDatasetCreator.from_tsv(args.data_path, args.data_split)
  
    if not os.path.exists(os.path.join(args.result, args.data_split)):
        os.makedirs(os.path.join(args.result, args.data_split))
        
    ref_file = open(os.path.join(args.result, args.data_split, "ref"), "w")
    hyp_file = open(os.path.join(args.result, args.data_split, "hyp"), "w")
    conv = conversation_lib.default_conversation.copy()

    for test_data in tqdm(test_dataset):
        source, ref, id = test_data.source, test_data.target, test_data.id                  
        speech_batch = _collate_frames([source], is_audio_input=True)
        n_frames = torch.tensor([source.size(0)], dtype=torch.long)
        speech_lens = length_after_adp(length_after_ssl(n_frames))
    
        to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]
    
        
        conv.messages = []
        conv.append_message(conv.roles[0], to_adds[0])
        conv.append_message(conv.roles[1], None)
        prompt_inputs = conv.get_prompt()
        inputs = tokenizer([prompt_inputs])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)  

        # extract new speech features 
        model.model.speech_features_extracted = False
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    attention_mask=input_ids.ne(tokenizer.pad_token_id),
                    input_ids=input_ids.to(device=device_input),
                    speech_batch=speech_batch.to(dtype=load_type, device=device_input),
                    src_lengths=n_frames.to(device=device_input),
                    after_lens=speech_lens.to(device=device_input),
                    #do_sample=True,
                    #temperature=0.8,
                    num_beams=args.beam,
                    max_new_tokens=500,
                    stopping_criteria=[stopping_criteria])    
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            print(outputs)
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            outputs = ""
            print(e)
        print(f"{id} decode complete,\nref:{ref} \nhyp:{outputs}")
        print(f"{id}\t{ref}", file=ref_file)
        print(f"{id}\t{outputs}", file=hyp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--length-adapter-path", type=str, required=True)
    parser.add_argument("--mlp-adapter-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--speech-tower-path", type=str, required=True)
    parser.add_argument("--blocksize", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
