import argparse, sys, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch, transformers
from eval.utils import disable_torch_init
from model.model import SpeechLlamaForCausalLM, SpeechLlamaConfig
from model.utils import KeywordsStoppingCriteria
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
import torch.utils.data as data

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

def eval_model(args):
    load_type = torch.float16
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    
    config = json.load(open(os.path.join(args.model_name, 'config.json')))
    config['large_model'] = True
    update_config = os.path.join(args.model_name, 'config_large.json')
    if not os.path.exists(update_config):
        json.dump(config, open(update_config, 'w'), indent=2)

    if args.uni:
        print("Replace forward function in UniWav2Vec2")
        replace_uni_train()

    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name,
                                                   torch_dtype=load_type,
                                                   device_map='auto',
                                                   config=update_config,).eval()
    
    if 'model.embed_tokens' in model.hf_device_map.keys():
        device_input = 'cuda:' + str(model.hf_device_map['model.embed_tokens'])    
        device_output = 'cuda:' + str(model.hf_device_map['lm_head'])    
    else:
        device_input = 'cuda'  
        device_output = 'cuda'
        
    model.length_after_ssl, model.length_after_adp = model.model.initialize_speech_modules(
        speech_tower_path="/data/user_data/yuanjinw/models/wav2_vec_vox_960h_pl.pt",
        speech_tower_type=args.speech_tower_type,
        len_adapter_channels=model.config.len_adapter_channels,
        len_adapter_kernel_sizes=model.config.len_adapter_kernel_sizes,
        ssl_fintuned=model.config.ssl_fintuned,
        stage1_complete=model.config.stage1_complete,
    )

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
    
    # Simple batching
    batch_size = args.batch_size
    num_samples = len(test_dataset)
    
    if not os.path.exists(os.path.join(args.result, args.data_split)):
        os.makedirs(os.path.join(args.result, args.data_split))
        
    ref_file = open(os.path.join(args.result, args.data_split, "ref"), "w")
    hyp_file = open(os.path.join(args.result, args.data_split, "hyp"), "w")

    for i in tqdm(range(0, num_samples, batch_size)):
        # Get batch items
        batch = [test_dataset[j] for j in range(i, min(i + batch_size, num_samples))]
        
        # Prepare batch data
        sources = [item.source for item in batch]
        refs = [item.target for item in batch]
        ids = [item.id for item in batch]
        
        # Process speech input in batch
        speech_batch = _collate_frames(sources, is_audio_input=True)
        n_frames = torch.tensor([source.size(0) for source in sources], dtype=torch.long)
        speech_lens = model.length_after_adp(model.length_after_ssl(n_frames))
    
        # Create speech tokens for batch
        speech_tokens = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        speech_tokens = [DEFAULT_SPEECH_START_TOKEN + tokens + DEFAULT_SPEECH_END_TOKEN for tokens in speech_tokens]
        
        # Create instruction prompts for batch
        instruction = f"Translate the following speech from {args.source_lang} to {args.target_lang}:"
        prompts = [f"{instruction} {tokens}" for tokens in speech_tokens]
    
        # Tokenize input batch
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()

        # Set up stopping criteria
        stop_str = "<|end_of_text|>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Generate outputs for batch
        model.model.speech_features_extracted = False
        model.eval()
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    attention_mask=attention_mask,
                    input_ids=input_ids.to(device=device_input),
                    speech_batch=speech_batch.to(dtype=load_type, device=device_input),
                    src_lengths=n_frames.to(device=device_input),
                    after_lens=speech_lens.to(device=device_input),
                    num_beams=args.beam,
                    max_new_tokens=500,
                    stopping_criteria=[stopping_criteria],
                    no_repeat_ngram_size=3,          
                    repetition_penalty=1.2,          
                    length_penalty=1.0)
                
            # Process outputs for batch
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            
            # Clean outputs
            outputs = [output.strip() for output in outputs]
            outputs = [output[:-len(stop_str)] if output.endswith(stop_str) else output for output in outputs]
            outputs = [output.strip() for output in outputs]
            
        except Exception as e:
            outputs = [""] * len(batch)
            print(e)
            
        # Write results for batch
        for id, ref, output in zip(ids, refs, outputs):
            print(f"{id} decode complete,\nref:{ref} \nhyp:{output}")
            print(f"{id}\t{ref}", file=ref_file)
            print(f"{id}\t{output}", file=hyp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--length-adapter-path", type=str, required=True)
    parser.add_argument("--mlp-adapter-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--speech-tower-path", type=str, required=True)
    parser.add_argument("--speech-tower-type", type=str, required=True)
    parser.add_argument("--uni", action='store_true', default=False)
    parser.add_argument("--source-lang", type=str, default="English",
                       help="Source language name")
    parser.add_argument("--target-lang", type=str, default="French",
                       help="Target language name")    
    args = parser.parse_args()

    eval_model(args)