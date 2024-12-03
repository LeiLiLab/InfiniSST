import os
import json
import argparse

from tqdm import tqdm

import torch
import transformers
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from eval.utils import disable_torch_init
from model.model_new import SpeechLlamaForCausalLM, SpeechLlamaConfig
from model.speech_encoder import SpeechEncoderHuBERTRope, SpeechEncoderW2V2RoPE
from model.utils import KeywordsStoppingCriteria
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem

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
    load_type = torch.bfloat16
    # Model
    disable_torch_init()

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

    model = SpeechLlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_type,
        device_map='auto',
        config=update_config
    ).eval()

    speech_encoder_args = [
        args.w2v2_path,
        args.ctc_finetuned,
        args.length_shrink_cfg,
        
        args.block_size,
        args.max_cache_size,
        model.model.embed_tokens.embedding_dim,
        None,
        bool(args.xpos)
    ]
    if args.w2v2_type == 'hubert':
        speech_encoder = SpeechEncoderHuBERTRope(*speech_encoder_args)
    else:
        speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
    
    speech_encoder_weights = torch.load(
        os.path.join(args.model_name, "speech_encoder.bin"),
        map_location='cpu'
    )
    speech_encoder.load_state_dict(speech_encoder_weights)
    speech_encoder.to(dtype=model.dtype, device=model.device)
    model.model.speech_encoder = speech_encoder

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
        speech_lens = model.model._get_feat_extract_output_lengths(n_frames)
    
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
        with torch.inference_mode():
            output_ids = model.generate(
                attention_mask=attention_mask,
                input_ids=input_ids.cuda(),
                speech_batch=speech_batch.to(dtype=model.dtype, device=model.device),
                src_lengths=n_frames.cuda(),
                after_lens=speech_lens.cuda(),
                num_beams=args.beam,
                max_new_tokens=500,
                stopping_criteria=[stopping_criteria],
                no_repeat_ngram_size=3,          
                repetition_penalty=1.2,          
                length_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        # Process outputs for batch
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        
        # Clean outputs
        outputs = [output.strip() for output in outputs]
        outputs = [output[:-len(stop_str)] if output.endswith(stop_str) else output for output in outputs]
        outputs = [output.strip() for output in outputs]
            
        # Write results for batch
        for id, ref, output in zip(ids, refs, outputs):
            print(f"{id} decode complete,\nref:{ref} \nhyp:{output}")
            print(f"{id}\t{ref}", file=ref_file)
            print(f"{id}\t{output}", file=hyp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--w2v2-path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--w2v2-type",
        type=str,
        default=None
    )
    parser.add_argument(
        "--ctc-finetuned",
        action="store_true"
    )
    parser.add_argument(
        "--length-shrink-cfg",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=12, # blocksize=1 means 80ms
    )
    parser.add_argument(
        "--max-cache-size",
        type=int, 
        default=125, # 125 * 0.08 = 1 second
    )
    parser.add_argument(
        "--xpos",
        type=int,
        default=1, # 1 for True, 0 for False
    )

    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source-lang", type=str, default="English",
                       help="Source language name")
    parser.add_argument("--target-lang", type=str, default="French",
                       help="Target language name")    
    args = parser.parse_args()

    eval_model(args)