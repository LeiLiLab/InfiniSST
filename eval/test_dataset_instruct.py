import os
import json
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import transformers
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from eval.utils import disable_torch_init
from model.model_new import SpeechLlamaForCausalLM, SpeechLlamaConfig
from model.speech_encoder import (
    SpeechEncoderHuBERTRope, 
    SpeechEncoderW2V2RoPE, 
    SpeechEncoderW2VBERT2
)
from model.utils import KeywordsStoppingCriteria
from train.dataset import (
    PromptSpeechToTextDatasetCreator, 
    DataCollatorForSupervisedInstructDataset,
    SpeechSampler,
)

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

    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "9105"
    torch.distributed.init_process_group(
        rank=0, world_size=1, 
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    model = SpeechLlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_type,
        device_map='auto',
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
    elif args.w2v2_type == 'w2v-bert':
        speech_encoder = SpeechEncoderW2VBERT2(
            args.w2v2_path,
            args.length_shrink_cfg,
            args.block_size,
            args.max_cache_size,
            model.model.embed_tokens.embedding_dim,
        )
    else:
        speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
    
    # speech_encoder_weights = torch.load(
    #     os.path.join(args.model_name, "speech_encoder.bin"),
    #     map_location='cpu'
    # )
    # speech_encoder.load_state_dict(speech_encoder_weights)
    speech_encoder.to(dtype=model.dtype, device=model.device)
    model.model.speech_encoder = speech_encoder

    model.preprocess(tokenizer=tokenizer)

    state_dict = torch.load(args.state_dict_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    model.model.inference = True

    test_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
        args.data_path, args.data_split
    )
    data_collator = DataCollatorForSupervisedInstructDataset(
        tokenizer, 
        speech_encoder._get_feat_extract_output_lengths,
        args.source_lang,
        args.target_lang
    )

    test_sampler = SpeechSampler(
        test_dataset, 
        shuffle=False, 
        batch_size=args.batch_size, 
        min_ms=0,
        multiplier=1,
        filter=False,
        tokenizer=tokenizer
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_sampler=test_sampler, 
        collate_fn=data_collator
    )
    
    if not os.path.exists(os.path.join(args.result, args.data_split)):
        os.makedirs(os.path.join(args.result, args.data_split))
        
    ref_file = open(os.path.join(args.result, args.data_split, "ref"), "w")
    hyp_file = open(os.path.join(args.result, args.data_split, "hyp"), "w")

    for batch in tqdm(list(test_dataloader)[::-1]):
        refs = batch["target_text"]
        ids = batch["ids"]

        input_ids = batch["input_ids"]
        end_pos = (input_ids[0] == model.config.assist_token_id).nonzero()
        end_pos = [
            pos for pos in end_pos if input_ids[0, pos[0] - 1] == model.config.start_header_id
        ]
        end_pos = end_pos[0][0]
        input_ids = input_ids[:, : end_pos + 3]

        attention_mask = batch["attention_mask"][:, : end_pos + 3]
        speech_batch = batch["speech_batch"]
        n_frames = batch["src_lengths"]
        speech_lens = batch["after_lens"]
        
        # Set up stopping criteria
        # stop_str = "<|eot_id|>"
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Generate outputs for batch
        model.model.speech_features_extracted = False
        model.eval()
        max_new_tokens = n_frames.max() // 16000 * 7 + 20
        print("max_new_tokens", max_new_tokens.item(), "batch_size", len(ids))
        with torch.inference_mode():
            output_ids = model.generate(
                attention_mask=attention_mask.cuda(),
                input_ids=input_ids.cuda(),
                speech_batch=speech_batch.to(dtype=model.dtype, device=model.device),
                src_lengths=n_frames.cuda(),
                after_lens=speech_lens.cuda(),
                do_sample=False,
                num_beams=args.beam,
                max_new_tokens=max_new_tokens,
                # stopping_criteria=[stopping_criteria],
                no_repeat_ngram_size=3,          
                repetition_penalty=1.2,          
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # Process outputs for batch
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:-1], skip_special_tokens=True)
        
        # Clean outputs
        # outputs = [output.strip() for output in outputs]
        # outputs = [output[:-len(stop_str)] if output.endswith(stop_str) else output for output in outputs]
        # outputs = [output.strip() for output in outputs]
            
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
    parser.add_argument("--state-dict-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1500)
    parser.add_argument("--source-lang", type=str, default="English",
                       help="Source language name")
    parser.add_argument("--target-lang", type=str, default="French",
                       help="Target language name")    
    args = parser.parse_args()

    eval_model(args)