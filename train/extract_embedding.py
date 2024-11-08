import os
import argparse
import torch
import json
from collections import defaultdict
from model.model import SpeechEncoder, SpeechLlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Embedding')
    parser.add_argument('--model_name_or_path', type=str, help='model folder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()    

    llm = SpeechLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        device_map='cpu',
    )
    embedding = llm.model.embed_tokens
    torch.save(embedding, os.path.join(args.model_name_or_path, 'embedding.pt'))
