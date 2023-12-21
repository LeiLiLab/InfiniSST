import os
import argparse
import torch
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector weights')
    parser.add_argument('--model_name_or_path', type=str, help='model folder')
    parser.add_argument('--extracted_name', type=str, help='extracted module name')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()    
    key_to_match = args.extracted_name
    loaded_weights = {}
    index_path = os.path.join(args.model_name_or_path, 'pytorch_model.bin.index.json')
    
    if os.path.exists(index_path):
        model_indices = json.load(open(index_path))
    
        ckpt_to_key = defaultdict(list)
        for k, v in model_indices['weight_map'].items():
            if key_to_match in k:
                ckpt_to_key[v].append(k)

        for ckpt_name, weight_keys in ckpt_to_key.items():
            ckpt = torch.load(os.path.join(args.model_name_or_path, ckpt_name), map_location='cpu')
            for k in weight_keys:
                save_k = k.replace('model.' + key_to_match + '.', '')
                loaded_weights[save_k] = ckpt[k]
    else:
        ckpt = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
        for k, v in ckpt.items():
            if key_to_match in k:
                save_k = k.replace('model.' + key_to_match + '.', '')
                loaded_weights[save_k] = ckpt[k]
                
    torch.save(loaded_weights, args.output)
