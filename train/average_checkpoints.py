import os
import argparse
import torch
import json
from collections import defaultdict

file_dict = {'7B':['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'],
              '13B': ['pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin']}

def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector weights')
    parser.add_argument('--model_name_or_paths', type=str, nargs='+', help='model folder')
    parser.add_argument('--extracted_name', type=str, help='extracted module name')
    parser.add_argument('--size', type=str, help='model size')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    file_names = file_dict[args.size]  
    dir = os.path.dirname(args.model_name_or_paths[0])
    os.makedirs(os.path.join(dir, args.output), exist_ok=True)
    for name in file_names:
        avg = None
        for model_name_or_path in args.model_name_or_paths:
            ckpt = torch.load(os.path.join(model_name_or_path, name), map_location='cpu')
            if avg is None:
                avg = ckpt
            else:
                for k in avg.keys():
                    avg[k] += ckpt[k]
        for k in avg.keys():
            avg[k] /= len(args.model_name_or_paths)
        torch.save(avg, os.path.join(dir, args.output, name))

