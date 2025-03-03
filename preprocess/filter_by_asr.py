import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

import sacrebleu
import soundfile as sf

import yaml
from tqdm import tqdm

from IPython.display import display, Audio

def read_logs(path):
    logs = []
    with open(path, "r") as r:
        for l in r.readlines():
            l = l.strip()
            if l != "":
                logs.append(json.loads(l))
    return logs

def read_wav(wav_path):
    if ':' in wav_path:
        wav_path, offset, duration = wav_path.split(':')
        offset = int(offset)
        duration = int(duration)
    else:
        offset = 0
        duration = -1
    source, rate = sf.read(wav_path, start=offset, frames=duration)
    return source, rate

def read_tsv(tsv_path):
    import csv
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    return samples

def write_tsv(samples, tsv_path):
    with open(tsv_path, "w") as w:
        writer = csv.DictWriter(
            w,
            samples[0].keys(),
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        writer.writerows(samples)

def play(audio_path):
    display(Audio(read_wav(audio_path)[0], rate=16000))


import glob
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Filter ASR results based on WER.")
    parser.add_argument('--tsv-path', type=str, required=True, help='Path to the input TSV file.')
    return parser.parse_args()

args = parse_args()

# Path to the ASR files
base_path = os.path.dirname(args.tsv_path)

# Get all asr.* files

# Read and concatenate all ASR results
all_asrs = []
for i in range(8):
    with open(os.path.join(base_path, f"asr.{i}")) as f:
        asrs = [line.strip() for line in f.readlines() if line.strip() != ""]
        all_asrs.extend(asrs)

print(f"Total ASR transcriptions: {len(all_asrs)}")

samples = read_tsv(args.tsv_path)

from evaluate import load
wer_scorer = load("wer")

all_asrs = np.array(all_asrs)
wers = []
for i in tqdm(range(len(samples))):
    asr_orig = samples[i]['src_text'].replace('"', '').lower()
    asr_whisper = all_asrs[i].lower()
    wer = wer_scorer.compute(predictions=[asr_orig], references=[asr_whisper])
    wers.append(wer)
wers = np.array(wers)
samples = np.array(samples)

# special case 1
special_words = [
    "(Music)", 
    "(Laughter)", 
    "(Applause)", 
]
remove_mask = wers > 0.4
for i in range(len(samples)):
    if remove_mask[i]:
        if len(all_asrs[i].split(' ')) <= 3:
            if any(w in samples[i]['src_text'] for w in special_words) or samples[i]['src_text'] == "":
                remove_mask[i] = False

filtered_samples = samples[~remove_mask]
print(f"Number of samples filtered: {len(samples) - len(filtered_samples)}")

write_tsv(filtered_samples, args.tsv_path.replace('.tsv', '_filtered.tsv'))