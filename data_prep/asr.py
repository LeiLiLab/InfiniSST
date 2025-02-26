import os
import csv
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sacrebleu
import soundfile as sf

import yaml
from tqdm import tqdm

def read_logs(path):
    logs = []
    with open(path, "r") as r:
        for l in r.readlines():
            l = l.strip()
            if l != "":
                logs.append(json.loads(l))
    return logs

def read_wav(wav_path):
    wav_path, offset, duration = wav_path.split(':')
    offset = int(offset)
    duration = int(duration)
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
    from IPython.display import display, Audio
    display(Audio(read_wav(audio_path)[0], rate=16000))

# Whisper ASR
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# load model and processor
model_id = "openai/whisper-large-v3"
torch_dtype = torch.float16
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to('cuda')
processor = AutoProcessor.from_pretrained(model_id)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

batch_size = 32
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=batch_size,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device='cuda',
)

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--num_splits', type=int, required=True, help='Number of splits to divide the data into')
parser.add_argument('--split_id', type=int, required=True, help='Which split to process (0-based index)')
parser.add_argument('--tsv_path', type=str, required=True, help='Path to the tsv file')
args = parser.parse_args()

samples = read_tsv(args.tsv_path)

# Calculate split size and indices
total_samples = len(samples)
split_size = total_samples // args.num_splits
start_idx = args.split_id * split_size
end_idx = start_idx + split_size if args.split_id < args.num_splits - 1 else total_samples

# Get samples for this split
split_samples = samples[start_idx:end_idx]

asrs = []
for i in tqdm(range(0, len(split_samples), batch_size)):
    batch = split_samples[i:i + batch_size]
    wav_paths = [x['audio'] for x in batch]
    offsets = [int(x.split(':')[1]) for x in wav_paths]
    durations = [int(x.split(':')[2]) for x in wav_paths]
    sources, rates = zip(*[read_wav(x) for x in wav_paths])
    
    # Find max length in batch
    max_len = max(max(len(x) for x in sources), 691200)

    # Pad each source with zeros to match max length
    padded_sources = []
    for source in sources:
        padding = np.zeros(max_len - len(source))
        padded = np.concatenate([source, padding])
        padded_sources.append(padded)

    output = pipe(
        padded_sources,
        generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
    )
    transcriptions = [t['text'] for t in output]
    asrs.extend([t.strip() for t in transcriptions])

# Save results for this split
tsv_dirname = os.path.dirname(args.tsv_path)
output_path = f"{tsv_dirname}/asr.{args.split_id}"
with open(output_path, 'w') as f:
    for asr in asrs:
        f.write(asr + '\n')