import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

import sacrebleu
import soundfile as sf

import copy
import yaml
import math
import torch
from tqdm.notebook import tqdm

from IPython.display import display, Audio

def read_logs(path):
    logs = []
    with open(path, "r") as r:
        for l in r.readlines():
            l = l.strip()
            if l != "":
                logs.append(json.loads(l))
    return logs

def write_logs(logs, path):
    with open(path, "w") as w:
        for log in logs:
            w.write(json.dumps(log) + "\n")

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