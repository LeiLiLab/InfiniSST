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
# filter speaker information
import re
import csv
from collections import defaultdict

def extract_names_and_ted_talks(samples):
    ted_talk_dict = defaultdict(set)
    
    # Regex for extracting names: matches 'Firstname Lastname:' and initials like 'CA:' or 'RSW:'
    name_regex = re.compile(r'\b(?<!\")(Audience|Narrator|Video|Man|Woman|Bono|Voice|Announcer|Rives|George W\. Bush|Broadcasting|Boy|Professor|Engineer|Interviewer|Shereen El-Feki|Tina|Girl|Dad|Voice):|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*:|[A-Z]{1,3}:')

    error_samples = []
    cleaned_samples = []
        
    for sample in samples:
        ted_id = sample['id'].split('_')[1]
        names = name_regex.findall(sample['src_text'])
        cleaned_names = {name.strip(':').strip() for name in names}

        if len(cleaned_names) > 0:
            ted_talk_dict[ted_id].update(cleaned_names)
            error_samples.append(sample)
        else:
            cleaned_samples.append(sample)

    return ted_talk_dict, error_samples, cleaned_samples

# New product from Coke Japan: water salad. | New product from Coke Japan, water salad.
# Consider this: Make a decision to live a carbon-neutral life.
# Video: Don Blankenship: Let me be clear about it. | Let me be clear about it.
# Richard Koshalek: [Unclear] starts from 2004. | THANK YOU. # (2400 frames) delete short utterance
# DP: Wow.	| David Perry: Wow. | Wow. 
# DNA: 
# Stephen Pink's Girlfriend: | Stephen Pinks Freundin:
# Audience|Narrator|Video|Man|Woman|Bono|Voice|Announcer|Rives|George W. Bush|Broadcasting|Boy|Professor|Engineer|Interviewer|Shereen El-Feki|Tina|Girl|Dad|Voice
# Then: working concentrated, without being frazzled.	Dann: konzentriert arbeiten, ohne genervt zu werden.

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process TSV file for speaker extraction.')
    parser.add_argument('--tsv-path', type=str, help='Path to the input TSV file')
    return parser.parse_args()
args = parse_arguments()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('labse')

samples = read_tsv(args.tsv_path)
last_len = len(samples)
while True:
    ted_talk_data, error_samples, cleaned_samples = extract_names_and_ted_talks(samples)

    if len(error_samples) == 0:
        break

    srcs = []
    tgts = []
    for x in error_samples:
        src = x['src_text']
        tgt = x['tgt_text']

        src = src[:src.find(':')]
        if ':' in tgt:
            tgt = tgt[:tgt.find(':')]
        elif '：' in tgt:
            tgt = tgt[:tgt.find('：')]
        else:
            tgt = ""

        srcs.append(src)
        tgts.append(tgt)        

    src_embeddings = model.encode(srcs)
    tgt_embeddings = model.encode(tgts)

    sims = []
    for i in range(len(src_embeddings)):
        cosine_similarity = model.similarity(src_embeddings[i], tgt_embeddings[i]).item()
        sims.append(cosine_similarity)
    sims = np.array(sims)
    src_lens = [len(src.split(' ')) for src in srcs]
    tgt_lens = [len(tgt) if 'zh' in args.tsv_path else len(tgt.split(' ')) for tgt in tgts]
    corrected_samples = []
    for i in range(len(sims)):
        if re.search(r'One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|LG', srcs[i]):
            continue
        if srcs[i] != "" and tgts[i] != "" and src_lens[i] <= 3 and (tgt_lens[i] <= 3 or sims[i] > 0.5):
            # print(error_samples[i]['src_text'], error_samples[i]['tgt_text'], sims[i], sep='\n', end='\n\n')
            x = copy.deepcopy(error_samples[i])
            x['src_text'] = x['src_text'][len(srcs[i]) + 1:].strip()
            x['tgt_text'] = x['tgt_text'][len(tgts[i]) + 1:].strip()
            corrected_samples.append(x)

    samples = cleaned_samples + corrected_samples
ted_talk_data, error_samples, cleaned_samples = extract_names_and_ted_talks(samples)
write_tsv(samples, args.tsv_path.replace('.tsv', '_nospeaker.tsv'))