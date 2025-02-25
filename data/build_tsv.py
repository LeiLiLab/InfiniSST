import yaml
from data.utils import write_tsv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="/compute/babel-14-5/siqiouya/en-zh/",
                    help="Root directory containing the data")
parser.add_argument("--split", type=str, default="train",
                    help="Data split (train/valid/test)")
parser.add_argument("--src-lang", type=str, default="en",
                    help="Source language code")
parser.add_argument("--tgt-lang", type=str, default="zh", 
                    help="Target language code")

args = parser.parse_args()

root = args.root
split = args.split
src_lang = args.src_lang
tgt_lang = args.tgt_lang

with open(f"{root}/data/{split}/txt/{split}.yaml") as f:
    manifests = yaml.safe_load(f)
with open(f"{root}/data/{split}/txt/{split}.{src_lang}", "r") as r:
    src_texts = [l.strip() for l in r.readlines() if l.strip() != '']
with open(f"{root}/data/{split}/txt/{split}.{tgt_lang}", "r") as r:
    tgt_texts = [l.strip() for l in r.readlines() if l.strip() != '']

samples = []
ted_id = ""
id_in_ted = 0
for i, manifest in enumerate(manifests):
    cur_ted_id = manifest['wav'].split('.')[0]
    if cur_ted_id != ted_id:
        ted_id = cur_ted_id
        id_in_ted = 0
    else:
        id_in_ted += 1

    segment_id = f"{ted_id}_{id_in_ted}"

    offset = int(manifest['offset'] * 16000)
    duration = int(manifest['duration'] * 16000)
    segment_path = f"{root}/data/{split}/wav/{manifest['wav']}:{offset}:{duration}"

    samples.append({
        "id": segment_id,
        "audio": segment_path,
        "n_frames": duration,
        "speaker": manifest['speaker_id'],
        "src_text": src_texts[i],
        "tgt_text": tgt_texts[i],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    })
write_tsv(samples, f"{root}/{split}.tsv")