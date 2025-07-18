import os
import sys
import json
import types
import torch
from comet import download_model, load_from_checkpoint

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

lang = sys.argv[2]

data_root = f"/compute/babel-14-5/siqiouya/en-{lang}/"

try:
    samples = read_tsv(os.path.join(data_root, f"tst-COMMON_st_{lang}.tsv")) # TODO: tsv file of the full ted data
except:
    samples = read_tsv(os.path.join(data_root, f"tst-COMMON.tsv")) # TODO: tsv file of the full ted data

srcs = [s["src_text"] for s in samples]

with open(os.path.join(sys.argv[1], 'hyp'), "r") as r:
    hyps = r.read().strip().split("\n")

with open(os.path.join(sys.argv[1], 'ref'), "r") as r:
    refs = r.read().strip().split("\n")

hs = []
rs = []
for hyp, ref in zip(hyps, refs):
    idx, h = hyp.split('\t')
    idx, r = ref.split('\t')
    hs.append((h, int(idx)))
    rs.append((r, int(idx)))

hs.sort(key=lambda x: x[1])
rs.sort(key=lambda x: x[1])

hs = [h[0].strip() for h in hs]
rs = [r[0].strip() for r in rs]

import sacrebleu

bleu = sacrebleu.corpus_bleu(hs, [rs], tokenize="zh" if lang == "zh" else "13a")
print("BLEU", bleu.score)

model_names = ["XCOMET-XL", "XCOMET-XXL"]

comet_scores = []

for model_name in model_names:
    comet_model_path = download_model(f"Unbabel/{model_name}", saving_directory=f"/compute/babel-13-9/siqiouya/{model_name}/")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_model.to("cuda")

    comet_data = [
        {
            "src": srcs[i],
            "mt" : hs[i],
            "ref": rs[i]
        }
        for i in range(len(srcs))
    ]

    comet_output = comet_model.predict(comet_data, batch_size=4, gpus=1)
    comet_scores.append(comet_output.system_score)
    print(f"{model_name}: {comet_output.system_score * 100:.2f}")

    del comet_model
    torch.cuda.empty_cache()

print(f"XCOMET-Ensemble: {sum(comet_scores)/len(comet_scores) * 100:.2f}")

# python compute_comet_offline.py /compute/babel-5-23/siqiouya/runs/en-zh/8B-s2-bi-v3.5/last.ckpt/offline_beam4/tst-COMMON/ zh