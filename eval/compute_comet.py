import os
import sys
import json
import types
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

comet_model = load_from_checkpoint("/compute/babel-7-5/siqiouya/xcomet_xxl/snapshots/bad20b47daa64c41a8b29f3d3016be75baf0d7b4/checkpoints/model.ckpt")
data_root = "/compute/babel-14-5/siqiouya/en-zh/"
samples = read_tsv(os.path.join(data_root, "tst-COMMON.tsv"))
srcs = [s["src_text"] for s in samples]

with open(sys.argv[1], "r") as r:
    instances = [json.loads(line) for line in r.readlines()]
comet_data = [
    {
        "src": srcs[i],
        "mt" : instances[i]['prediction'].strip(),
        "ref": instances[i]['reference'].strip()
    }
    for i in range(len(srcs))
]
comet_output = comet_model.predict(comet_data, batch_size=4, gpus=1)
print(comet_output.system_score)