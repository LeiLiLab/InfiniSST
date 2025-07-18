import csv, argparse, os
import pandas as pd
from pathlib import Path
from examples.speech_to_text.data_utils import (
    save_df_to_tsv,
)

MANIFEST_COLUMNS = ["id", "audio", "tgt_text", "task"]

def load_samples_from_tsv(root: str, split: str):
    tsv_path = Path(root) / f"{split}.tsv"
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
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
    if len(samples) == 0:
        raise ValueError(f"Empty manifest: {tsv_path}")
    return samples
    
parser = argparse.ArgumentParser()
parser.add_argument("--tsv_root", type = str,)
parser.add_argument("--merge_tsv_name", nargs='+')
parser.add_argument("--tgt_tsv_name", type = str, )
args = parser.parse_args()

manifest = {c: [] for c in MANIFEST_COLUMNS}

for tsv in args.merge_tsv_name:
    samples = load_samples_from_tsv(args.tsv_root, tsv)
    for sample in samples:
        manifest["id"].append(sample["id"])
        manifest["audio"].append(sample["audio"])
        manifest["tgt_text"].append(sample["tgt_text"])
        if "src_text" in sample.keys():
            manifest["task"].append("st")
        else:
            manifest["task"].append("asr")
            
save_df_to_tsv(
    pd.DataFrame.from_dict(manifest), os.path.join(args.tsv_root, args.tgt_tsv_name)
)
