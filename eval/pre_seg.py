import os
import json
import argparse
from elapsed import update_log_file

parser = argparse.ArgumentParser(description="Correct the log")
parser.add_argument("--dir", type=str, required=True, help="directory to the original instances.log file.")
parser.add_argument("--segmented-refs", type=str, required=True, help="Path to the segmented references.")
parser.add_argument("--unit", type=str, default='word')
args = parser.parse_args()

input_path = os.path.join(args.dir, "instances.log")
output_path = os.path.join(args.dir, "instances.log.corrected")

update_log_file(input_path, output_path)

with open(output_path, "r") as r:
    logs = [json.loads(l) for l in r.readlines()]

with open(args.segmented_refs, "r") as r:
    refs_per_doc = r.read().strip().split("\n\n")

assert len(logs) == len(refs_per_doc)

tmp_dir = os.path.join(args.dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)

for i, (log, refs) in enumerate(zip(logs, refs_per_doc)):
    with open(os.path.join(tmp_dir, "hyp.{}".format(i)), "w") as w_hyp, open(os.path.join(tmp_dir, "ref.{}".format(i)), "w") as w_ref:
        if args.unit == 'word':
            w_hyp.write(log["prediction"])
            w_ref.write(refs)
        else:
            w_hyp.write(" ".join(log["prediction"]))
            w_ref.write(" ".join(refs))