import os
from preprocess.utils import read_tsv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tsv-path', type=str, required=True)
args = parser.parse_args()

root = os.path.dirname(args.tsv_path)
split = os.path.basename(args.tsv_path).split('.')[0]
samples = read_tsv(args.tsv_path)

def key_func(x):
    _, offset, _ = x['audio'].split(':')
    offset = int(offset)

    ted_id = int(x['id'].split('_')[1])

    return (ted_id, offset)

sorted_samples = sorted(
    samples, 
    key=key_func
)

ted_id = -1
document = ""
documents = []
for x in sorted_samples:
    cur_ted_id = int(x['id'].split('_')[1])
    if cur_ted_id != ted_id:
        documents.append((ted_id, document))
        ted_id = cur_ted_id
        document = x['tgt_text']
    else:
        document += ' ' + x['tgt_text']
documents.append((ted_id, document))
documents = documents[1:]

with open(os.path.join(root, split + '_full.source'), 'w') as w_source, open(os.path.join(root, split + '_full.target'), 'w') as w_target:
    for ted_id, document in documents:
        w_source.write(os.path.join(root, "data", split, "wav", f"ted_{ted_id}.wav") + '\n')
        w_target.write(document + '\n')