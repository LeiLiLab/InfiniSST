import os, sys
from sacrebleu import BLEU

with open(os.path.join(sys.argv[1], 'hyp'), 'r') as f:
    hyps = [l for l in f.readlines() if len(l.split('\t')) > 1]
with open(os.path.join(sys.argv[1], 'ref'), 'r') as f:
    refs = [l for l in f.readlines() if len(l.split('\t')) > 1]

scorer = BLEU(tokenize='13a')

hs = []
rs = []
for hyp, ref in zip(hyps, refs):
    hs.append(hyp.split('\t')[1].strip())
    rs.append(ref.split('\t')[1].strip())

print(scorer.corpus_score(hs, [rs]))
