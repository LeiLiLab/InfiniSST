import os, sys
from fairseq import checkpoint_utils, options, scoring, tasks, utils

with open(os.path.join(sys.argv[1], 'hyp'), 'r') as f:
    hyps = f.readlines()
    hyps = [hyp for hyp in hyps if hyp.startswith('ted_')]
with open(os.path.join(sys.argv[1], 'ref'), 'r') as f:
    refs = f.readlines()

scorer = scoring.build_scorer('sacrebleu', None)

for hyp, ref in zip(hyps, refs):
    hyp = hyp.split('\t')[1].strip()#.strip('"')
    ref = ref.split('\t')[1].strip()#.strip('"')
    scorer.add_string(ref, hyp)

print(scorer.result_string())
