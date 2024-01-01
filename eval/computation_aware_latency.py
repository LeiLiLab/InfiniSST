import json
import types
import argparse
from simuleval.evaluator.instance import Instance
from simuleval.evaluator.scorers.latency_scorer import LAALScorer

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

path = args.path

instances = []
with open(path, 'r') as r:
    for line in r.readlines():
        line = line.strip()
        if line != '':
            d = json.loads(line)
            instance = types.SimpleNamespace(**d)
            instance.reference_length = len(instance.reference.split(" "))
            instances.append(instance)

scorer_c = LAALScorer(computation_aware=True)
scorer = LAALScorer()

laal_c_acc, laal_acc, n = 0, 0, 0
for instance in instances:
    try:
        laal_c, laal = scorer_c.compute(instance), scorer.compute(instance)
        laal_c_acc += laal_c
        laal_acc += laal
        n += 1
    except:
        continue
laal_c_avg = laal_c_acc / n
laal_avg = laal_acc / n

print('Computation aware LAAL {:.0f} ms'.format(laal_c_avg))