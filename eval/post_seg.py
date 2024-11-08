import os
import json
import argparse

parser = argparse.ArgumentParser(description="Correct the log")
parser.add_argument("--dir", type=str, required=True, help="directory to the original instances.log file.")
parser.add_argument("--segmented-srcs", type=str, required=True, help="Path to the segmented sources.")
parser.add_argument("--segmented-refs", type=str, required=True, help="Path to the segmented references.")
args = parser.parse_args()

with open(os.path.join(args.dir, "instances.log.corrected"), "r") as r:
    logs = [json.loads(l) for l in r.readlines()]

with open(args.segmented_srcs, "r") as r:
    srcs_per_doc = r.read().strip().split("\n\n")

with open(args.segmented_refs, "r") as r:
    refs_per_doc = r.read().strip().split("\n\n")

hyps_per_doc = []
for i in range(len(logs)):
    with open(os.path.join(args.dir, "tmp/hyp.{}.seg".format(i)), "r") as r:
        hyps = r.read().strip()
    hyps_per_doc.append(hyps)

sum_laal = sum_laal_ca = 0
n_sent = 0
for log, srcs, refs, hyps in zip(logs, srcs_per_doc, refs_per_doc, hyps_per_doc):
    srcs = srcs.strip().split("\n")
    refs = refs.strip().split("\n")
    hyps = hyps.strip().split("\n")

    n_word_acc = 0
    for src, ref, hyp in zip(srcs, refs, hyps):
        hyp = hyp.strip()
        if hyp == "":
            continue
        n_word = len(hyp.split(" "))

        delays = log['delays'][n_word_acc : n_word_acc + n_word]
        elapsed = log['elapsed'][n_word_acc : n_word_acc + n_word]

        # print(len(log['elapsed']), n_word_acc + n_word, hyp.strip().split(" "))
        
        offset, duration = map(float, src.split(' '))
        offset *= 1000
        duration *= 1000

        s = s_ca = 0
        t = duration / max(n_word, len(ref.split(' ')))
        for i, (d, e) in enumerate(zip(delays, elapsed)):
            s += d - offset - (i + 1) * t
            s_ca += e - offset - (i + 1) * t
        laal_per_sent = s / len(delays)
        laal_ca_per_sent = s_ca / len(elapsed)

        if laal_per_sent < -100000:
            print(laal_per_sent)
            print(hyp, ref, sep="\n")

        sum_laal += laal_per_sent
        sum_laal_ca += laal_ca_per_sent

        n_sent += 1

        n_word_acc += n_word

print("LAAL    : {:.2f}".format(sum_laal / n_sent))
print("LAAL_CA*: {:.2f}".format(sum_laal_ca / n_sent))