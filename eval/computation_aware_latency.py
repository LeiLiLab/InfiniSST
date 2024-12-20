import os
import csv
import json
import types
import argparse
from natsort import natsorted
import sacrebleu
from simuleval.evaluator.instance import Instance
from simuleval.evaluator.scorers.latency_scorer import LAALScorer
from fairseq.examples.speech_to_text.data_utils import load_df_from_tsv
from comet import download_model, load_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", type=str, required=True)
args = parser.parse_args()

# if '30s' in args.dirname:
#     df = load_df_from_tsv('/data/user_data/siqiouya/dataset/must-c-v1.0/en-es/tst-COMMON_30s.tsv')
# else:
#     df = load_df_from_tsv('/data/user_data/siqiouya/dataset/must-c-v1.0/en-es/tst-COMMON.tsv')
# src_texts = df['src_text'].tolist()

scorer_c = LAALScorer(computation_aware=True)
scorer = LAALScorer()

# os.makedirs('/data/user_data/siqiouya/cache', exist_ok=True)
# comet_model_path =  download_model("Unbabel/XCOMET-XXL", saving_directory='/data/user_data/siqiouya/cache')
# comet_model = load_from_checkpoint(comet_model_path)

dirname = args.dirname
sub_dirs = natsorted(os.listdir(dirname))
for sub_dir in sub_dirs:
    sub_dir_full = os.path.join(dirname, sub_dir)
    
    instances = []
    instances_log_path = os.path.join(sub_dir_full, 'instances.log')

    try:
        hyps = []
        refs = []
        with open(instances_log_path, 'r') as r:
            for line in r.readlines():
                line = line.strip()
                if line != '':
                    d = json.loads(line)
                    instance = types.SimpleNamespace(**d)
                    instance.reference_length = len(instance.reference.split(" "))
                    instances.append(instance)
                    hyps.append(instance.prediction)
                    refs.append(instance.reference)
        
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

        ## comet
        # comet_data = [
        #     {
        #         "src": src_texts[i],
        #         "mt" : hyps[i],
        #         "ref": refs[i]
        #     }
        #     for i in range(len(hyps))
        # ]
        # comet_output = comet_model.predict(comet_data, batch_size=7, gpus=1)
        # comet_score = comet_output.system_score
        
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

        print(sub_dir, ':')
        print('  ', 'BLEU    {:.1f}'.format(bleu))
        # print('  ', 'COMET   {:.2f}'.format(comet_score))
        print('  ', 'LAAL    {:.0f} ms'.format(laal_avg))
        print('  ', 'LAAL_CA {:.0f} ms'.format(laal_c_avg))
        print()
    except:
        pass