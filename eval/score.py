import os
import json
import re
import types
import logging
import argparse
from natsort import natsorted
import sacrebleu
from simuleval.evaluator.scorers.latency_scorer import LAALScorer, ALScorer
from elapsed import update_log_file  # Import the function from elapsed.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate latency and BLEU for original and corrected logs.")
parser.add_argument("--log_path", type=str, required=True, help="Path to the original instances.log file.")
parser.add_argument("--lang", type=str, default="zh", help="Language for scoring.")
args = parser.parse_args()

def load_instances_log(log_path):
    """
    Load the instances from the given log file.
    
    Args:
        log_path (str): Path to the instances.log file.
        
    Returns:
        list: A list of instances with predictions and references.
    """
    instances = []
    hyps = []
    refs = []

    try:
        with open(log_path, 'r') as r:
            for line in r.readlines():
                line = line.strip()
                if line != '':
                    d = json.loads(line)
                    d['prediction'] = re.sub(r' \s*</s>\s*$', '', d['prediction'])  # Remove </s> at the end
                    instance = types.SimpleNamespace(**d)
                    if args.lang == 'zh':
                        instance.reference_length = len(instance.reference)
                    else:
                        instance.reference_length = len(instance.reference.split(" "))
                    instances.append(instance)
                    hyps.append(instance.prediction)
                    refs.append(instance.reference)
    except FileNotFoundError:
        logging.error(f"File not found: {log_path}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {log_path}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return instances, hyps, refs

def evaluate_latency_and_bleu(instances, hyps, refs):
    """
    Evaluate BLEU and LAAL scores for the instances.
    
    Args:
        instances (list): List of instances with predictions and references.
        hyps (list): List of hypothesis sentences.
        refs (list): List of reference sentences.
    
    Returns:
        tuple: BLEU score, LAAL score, and LAAL_CA score.
    """
    # Initialize scorers
    scorer_c = LAALScorer(computation_aware=True)
    scorer = LAALScorer()
    scorer_al = ALScorer()
    scorer_al_c = ALScorer(computation_aware=True)

    bleu = sacrebleu.corpus_bleu(
        hyps, 
        [refs], 
        tokenize='zh' if args.lang == 'zh' else '13a',
    ).score

    laal_c_acc, laal_acc, n = 0, 0, 0
    al_c_acc, al_acc = 0, 0
    for instance in instances:
        try:
            al_c, al = scorer_al_c.compute(instance), scorer_al.compute(instance)
            laal_c, laal = scorer_c.compute(instance), scorer.compute(instance)
            al_c_acc += al_c
            al_acc += al
            laal_c_acc += laal_c
            laal_acc += laal
            n += 1
        except Exception as inner_e:
            logging.warning(f"Error in instance computation: {inner_e}")
            continue

    laal_c_avg = laal_c_acc / n if n > 0 else float('nan')
    laal_avg = laal_acc / n if n > 0 else float('nan')
    al_c_avg = al_c_acc / n if n > 0 else float('nan')
    al_avg = al_acc / n if n > 0 else float('nan')

    return bleu, laal_avg, laal_c_avg, al_avg, al_c_avg

def process_log_files(log_path):
    """
    Process both original and corrected instance logs and compare the LAAL scores.
    
    Args:
        log_path (str): Path to the original instances.log file.
    
    Returns:
        None
    """
    corrected_log_path = 'instance_corrected.log'

    # Generate corrected log file
    update_log_file(log_path, corrected_log_path)

    # Evaluate original log
    logging.info("Evaluating original instances.log:")
    original_instances, original_hyps, original_refs = load_instances_log(log_path)
    print("hypotheses: ", len(original_hyps), original_hyps)
    print("references: ", len(original_refs), original_refs) 
    original_bleu, original_laal_avg, original_laal_c_avg, al, al_c = evaluate_latency_and_bleu(original_instances, original_hyps, original_refs)

    logging.info(f"Original BLEU: {original_bleu:.1f}")
    logging.info(f"Original LAAL: {original_laal_avg:.0f} ms")
    logging.info(f"Original LAAL_CA: {original_laal_c_avg:.0f} ms")
    logging.info(f"Original AL: {al:.0f} ms")
    logging.info(f"Original AL_CA: {al_c:.0f} ms")

    # Evaluate corrected log
    logging.info("Evaluating corrected instance_corrected.log:")
    corrected_instances, corrected_hyps, corrected_refs = load_instances_log(corrected_log_path)
    corrected_bleu, corrected_laal_avg, corrected_laal_c_avg, corrected_al, corrected_al_c = evaluate_latency_and_bleu(corrected_instances, corrected_hyps, corrected_refs)

    logging.info(f"Corrected BLEU: {corrected_bleu:.1f}")
    logging.info(f"Corrected LAAL: {corrected_laal_avg:.0f} ms")
    logging.info(f"Corrected LAAL_CA: {corrected_laal_c_avg:.0f} ms")
    logging.info(f"Corrected AL: {corrected_al:.0f} ms")
    logging.info(f"Corrected AL_CA: {corrected_al_c:.0f} ms")

if __name__ == "__main__":
    process_log_files(args.log_path)