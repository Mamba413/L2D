import numpy as np
import torch
import tqdm
from RevisedDetect.bart_score import BARTScorer
from rewrite_machine import PrefixSampler, get_regen_samples
import argparse
import json
from utils import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
import os

def get_revised_gpt_simple_statistic(text, model, regens=None):
    sim_score = model.score(regens, [text], batch_size=1)
    return sim_score.item()

def experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## below we set rewrite_prompt to 'l2d' to ensure the fairness of various rewrite based method
    ## set rewrite_prompt to 'bartscore' applies the orginal implementation for BartScore 
    sampler = PrefixSampler(args, rewrite_prompt='l2d')  
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # evaluate criterion
    name = "bartscorer" ## existing methods (just resampling one time)
    model = BARTScorer(device=args.device, model_name="facebook/bart-large-cnn", cache_dir=args.cache_dir)
    criterion_fn = get_revised_gpt_simple_statistic

    rewrite_texts_file = f'{args.output_file}.rewrite_{args.regen_number}.json'
    if os.path.exists(rewrite_texts_file):
        print(f"Load LLM-rewritten texts file {rewrite_texts_file}")
        with open(rewrite_texts_file, 'r') as fin:
            rewrite_text = json.load(fin)
    else:
        sampler = PrefixSampler(args, rewrite_prompt='l2d')
        rewrite_text = []
        for idx in tqdm.tqdm(range(n_samples), desc=f"Generating LLM-rewritten texts"):
            rewrite_original = get_regen_samples(sampler, data["original"][idx])  # list of length K
            rewrite_sampled = get_regen_samples(sampler, data["sampled"][idx])  # list of length K
            rewrite_text.append({'rewrite_original': rewrite_original, 'rewrite_sampled': rewrite_sampled})

        with open(rewrite_texts_file, 'w') as fout:
            json.dump(rewrite_text, fout, indent=2)
            print(f'Rewritten texts saved into {rewrite_texts_file}')
    
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        # original text
        original_text = data["original"][idx]
        rewrite_original_text = [rewrite_text[idx]['rewrite_original'][0]]
        original_crit = criterion_fn(original_text, model, rewrite_original_text)
        # sampled text
        sampled_text = data["sampled"][idx]
        rewrite_sampled_text = [rewrite_text[idx]['rewrite_sampled'][0]]
        sampled_crit = criterion_fn(sampled_text, model, rewrite_sampled_text)
        # result
        results.append({
            "original_crit": original_crit,
            "sampled_crit": sampled_crit
        })

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_gemma-9b-instruct_expand")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_prompt/data/xsum_gemma-9b-instruct_expand")
    parser.add_argument('--regen_number', type=int, default=4)
    parser.add_argument('--rewrite_model', type=str, default="mistralai-8b-instruct")
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
