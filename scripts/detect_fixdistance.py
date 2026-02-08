import numpy as np
import torch
import tqdm
import argparse
import json
from utils import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
from rewrite_machine import PrefixSampler, get_regen_samples
import os

def compute_total_logprob_from_logits(logits, labels, pad_index):
    """
    返回未归一的总 log-prob（sum over non-pad tokens）。
    logits: [B, T, V], labels: [B, T]
    """
    lprobs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    # gather true-token logprobs
    labels_expanded = labels.unsqueeze(-1)  # [B, T, 1]
    token_logprobs = lprobs.gather(dim=-1, index=labels_expanded).squeeze(-1)  # [B, T]
    mask = (labels != pad_index).float()  # [B, T]
    total = (token_logprobs * mask).sum(dim=1)  # [B]
    return total  # summed log-prob per example

def sequence_total_logprob(tokenizer, model, input_ids, attention_mask):
    """
    计算整个序列 input_ids 的 total log-prob
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
    # shift for causal language modeling: predict input_ids[:,1:] from logits[:,:-1]
    shifted_logits = logits[:, :-1, :]  # [B, T-1, V]
    shifted_labels = input_ids[:, 1:]  # [B, T-1]
    pad_id = tokenizer.pad_token_id
    total_lp = compute_total_logprob_from_logits(shifted_logits, shifted_labels, pad_id)  # [B]
    return total_lp  # [B]

def compute_sequence_logprob(tokenizer, model, texts, device):
    """
    计算一批文本的 total log-prob（sum over tokens） under the base model.
    返回 tensor shape [len(texts)]，每个是 log p(text).
    """
    pad_id = tokenizer.pad_token_id

    # Tokenize full texts
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    input_ids = encoded["input_ids"]  # [B, T]
    attention_mask = encoded["attention_mask"]

    # Use your existing helper: sequence_total_logprob expects full input_ids and mask
    total_logp = sequence_total_logprob(tokenizer, model, input_ids, attention_mask)  # [B]
    lengths = attention_mask.sum(dim=1).clamp(min=1).float()  # [B]

    return total_logp / lengths  # summed log-prob per example

def get_rewrite_gpt_simple_statistic(tokenizer, model, text, regens):
    """
    用新的统计量 score = z - (1/K) sum_i tilde_z_i，
    其中 z = log p(X)， tilde_z_i = log p(rewrite_i)
    """
    # 1. 计算 z = log p(X)  和  each \tilde z_i = log p(\tilde X_i)
    z_tensor = compute_sequence_logprob(tokenizer, model, [text], args.device)  # [1]
    tilde_zs = compute_sequence_logprob(tokenizer, model, regens, args.device)  # [K]

    # 2. 统计量： z - mean(tilde_zs)
    score = z_tensor.mean() - tilde_zs.mean()  # scalar
    return score.item()

def experiment(args):
    tokenizer = load_tokenizer(args.base_model, args.cache_dir)
    model = load_model(args.base_model, args.device, args.cache_dir)
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # n_samples = 2
    # evaluate criterion
    name = "fixdistance"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    criterion_fn = get_rewrite_gpt_simple_statistic
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        original_crit = criterion_fn(tokenizer, model, original_text, rewrite_text[idx]['rewrite_original'])
        sampled_crit = criterion_fn(tokenizer, model, sampled_text, rewrite_text[idx]['rewrite_sampled'])
        # result
        results.append({
            "original_crit": original_crit,
            "sampled_crit": sampled_crit
        })

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results], 'samples': [x["sampled_crit"] for x in results]}
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
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_gpt-4o_expand")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_prompt/data/xsum_gpt-4o_expand")
    parser.add_argument('--regen_number', type=int, default=4)
    parser.add_argument('--rewrite_model', type=str, default="gemma-9b-instruct")
    parser.add_argument('--base_model', type=str, default="gemma-9b-instruct")
    parser.add_argument('--batch_size', type=int, default=4)
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
