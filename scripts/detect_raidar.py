import os
import json
import numpy as np
from ImBD.dataset import CustomDatasetRewrite

from metrics import get_roc_metrics, get_precision_recall_metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from fuzzywuzzy import fuzz
import torch
from torch.utils.data import DataLoader, Subset
import argparse
import random
import time
from tqdm import tqdm
from rewrite_machine import PrefixSampler, get_regen_samples
from utils import load_data

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    # Extract n-grams from the list of tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    # Find common words
    common_words = common_elements(tokens1, tokens2)

    # Find common n-grams (let's say up to 3-grams for this example)
    common_ngrams = set()

    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 3-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]

def get_data_stat(data_loader, human=True, verbose=False):

    data_stats = []
    total_len = len(data_loader)
    for idxx, each in enumerate(data_loader):
        if human:
            text = each[0][0]
            text_rewrite = [x[0] for x in each[2]]
        else:
            text = each[1][0]
            text_rewrite = [x[0] for x in each[3]]
        raw = tokenize_and_normalize(text)
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue
        else:
            if verbose:
                print(idxx, total_len)

        all_text = [text]
        all_text.extend(text_rewrite)

        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for i in range(ngram_num)]
        cnt = 0
        whole_combined = ''
        for pp in range(len(all_text)):
            whole_combined += (' ' + all_text[pp])
            
            res = calculate_sentence_common(text, all_text[pp])
            statistic_res[pp] = res
            all_statistic_res = sum_for_list(all_statistic_res, res)

            ratio_fzwz[pp] = [fuzz.ratio(text, all_text[pp]), fuzz.token_set_ratio(text, all_text[pp])]
            cnt += 1
        
        each_stat = {}
        each_stat['input'] = text
        each_stat['fzwz_features'] = ratio_fzwz
        each_stat['common_features'] = statistic_res
        each_stat['avg_common_features'] = [a / cnt for a in all_statistic_res]
        each_stat['common_features_ori_vs_allcombined'] = calculate_sentence_common(text, whole_combined)

        data_stats.append(each_stat)

    return data_stats

def get_feature_vec(input_json):
    all_list = []
    for idxx, each in enumerate(input_json):   
        try:
            raw = tokenize_and_normalize(each['input'])
            r_len = len(raw)*1.0
        except:
            import pdb; pdb.set_trace()
        each_data_fea  = []

        if r_len == 0:
            continue
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue

        each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
        for ek in each['common_features'].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])
        
        each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])
        for ek in each['fzwz_features'].keys():
            each_data_fea.extend(each['fzwz_features'][ek])

        all_list.append(np.array(each_data_fea))

    all_list = np.vstack(all_list)
    return all_list

def train_classifier(human_stat, llm_stat):
    llm_all = get_feature_vec(llm_stat)
    human_all = get_feature_vec(human_stat)

    X_train = np.concatenate((human_all, llm_all), axis=0)
    y_train = np.concatenate((np.zeros(human_all.shape[0]), np.ones(llm_all.shape[0])), axis=0)

    # Neural network
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
    clf.fit(X_train, y_train)
    return scaler, clf

def classifier_eval(scaler, clf, texts_stats):
    X_test = get_feature_vec(texts_stats)

    X_test = scaler.transform(X_test)
    prob_predict = clf.predict_proba(X_test)[:, 1]
    return prob_predict   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='exp_prompt/data/squad_claude-3-5-haiku_expand&exp_prompt/data/squad_claude-3-5-haiku_rewrite&exp_prompt/data/squad_claude-3-5-haiku_polish&exp_prompt/data/writing_claude-3-5-haiku_expand&exp_prompt/data/writing_claude-3-5-haiku_rewrite&exp_prompt/data/writing_claude-3-5-haiku_polish')
    parser.add_argument('--rewrite_model', type=str, default="gemma-9b-instruct")
    parser.add_argument('--regen_number', type=int, default=2, help="rewrite number for each input")
    parser.add_argument('--eval_dataset', type=str, default="./exp_prompt/data/xsum_claude-3-5-haiku_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_claude-3-5-haiku_rewrite")
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    name = 'raidar'
    ngram_num = 4
    cutoff_start = 0
    cutoff_end = 6000000

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    ## load data and rewrite if necessary
    if "&" in args.train_dataset:
        data_name_list = args.train_dataset.split('&')
    else:
        data_name_list = [args.train_dataset]
    rewrite_data_name_list = [x.replace("/data/", "/results/") + f".rewrite_{args.regen_number}" for x in data_name_list]
    all_rewrited = all([os.path.exists(x + ".json") for x in rewrite_data_name_list])
    if not all_rewrited:
        sampler = PrefixSampler(args)
        for data_name, rewrite_data_name in zip(data_name_list, rewrite_data_name_list):
            if not os.path.exists(rewrite_data_name + ".json"):
                data = load_data(data_name)
                n_samples = len(data["sampled"])
                # n_samples = 2
                rewrite_text = []
                for idx in tqdm(range(n_samples), desc=f"Rewriting {data_name}"):
                    # original text
                    original_text = data["original"][idx]
                    rewrite_original = get_regen_samples(sampler, original_text)
                    # sampled text
                    sampled_text = data["sampled"][idx]
                    rewrite_sampled = get_regen_samples(sampler, sampled_text)

                    rewrite_text.append({'rewrite_original': rewrite_original, 'rewrite_sampled': rewrite_sampled})

                rewrite_texts_file = f'{rewrite_data_name}.json'
                with open(rewrite_texts_file, 'w') as fout:
                    json.dump(rewrite_text, fout, indent=2)
                    print(f'Rewritten texts saved into {rewrite_texts_file}')
    train_data = CustomDatasetRewrite(data_json_dir=args.train_dataset, args=args)
    subset_indices = torch.randperm(len(train_data))
    train_subset = Subset(train_data, subset_indices)

    if "&" in args.eval_dataset:
        data_name_list = args.eval_dataset.split('&')
    else:
        data_name_list = [args.eval_dataset]
    rewrite_data_name_list = [x.replace("/data/", "/results/") + f".rewrite_{args.regen_number}" for x in data_name_list]
    all_rewrited = all([os.path.exists(x + ".json") for x in rewrite_data_name_list])
    if not all_rewrited:
        sampler = PrefixSampler(args)
        for data_name, rewrite_data_name in zip(data_name_list, rewrite_data_name_list):
            if not os.path.exists(rewrite_data_name + ".json"):
                data = load_data(data_name)
                n_samples = len(data["sampled"])
                rewrite_text = []
                for idx in tqdm(range(n_samples), desc=f"Rewriting {data_name}"):
                    # original text
                    original_text = data["original"][idx]
                    rewrite_original = get_regen_samples(sampler, original_text)
                    # sampled text
                    sampled_text = data["sampled"][idx]
                    rewrite_sampled = get_regen_samples(sampler, sampled_text)

                    rewrite_text.append({'rewrite_original': rewrite_original, 'rewrite_sampled': rewrite_sampled})

                rewrite_texts_file = f'{rewrite_data_name}.json'
                with open(rewrite_texts_file, 'w') as fout:
                    json.dump(rewrite_text, fout, indent=2)
                    print(f'Rewritten texts saved into {rewrite_texts_file}')
    val_data = CustomDatasetRewrite(data_json_dir=args.eval_dataset, args=args)
    n_samples = len(val_data)
    val_data = Subset(val_data, torch.randperm(len(val_data)))

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=False)
    eval_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    human_stat = get_data_stat(train_loader, human=True)
    llm_stat = get_data_stat(train_loader, human=False)
    scaler, clf = train_classifier(human_stat, llm_stat)
    start = time.perf_counter()
    eval_human_stat = get_data_stat(eval_loader, human=True)
    human_pred = classifier_eval(scaler, clf, eval_human_stat)
    eval_llm_stat = get_data_stat(eval_loader, human=False)
    llm_pred = classifier_eval(scaler, clf, eval_llm_stat)
    eval_time = time.perf_counter() - start
    eval_time = eval_time / (2*n_samples)
    
    print("-----------------------------------------------------------------")
    predictions = {'real': human_pred.tolist(), 'samples': llm_pred.tolist()}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name} ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results
    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc, 
        'compute_info': {'eval_time': eval_time, }
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')


