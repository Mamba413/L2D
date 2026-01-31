# Copyright (c) Jin Zhu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import tqdm
import argparse
import json
from model import load_tokenizer, load_model
from utils import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from threading import Thread
from scipy.spatial.distance import cdist
from skdim.id import MLE, TwoNN

MIN_SUBSAMPLE = 40 
INTERMEDIATE_POINTS = 7
MINIMAL_CLOUD = 80

def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10
    
    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty
        
        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)
        
    return s.item()

def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')

class PHD():
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=3, n_points=7, n_points_min=3):
        '''
        Initializes the instance of PH-dim computer
        Parameters:
        1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
        2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
        3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
        4) n_points --- Number of subsamples to be drawn at each subsample
        5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        '''
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False
        self.distance_matrix = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        if self.distance_matrix:
            return W[random_indices][:, random_indices]
        else:
            return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points
               
            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                if self.distance_matrix:
                    reruns[i] = prim_tree(tmp, self.alpha)
                else:
                    reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)   
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
        
    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40, dist=False):
        '''
        Computing the PH-dim 
        Parameters:
        1) X --- point cloud of shape (n_points, n_features), or precomputed distance matrix (n_points, n_points) if parameter dist set to 'True'
        2) y --- fictional parameter to fit with Sklearn interface
        3) min_points --- size of minimal subsample to be drawn
        4) max_points --- size of maximal subsample to be drawn
        5) point_jump --- step between subsamples
        6) dist --- bool value whether X is a precomputed distance matrix
        '''
        self.distance_matrix = dist
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)

def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')

def get_phd_single(text, tokenizer, model, PHD_solver):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outp = model(**inputs)
    
    # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
    mx_points = inputs['input_ids'].shape[1] - 2

    
    mn_points = MIN_SUBSAMPLE
    step = ( mx_points - mn_points ) // INTERMEDIATE_POINTS
        
    return PHD_solver.fit_transform(outp[0][0].cpu().numpy()[1:-1], min_points=mn_points, max_points=mx_points - step, point_jump=step)

def get_mle_single(text, tokenizer, model, MLE_solver):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outp = model(**inputs)

    return MLE_solver.fit_transform(outp[0][0].cpu().numpy()[1:-1])

def experiment(args):
    # load model and tokenizer
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.device, args.cache_dir)
    model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # n_samples = 2
    # evaluate criterion
    name = "ide_{}"
    if args.solver == 'PHD':
        name = name.format('phd')
        solver = PHD()
    elif args.solver == 'MLE':   
        name = name.format('mle')
        solver = MLE()
    elif args.solver == 'TwoNN':   
        name = name.format('twonn')
        solver = TwoNN()
    criterion_fn = get_phd_single if args.solver == 'PHD' else get_mle_single

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        original_crit = criterion_fn(original_text, tokenizer, model, solver)
        sampled_crit = criterion_fn(sampled_text, tokenizer, model, solver)
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_falcon-7b-instruct_rewrite")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_prompt/data/xsum_falcon-7b-instruct_rewrite")
    parser.add_argument('--model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--solver', type=str, default='PHD', choices=['PHD', 'MLE', 'TwoNN'])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
