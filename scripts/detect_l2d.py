from AdaDist.dataset import CustomDatasetRewrite
from AdaDist.model import AdaDist
from AdaDist.engine import evaluate_model, train_dist
import torch
from torch.utils.data import Subset
import argparse
import numpy as np
import random
import time
import json
from rewrite_machine import PrefixSampler, get_regen_samples
from utils import load_data, GpuMem
from tqdm import tqdm
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--a', type=int, default=1, help="accumulation steps")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('--datanum', type=int, default=200, help="num of training data")
    parser.add_argument('--rewrite_model', type=str, default="gemma-9b-instruct")
    parser.add_argument('--regen_number', type=int, default=2, help="rewrite number for each input")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size for rewriting")
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--eval_after_train', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--save_trained', action="store_true")
    parser.add_argument('--eval_dataset', type=str, default="./exp_prompt/data/xsum_gpt-4o_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_gpt-4o_rewrite")
    parser.add_argument('--base_model', type=str, default="gemma-9b")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--train_dataset', type=str, default='./exp_prompt/data/squad_gpt-4o_polish&./exp_prompt/data/writing_gpt-4o_expand')
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    print(f"Running with args: {args}")
    set_seed(args.seed)

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
    if len(train_data) < args.datanum:
        args.datanum = len(train_data)
    subset_indices = torch.randperm(len(train_data))[:args.datanum]
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
    val_data = CustomDatasetRewrite(data_json_dir=args.eval_dataset, args=args)

    ## load model
    model = AdaDist(args.base_model, cache_dir=args.cache_dir)
    criterion_fn_name = 'auc'
    model.set_criterion_fn(criterion_fn_name)
    if args.from_pretrained:
        print(f"Loading ckpt from {args.from_pretrained}...")
        model.from_pretrained(args.from_pretrained)

    if args.eval_only:
        print("Evaluating model before tuning...")
        d = evaluate_model(model, val_data, args.device)
        output_path = f"{args.output_file}.l2d.json"
        with open(output_path, "w") as j:
            json.dump(d, j, indent=2)
        print(f"Results saved to {output_path}.")
    else:
        tracker = GpuMem()
        print('Fine-tuning model...')
        start = time.perf_counter()
        with tracker:
            model = train_dist(
                model, 
                train_subset,
                device=args.device, 
                ckpt_dir=f"./scripts/AdaDist/ckpt/",
                args=args
            )
        pre_time = time.perf_counter() - start
        pre_memory = tracker.memory_usage()
        
        if args.eval_after_train:
            print("Evaluating model after tuning...")
            start = time.perf_counter()
            with tracker:
                d = evaluate_model(model, val_data, args.device)
            eval_time = time.perf_counter() - start
            eval_time = eval_time / (len(val_data) << 1)
            eval_memory = tracker.memory_usage()
            d['compute_info'] = {'pre_time': pre_time, 'eval_time': eval_time, 
                                'pre_memory': pre_memory, 'eval_memory': eval_memory,}
            output_path = f"{args.output_file}.l2d.json"
            with open(output_path, "w") as j:
                json.dump(d, j, indent=2)
            print(f"Results saved to {output_path}.")