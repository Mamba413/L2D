# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import os
import argparse
import json
import numpy as np
from scipy.stats import norm
from itertools import chain
import pandas as pd

def save_lines(lines, file):
    with open(file, 'w') as fout:
        fout.write('\n'.join(lines))

def get_auroc(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['roc_auc']

def get_sampled_mean(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return np.mean(res['predictions']['samples'])

def get_fpr_tpr(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['fpr'], res['metrics']['tpr']

def report_black_prompt_results(args):
    import os

    # Datasets to show (行)
    datasets = {
        'xsum': 'News',
        'squad': 'Wiki',
        'writing': 'Story',
    }
    # 模型（列块）
    source_models = {
        'claude-3-5-haiku': 'Claude-3.5',
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini',
    }
    # 任务子列
    task_list = ['rewrite', 'polish', 'expand']
    include_avg = True
    subcols = task_list + (['Avg.'] if include_avg else [])

    # 方法（行块）
    methods1 = {
        'likelihood': 'Likelihood',
        'lrr': 'LRR',
        'binoculars': 'Binoculars',
        'ide_mle': 'IDE',
    }
    methods2 = {
        'sampling_discrepancy_analytic': 'FDGPT',
        'bartscorer': 'BARTScore',
        'roberta-large-openai-detector': 'RoBERTa',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        'fixdistance': 'FD',
        # 'adarewritegpt_auc.rewrite_4': 'Ours',
        'l2d': 'L2D',
    }
    all_methods = {**methods1, **methods2}  # 按顺序合并

    # 读取单元 AUC
    def _get_auc(dataset_key, model_key, task_key, method_key):
        path = f'{args.result_path}/{dataset_key}_{model_key}_{task_key}.{method_key}.json'
        if os.path.exists(path):
            return get_auroc(path)
        return 0.0

    # 打印整张表格
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{3pt}")
    n_models = len(source_models)
    n_subcols = len(subcols)
    print("\\begin{tabular}{ll" + "c" * (n_models * n_subcols) + "}")
    print("\\toprule")

    # 表头第一行
    header_top = ["Dataset", "Method"]
    for model_name in source_models.values():
        header_top.append(f"\\multicolumn{{{n_subcols}}}{{c}}{{{model_name}}}")
    print(" & ".join(header_top) + " \\\\")

    # cmidrule 范围
    start_col = 3
    cmis = []
    for _ in source_models:
        end_col = start_col + n_subcols - 1
        cmis.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    print("".join(cmis))

    # 表头第二行
    header_sub = [" ", " "]
    for _ in source_models:
        header_sub.extend(subcols)
    print(" & ".join(header_sub) + " \\\\")
    print("\\midrule")

    # 遍历每个数据集
    our_auc = []
    fd_auc = []
    for d_idx, (d_key, d_name) in enumerate(datasets.items()):
        n_methods = len(all_methods)
        scores = np.zeros((n_methods, n_models, n_subcols), dtype=float)

        # 填 scores
        for m_idx, m_key in enumerate(source_models.keys()):
            for s_idx, sub in enumerate(subcols):
                if sub == 'Avg.':
                    vals = []
                    for t in task_list:
                        row_vals = []
                        for k_idx, method_key in enumerate(all_methods.keys()):
                            auc = _get_auc(d_key, m_key, t, method_key)
                            scores[k_idx, m_idx, s_idx-1 if s_idx > 0 else 0] = auc
                            row_vals.append(auc)
                        vals.append(row_vals)
                    vals = np.array(vals)
                    method_means = np.mean(vals, axis=0)
                    for k_idx in range(n_methods):
                        scores[k_idx, m_idx, s_idx] = method_means[k_idx]
                else:
                    for k_idx, method_key in enumerate(all_methods.keys()):
                        auc = _get_auc(d_key, m_key, sub, method_key)
                        scores[k_idx, m_idx, s_idx] = auc
                        if method_key == 'l2d':
                            our_auc.append(auc)
                        if method_key == "rewrite_gpt":
                            fd_auc.append(auc)

        # 高亮：第一名蓝色，第二名橙色
        best_idx_map = np.argmax(scores, axis=0)  # (n_models, n_subcols)
        second_idx_map = np.zeros_like(best_idx_map)
        for m_idx in range(n_models):
            for s_idx in range(n_subcols):
                col_vals = scores[:, m_idx, s_idx]
                best_idx = best_idx_map[m_idx, s_idx]
                tmp = col_vals.copy()
                tmp[best_idx] = -np.inf
                second_idx_map[m_idx, s_idx] = np.argmax(tmp)

        # 打印方法行
        method_items = list(all_methods.items())
        ours_idx = list(all_methods.keys()).index("l2d")

        for k_idx, (method_key, method_name) in enumerate(method_items):
            row_cells = []
            if k_idx == 0:
                row_cells.append(f"\\multirow{{{n_methods+1}}}{{*}}{{{d_name}}}")
            else:
                row_cells.append("")

            row_cells.append(method_name)

            for m_idx, _ in enumerate(source_models.keys()):
                for s_idx, _ in enumerate(subcols):
                    val = scores[k_idx, m_idx, s_idx]
                    cell = f"{val:.3f}"
                    if k_idx == best_idx_map[m_idx, s_idx]:
                        cell = f"\\cellcolor{{cyan!24}} {cell}"
                    elif k_idx == second_idx_map[m_idx, s_idx]:
                        cell = f"\\cellcolor{{orange!24}} {cell}"
                    row_cells.append(cell)

            print(" & ".join(row_cells) + " \\\\")

        # 相对提升幅度行
        abs_row_cells = ["", "\\textit{Abs. Gain (\%)}"]
        row_cells = ["", "\\textit{Rel. Gain (\%)}"]
        improve_abs_gain = []
        improve_gain = []
        for m_idx, _ in enumerate(source_models.keys()):
            improve_gain_tmp = []
            improve_abs_gain_tmp = []
            for s_idx, _ in enumerate(subcols):
                ours_val = scores[ours_idx, m_idx, s_idx]
                baseline_vals = [scores[k, m_idx, s_idx] for k in range(n_methods) if k != ours_idx]
                best_baseline = max(baseline_vals)
                if ours_val > 0 and best_baseline < 1.0:
                    rel_gain = 100 * (ours_val - best_baseline) / (1.0 - best_baseline)
                    abs_gain = 100 * (ours_val - best_baseline)
                else:
                    rel_gain = 0.0
                    abs_gain = 0.0
                if rel_gain > 0.0:
                    row_cells.append(f"{rel_gain:.1f}")
                    abs_row_cells.append(f"{abs_gain:.1f}")
                    improve_abs_gain_tmp.append(abs_gain)
                    improve_gain_tmp.append(rel_gain)
                else:
                    row_cells.append("---")
                    abs_row_cells.append("---")
            improve_gain.append(improve_gain_tmp)
            improve_abs_gain.append(improve_abs_gain_tmp)
        print("\\cline{2-14}")
        print(" & ".join(abs_row_cells) + " \\\\")
        print(" & ".join(row_cells) + " \\\\")
        print("\\midrule")
    
    mean_improve_gain = [f"{np.mean(x):.2f}" for x in improve_gain]
    mean_improve_abs_gain = [f"{np.mean(x):.2f}" for x in improve_abs_gain]
    mean_improve_gain = ", ".join(mean_improve_gain)
    mean_improve_abs_gain = ", ".join(mean_improve_abs_gain)
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{AUROC across datasets, models, and tasks; best method highlighted in \\colorbox{{cyan!24}}{{cyan}}, second best in \\colorbox{{orange!24}}{{orange}}. The average absolute gain are {mean_improve_abs_gain}, and relative gain are {mean_improve_gain}.}}\\label{{tab:exp-prompt}}")
    print("\\end{table}")
    
    gain_abs_fd = (np.array(our_auc) - np.array(fd_auc))
    gain_fd = (np.array(our_auc) - np.array(fd_auc)) / (1.0 - np.array(fd_auc))
    print(f"Average Gain over FD: Abs. Gain {np.mean(gain_abs_fd):.4f} Rel. Gain {np.mean(gain_fd):.4f}")

def report_black_prompt_results_simplified(args):
    import os
    import numpy as np

    # 数据集
    datasets = {
        'xsum': 'News',
        'squad': 'Wiki',
        'writing': 'Story',
    }
    # LLM
    source_models = {
        'claude-3-5-haiku': 'Claude-3.5',
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini',
    }
    # 任务
    task_list = ['rewrite', 'polish', 'expand']

    # 方法
    methods = {
        'likelihood': 'Likelihood',
        'lrr': 'LRR',
        'binoculars': 'Binoculars',
        'ide_mle': 'IDE',
        'sampling_discrepancy_analytic': 'FDGPT',
        'revised_gpt2': 'BARTScore',
        'roberta-large-openai-detector': 'RoBERTa',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        'rewrite_gpt': 'FD',
        'l2d': 'Ours',
    }

    def _get_auc(dataset_key, model_key, task_key, method_key):
        path = f'{args.result_path}/{dataset_key}_{model_key}_{task_key}.{method_key}.json'
        if os.path.exists(path):
            return get_auroc(path)
        return None   # 改成 None，便于过滤

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{l l c}")
    print("\\toprule")
    print("Dataset & Method & Avg$\\pm$Std \\\\")
    print("\\midrule")

    # 每个 dataset 打印一个 block
    for d_key, d_name in datasets.items():
        method_items = list(methods.items())
        n_methods = len(method_items)

        # 打印 dataset 名（multirow）
        print(f"\\multirow{{{n_methods}}}{{*}}{{{d_name}}}")

        for idx, (method_key, method_name) in enumerate(method_items):
            # 收集所有 AUC
            auc_list = []
            for model_key in source_models:
                for task_key in task_list:
                    auc = _get_auc(d_key, model_key, task_key, method_key)
                    if auc is not None:
                        auc_list.append(auc)

            if len(auc_list) == 0:
                mean_auc, std_auc = 0.0, 0.0
            else:
                mean_auc, std_auc = np.mean(auc_list), np.std(auc_list)

            # 表格行
            print(f" & {method_name} & {mean_auc:.3f}$\\pm${std_auc:.3f} \\\\")
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Simplified AUROC table: for each method we report a single mean±std across all models and tasks.}")
    print("\\end{table}")

def report_black_prompt_results_condensed(args):
    # 数据集
    datasets = {
        'xsum': 'News',
        'squad': 'Wiki',
        'writing': 'Story',
    }

    # LLM
    source_models = {
        'claude-3-5-haiku': 'Claude-3.5',
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini',
    }

    # 任务
    task_list = ['rewrite', 'polish', 'expand']

    # 方法
    methods = {
        'likelihood': 'Likelihood',
        'lrr': 'LRR',
        'binoculars': 'Binoculars',
        'ide_mle': 'IDE',
        'sampling_discrepancy_analytic': 'FDGPT',
        'revised_gpt2': 'BARTScore',
        'roberta-large-openai-detector': 'RoBERTa',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        'rewrite_gpt': 'FD',
        'l2d': 'Ours',
    }

    # 从 JSON 获取 AUC
    def _get_auc(dataset_key, model_key, task_key, method_key):
        path = f'{args.result_path}/{dataset_key}_{model_key}_{task_key}.{method_key}.json'
        if os.path.exists(path):
            return get_auroc(path)
        return None

    # latex 表格开始
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{3pt}")

    # 列名数量 = 9 columns + 2 (avg,std)
    total_cols = 2 + len(source_models) * len(task_list) + 2
    print("\\begin{tabular}{l l " + "c" * (total_cols - 2) + "}")
    print("\\toprule")

    # 顶部表头
    header_top = ["Dataset", "Method"]
    for model_name in source_models.values():
        header_top += [f"\\multicolumn{{3}}{{c}}{{{model_name}}}"]
    header_top += ["Avg", "Std"]
    print(" & ".join(header_top) + " \\\\")

    # cmidrule
    col_index = 3
    cmis = []
    for _ in source_models:
        cmis.append(f"\\cmidrule(lr){{{col_index}-{col_index+2}}}")
        col_index += 3
    print("".join(cmis))

    # 第二行表头（task names）
    header_sub = [" ", " "]
    for _ in source_models:
        header_sub += ["rewrite", "polish", "expand"]
    header_sub += [" ", " "]
    print(" & ".join(header_sub) + " \\\\")
    print("\\midrule")

    # 遍历每个 dataset
    for d_key, d_name in datasets.items():
        method_items = list(methods.items())
        n_methods = len(method_items)

        print(f"\\multirow{{{n_methods}}}{{*}}{{{d_name}}}")

        for idx, (method_key, method_name) in enumerate(method_items):

            auc_values_flat = []  # 用于 avg/std

            row_cells = ["", method_name]

            for model_key in source_models.keys():
                for task_key in task_list:
                    auc = _get_auc(d_key, model_key, task_key, method_key)
                    if auc is None:
                        row_cells.append("---")
                    else:
                        row_cells.append(f"{auc:.3f}")
                        auc_values_flat.append(auc)

            # Avg + Std（对 9 个值）
            if len(auc_values_flat) > 0:
                avg = np.mean(auc_values_flat)
                std = np.std(auc_values_flat)
            else:
                avg = std = 0.0

            row_cells.append(f"{avg:.3f}")
            row_cells.append(f"{std:.3f}")

            print(" & ".join(row_cells) + " \\\\")
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{AUROC across datasets with per-model per-task results, plus overall average and standard deviation.}")
    print("\\end{table}")
            
def report_diverse_results(args):
    report_rel_gain = True

    datasets = [
        'AcademicResearch', 'ArtCulture', 'Business', 'Code', 'EducationMaterial',
        'Entertainment', 'Environmental', 'Finance', 'FoodCusine', 'GovernmentPublic',
        'LegalDocument', 'LiteratureCreativeWriting', 'MedicalText', 'NewsArticle',
        'OnlineContent', 'PersonalCommunication', 'ProductReview', 'Religious',
        'Sports', 'TechnicalWriting', 'TravelTourism'
    ]
    source_models = {
        'GPT-3-Turbo': 'GPT-3.5 Turbo',
        'Llama-3-70B': 'Llama-3-70B-Instruct',
        'Gemini-1.5-Pro': 'Gemini 1.5 Pro',
        'GPT-4o': 'GPT-4o',
    }
    methods = {
        'likelihood': 'Likelihood',
        'lrr': 'LRR',
        'ide_mle': 'IDE',
        'revised_gpt2': 'BARTScore',
        'sampling_discrepancy_analytic': 'FDGPT',
        'binoculars': 'Binoculars',
        'roberta-large-openai-detector': 'RoBERTa',
        'radar': 'RADAR',
        'classification.bspline': 'ADGPT',
        # 'fair.raidar': 'RAIDAR',
        # 'fair.imbd': 'ImBD',
        # 'fair.l2d': 'Ours',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        'l2d': 'Ours',
    }

    # ours_key = 'fair.l2d'
    ours_key = 'l2d'

    def _get_method_aurocs(dataset, model, method, filter=''):
        result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
        if os.path.exists(result_file):
            return get_auroc(result_file)
        else:
            return 0.0

    our_aucs = []
    for model_key, model_name in source_models.items():
        print("==============================================")
        print(f"Model: {model_name}")
        print("==============================================")

        # build matrix: datasets × methods
        auc_matrix = []
        our_auc_one_data = []
        for dataset in datasets:
            row = []
            for method in methods:
                auc = _get_method_aurocs(dataset, model_key, method)
                row.append(auc)
                if method == ours_key:
                    our_auc_one_data.append(auc)
            auc_matrix.append(row)
        our_aucs.append(our_auc_one_data)

        auc_matrix = np.array(auc_matrix)
        avg_row = np.mean(auc_matrix, axis=0)
        std_row = np.std(auc_matrix, axis=0)

        # LaTeX table header
        header_methods = " & ".join(methods.values())
        print("\\begin{table}[H]")
        print("\\centering")
        if report_rel_gain:
            print(f"\\caption{{AUROC scores of various detectors for detecting text generated by {model_name}. We use google/gemma-2-9b-it as the rewriting and scoring model for implementing both rewrite- and logits-based methods. The highest scores are highlighted in \\colorbox{{cyan!24}}{{cyan}}, the second best in \\colorbox{{orange!24}}{{orange}}. The last column shows the relative gain of Ours over the best baseline.}}\\label{{tab:exp-data-{model_key}}}")
        else:
            print(f"\\caption{{AUROC scores of various detectors for detecting text generated by {model_name}. We use google/gemma-2-9b-it as the rewriting and scoring model for implementing both rewrite- and logits-based methods. The highest scores are highlighted in \\colorbox{{cyan!24}}{{cyan}}, the second best in \\colorbox{{orange!24}}{{orange}}.}}\\label{{tab:exp-data-{model_key}}}")
        print("\\begin{adjustbox}{width=\\textwidth}")
        print("\\setlength{\\tabcolsep}{3pt}")
        if report_rel_gain:
            print("\\begin{tabular}{l" + "c" * len(methods) + "|cc}")  # extra col for Rel. Gain
        else:
            print("\\begin{tabular}{l" + "c" * len(methods) + "}")
        print("\\toprule")
        if report_rel_gain:
            print(f"Dataset & {header_methods} & AG (\\%) & RG (\\%) \\\\")
        else:
            print(f"Dataset & {header_methods} \\\\")
        print("\\midrule")

        ours_idx = list(methods.keys()).index(ours_key)

        # dataset rows
        for i, dataset in enumerate(datasets):
            row_vals = []
            row = auc_matrix[i]
            # 找第一名 & 第二名
            best_idx = np.argmax(row)
            tmp = row.copy()
            tmp[best_idx] = -np.inf
            second_idx = np.argmax(tmp)

            for j, val in enumerate(row):
                cell = f"{val:.3f}"
                if j == best_idx:
                    cell = f"\\cellcolor{{cyan!24}} {cell}"
                elif j == second_idx:
                    cell = f"\\cellcolor{{orange!24}} {cell}"
                row_vals.append(cell)

            # 计算相对提升
            ours_val = row[ours_idx]
            baseline_vals = [row[j] for j in range(len(row)) if j != ours_idx]
            best_baseline = max(baseline_vals)
            if report_rel_gain:
                if ours_val > 0 and best_baseline < 1.0:
                    abs_gain = 100 * (ours_val - best_baseline)
                    rel_gain = 100 * (ours_val - best_baseline) / (1.0 - best_baseline)
                else:
                    abs_gain = 0.0
                    rel_gain = 0.0
                    
                if rel_gain > 0:
                    row_vals.append(f"{abs_gain:.3f}")
                    row_vals.append(f"{rel_gain:.1f}")
                else:
                    row_vals.append("---")
                    row_vals.append("---")

            print(dataset + " & " + " & ".join(row_vals) + " \\\\")

        # avg row
        avg_vals = []
        best_idx = np.argmax(avg_row)
        best_value = np.max(avg_row)
        tmp = avg_row.copy()
        tmp[best_idx] = -np.inf
        second_idx = np.argmax(tmp)
        second_value = np.max(tmp)
        for j, val in enumerate(avg_row):
            cell = f"{val:.3f}"
            if j == best_idx:
                cell = f"\\cellcolor{{cyan!24}} {cell}"
            elif j == second_idx:
                cell = f"\\cellcolor{{orange!24}} {cell}"
            avg_vals.append(cell)
        
        if report_rel_gain:
            avg_abs_gain = 100 * (best_value - second_value)
            avg_rel_gain = 100 * (best_value - second_value) / (1.0 - second_value)
            avg_vals.append(f"{avg_abs_gain:.3f}")
            avg_vals.append(f"{avg_rel_gain:.1f}")
            
        print("\\midrule")
        print("Average & " + " & ".join(avg_vals) + " \\\\")

        # std row
        std_vals = [f"{val:.3f}" for val in std_row]
        if report_rel_gain:
            std_vals.append("---")  # std 行不计算增益
            std_vals.append("---")  # std 行不计算增益
        print("Std & " + " & ".join(std_vals) + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{adjustbox}")
        print("\\end{table}")
        print()
    
    our_aucs_mean = np.mean(np.array(our_aucs), axis=0)
    print(" & ".join(datasets))
    print(" & ".join([f"{val:.4f}" for val in our_aucs_mean]))

            
def report_diverse_results_old(args):
    datasets = {
        'AcademicResearch':'AcademicResearch', 
        'EducationMaterial':'EducationMaterial',
        'FoodCusine':'FoodCusine',
        'MedicalText':'MedicalText',
        'ProductReview':'ProductReview',
        'TravelTourism':'TravelTourism',
        'ArtCulture':'ArtCulture',
        'Entertainment':'Entertainment',
        'GovernmentPublic':'GovernmentPublic',
        'NewsArticle':'NewsArticle',
        'Religious':'Religious',
        'Business':'Business',
        'Environmental':'Environmental',
        'LegalDocument':'LegalDocument',
        'OnlineContent':'OnlineContent',
        'Sports':'Sports',
        'Code':'Code',
        'Finance':'Finance',
        'LiteratureCreativeWriting':'LiteratureCreativeWriting',
        'PersonalCommunication':'PersonalCommunication',
        'TechnicalWriting':'TechnicalWriting',
    }
    source_models = {
        'Llama-3-70B': 'Llama-3-70B',
        'GPT-3-Turbo': 'GPT-3-Turbo',
        'Gemini-1.5-Pro': 'Gemini-1.5-Pro',
        'GPT-4o': 'GPT-4o',
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        # 'lrr': 'LRR',
        # 'npr': 'NPR', 
        # 'dna_gpt': 'DNAGPT',
        'binoculars': 'Binoculars',
        'ide_mle': 'IntrinsicDim',
        # 'ide_twonn': 'IntrinsicDim(NN)',
    }
    methods2 = {
        # 'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'revised_gpt2': 'RevisedGPT',
        # 'rewrite_gpt': 'RewriteGPT(Likelihood)',
        # 'revised_gpt': 'RewriteGPT(Dist)',
        # 'adarewritegpt': 'RewriteGPT(Ada1)',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        # 'adarewritegpt_mean_gap': 'RewriteGPT(Ada2)',
        'l2d': 'RewriteGPT',
        # 'fluoroscopy': 'TextFluoroscopy',
        # 'biscope': 'BiScope',
        # 'classification.bspline': 'AdaDetectGPT',
        # 'superadadetectgpt': 'SuperAdaDetectGPT',
    }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def report_temperature_results(args):
    datasets = ["xsum", "squad", "writing"]
    temperatures = ["0.01", "0.2", "0.4", "0.6", "0.8", "1.0"]
    # methods = ["raidar", "imbd", "l2d"]
    methods = ["l2d"]

    records = []
    # loop over datasets, temperatures, methods
    for dataset in datasets:
        for temp in temperatures:
            row = {"Dataset": dataset, "Temperature": float(temp)}
            for method in methods:
                filename = f"{dataset}_claude-3-5-haiku_polish_{temp}.{method}.json"
                filepath = os.path.join(args.result_path, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                        row[method] = data.get("metrics", {}).get("roc_auc", None)
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
                        row[method] = None
                else:
                    row[method] = None
            records.append(row)

    # convert to dataframe
    df = pd.DataFrame(records)
    df = df.sort_values(by=["Dataset", "Temperature"]).reset_index(drop=True)
    print(df)


def get_typeIerror(result_file, alpha=0.1):
    critical_value = norm.ppf(1 - alpha, loc=0.0, scale=1.0)
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        return np.mean(real_stats > critical_value)

def get_power(result_file, alpha=0.1):
    critical_value = norm.ppf(1 - alpha, loc=0.0, scale=1.0)
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        fake_stats = np.array(res['predictions']['samples'])
        return np.mean(fake_stats > critical_value)  

def get_tpr(result_file, alpha=0.1):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        fake_stats = np.array(res['predictions']['samples'])
        critical_value = np.quantile(fake_stats, q=alpha)
        return np.mean(real_stats <= critical_value)

def get_fpr(result_file, alpha=0.1):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        fake_stats = np.array(res['predictions']['samples'])
        critical_value = np.quantile(real_stats, q=1-alpha)
        return np.mean(fake_stats <= critical_value)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_name', type=str, default="black_prompt_results")
    parser.add_argument('--result_path', type=str, default="./exp_prompt/results/")
    # parser.add_argument('--result_path', type=str, default="./exp_diverse/results/")
    # parser.add_argument('--report_name', type=str, default="diverse_results")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/paraphrasing")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/decoherence")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/random")
    # parser.add_argument('--report_name', type=str, default="temperature_results")
    # parser.add_argument('--result_path', type=str, default="./exp_topk_prompt/results")
    # parser.add_argument('--report_name', type=str, default="gpt3_results")
    # parser.add_argument('--result_path', type=str, default="./exp_theory/results_exact/")
    # parser.add_argument('--result_path', type=str, default="./exp_sup/results/")
    # parser.add_argument('--report_name', type=str, default="theory_results")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/")
    # parser.add_argument('--report_name', type=str, default="attack_results")
    # parser.add_argument('--result_path', type=str, default="./exp_topk/results")
    # parser.add_argument('--report_name', type=str, default="variance_results")
    # parser.add_argument('--alpha', type=float, default=0.10)
    # parser.add_argument('--attack_prop', type=float, default=0.05)
    # parser.add_argument('--attack_prop', type=int, default=150)
    args = parser.parse_args()

    if args.report_name == 'black_prompt_results':
        report_black_prompt_results(args)
        # report_black_prompt_results_simplified(args)
        # report_black_prompt_results_condensed(args)
    if args.report_name == 'diverse_results':
        report_diverse_results(args)