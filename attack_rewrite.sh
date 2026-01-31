#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_attack
mkdir -p $exp_path $exp_path/data/ $exp_path/results/
src_path=exp_prompt
src_data_path=$src_path/data
datasets="xsum squad writing"
source_models="claude-3-5-haiku"
M1='gemma-9b'
M2='gemma-9b-instruct'
paras="t5 random"  # "t5" for paraphrasing attack, or "random" for decoherence attack

for para in $paras; do
    data_path=$exp_path/data/$para
    res_path=$exp_path/results/$para
    mkdir -p $data_path $res_path

    # preparing dataset
    for D in $datasets; do
        for M in $source_models; do
            echo `date`, Preparing dataset ${D}_${M} by paraphrasing  ${src_data_path}/${D}_${M} ...
            python scripts/paraphrasing.py --dataset $D --dataset_file $src_data_path/${D}_${M}_polish --paraphraser $para --output_file $data_path/${D}_${M}_polish
        done
    done

    # evaluate RAIDAR
    for M in $source_models; do
        for D in $datasets; do
            train_dataset=""
            for D1 in $datasets; do
                if [ "$D1" = "$D" ]; then
                    continue  # 排除与测试集相同的 dataset
                fi

                if [ -z "$train_dataset" ]; then
                    train_dataset="${data_path}/${D1}_${M}_polish&${data_path}/${D1}_${M}_rewrite&${data_path}/${D1}_${M}_expand"
                else
                    train_dataset="${train_dataset}&${data_path}/${D1}_${M}_polish&${data_path}/${D1}_${M}_rewrite&${data_path}/${D1}_${M}_expand"
                fi
            done
            python scripts/detect_raidar.py --train_dataset ${train_dataset} --eval_dataset $data_path/${D}_${M}_polish --output_file $res_path/${D}_${M}_polish
        done
    done

    # evaluate the ada-rewrite-based method
    trained_model_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
    for M in $source_models; do
        for D in $datasets; do
            train_dataset=""
            for D1 in $datasets; do
                if [ "$D1" = "$D" ]; then
                    continue  # 排除与测试集相同的 dataset
                fi

                if [ -z "$train_dataset" ]; then
                    train_dataset="${data_path}/${D1}_${M}_polish&${data_path}/${D1}_${M}_rewrite&${data_path}/${D1}_${M}_expand"
                else
                    train_dataset="${train_dataset}&${data_path}/${D1}_${M}_polish&${data_path}/${D1}_${M}_rewrite&${data_path}/${D1}_${M}_expand"
                fi
            done
            echo "Train data: $train_dataset"
            python scripts/detect_rewrite_ada.py --datanum 500 --base_model ${M2} --train_dataset ${train_dataset} --eval_after_train --eval_dataset $data_path/${D}_${M}_polish --output_file $res_path/${D}_${M}_polish 

            python scripts/detect_ImBD.py --datanum 500 --base_model ${M2} --train_dataset ${train_dataset} --eval_after_train --eval_dataset $data_path/${D}_${M}_polish --output_file $res_path/${D}_${M}_polish 
        done
    done
done



