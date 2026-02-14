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
M_test="claude-3-5-haiku"
M_train="claude-3-5-haiku"  ## it is fair as the comparison is conducted over the ML-based method
M2='gemma-9b-instruct'
paras="vanilla random t5"  # "vanilla" for no attack, "t5" for paraphrasing attack, or "random" for decoherence attack

for para in $paras; do
    data_path=$exp_path/data/$para
    res_path=$exp_path/results/$para
    mkdir -p $data_path $res_path

    # preparing dataset
    if [ "$para" != "vanilla" ]; then
        for D in $datasets; do
            for M in $source_models; do
                echo "$(date)", Preparing ${D}_${M} using paraphraser: $para
                python scripts/paraphrasing.py \
                    --dataset $D \
                    --dataset_file ${src_data_path}/${D}_${M}_polish \
                    --paraphraser $para \
                    --output_file ${data_path}/${D}_${M}_polish
            done
        done
    else
        echo "$(date)", Copying original data to $data_path
        cp -r "${src_data_path}/." "$data_path/"
    fi

    # evaluate RAIDAR (train on other LLMs)
    for D in $datasets; do
        train_dataset=""

        # collect training data from other LLMs
        for D1 in $datasets; do
            if [ "$D1" = "$D" ]; then
                continue  # 排除与测试集相同的 dataset
            fi

            # append three tasks for each (D1, M_train)
            if [ -z "$train_dataset" ]; then
                train_dataset="${data_path}/${D1}_${M_train}_polish&${data_path}/${D1}_${M_train}_rewrite&${data_path}/${D1}_${M_train}_expand"
            else
                train_dataset="${train_dataset}&${data_path}/${D1}_${M_train}_polish&${data_path}/${D1}_${M_train}_rewrite&${data_path}/${D1}_${M_train}_expand"
            fi
        done

        echo "Train data (RAIDAR): $train_dataset"
        python scripts/detect_raidar.py \
            --train_dataset ${train_dataset} \
            --eval_dataset $data_path/${D}_${M_test}_polish \
            --output_file $res_path/${D}_${M_test}_polish \
            --regen_number 2 \
            --batch_size 2
    done

    # evaluate L2D & ImBD
    for D in $datasets; do
        train_dataset=""
        for D1 in $datasets; do
            if [ "$D1" = "$D" ]; then
                continue  # 排除测试 dataset
            fi

            if [ -z "$train_dataset" ]; then
                train_dataset="${data_path}/${D1}_${M_train}_polish&${data_path}/${D1}_${M_train}_rewrite&${data_path}/${D1}_${M_train}_expand"
            else
                train_dataset="${train_dataset}&${data_path}/${D1}_${M_train}_polish&${data_path}/${D1}_${M_train}_rewrite&${data_path}/${D1}_${M_train}_expand"
            fi
        done

        echo "Train data (AdaDist/ImBD): $train_dataset"

        python scripts/detect_l2d.py \
            --datanum 500 \
            --base_model ${M2} \
            --train_dataset ${train_dataset} \
            --eval_after_train \
            --eval_dataset $data_path/${D}_${M_test}_polish \
            --output_file $res_path/${D}_${M_test}_polish \
            --regen_number 2 \
            --batch_size 2

        python scripts/detect_ImBD_task.py \
            --datanum 500 \
            --base_model ${M2} \
            --train_dataset ${train_dataset} \
            --eval_after_train \
            --eval_dataset $data_path/${D}_${M_test}_polish \
            --output_file $res_path/${D}_${M_test}_polish
    done
done
