#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
python_path=/Users/j.zhu.7@bham.ac.uk/miniconda3/envs/dgpt/bin/python
exp_path=exp_topk_prompt
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# gpu_device='cuda:0'
# gpu_device='cuda:1'
gpu_device='cuda'

M="claude-3-5-haiku"
datasets="xsum squad writing"
task="polish"
M2="gemma-9b-instruct"
temperatures='0.01 0.2 0.4 0.6 0.8 1.0'

# preparing dataset
for temperature in $temperatures; do
  for D in $datasets; do
    echo date, Preparing dataset ${D}_${M}_${task}_${temperature} ...
    $python_path scripts/data_builder_prompt.py \
      --dataset $D \
      --task $task \
      --n_samples 100 \
      --base_model_name $M \
      --output_file $data_path/${D}_${M}_${task}_${temperature} \
      --do_temperature  --temperature ${temperature}
  done
done

# evaluate the rewrite-based method
for temperature in $temperatures; do
  for D in $datasets; do
    echo `date`, Evaluating Methods on ${D}_${M}_${task}_${temperature} ...
    python scripts/detect_rewrite2.py --base_model_name $M2 --dataset $D --dataset_file $data_path/${D}_${M}_${task}_${temperature} --output_file $res_path/${D}_${M}_${task}_${temperature} --device $gpu_device
  done
done

trained_model_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
target_temperatures1='0.01 0.4 1.0'
target_temperatures2='0.2 0.6 0.8'

# evaluate the ada-rewrite-based
run_experiment_ada_rewrite () {
  train_temps=$1
  eval_temps=$2
  for D in $datasets; do
    train_dataset=""
    for D1 in $datasets; do
      if [ "$D1" = "$D" ]; then
        continue  # 排除与测试集相同的 dataset
      fi

      for t2 in $train_temps; do
        if [ -z "$train_dataset" ]; then
            train_dataset="${data_path}/${D1}_${M}_${task}_${t2}"
        else
            train_dataset="${train_dataset}&${data_path}/${D1}_${M}_${task}_${t2}"
        fi
      done
    done
    echo "Train data: $train_dataset"
    python scripts/detect_rewrite_ada.py --datanum 500 --base_model "$M2" --train_dataset "$train_dataset" --save_trained

    for t1 in $eval_temps; do
      python scripts/detect_rewrite_ada.py --eval_only --base_model "$M2" --eval_dataset "$data_path/${D}_${M}_${task}_${t1}" --output_file "$res_path/${D}_${M}_${task}_${t1}" --from_pretrained "$trained_model_path"
    done
  done
}
run_experiment_ada_rewrite "$target_temperatures1" "$target_temperatures2"
run_experiment_ada_rewrite "$target_temperatures2" "$target_temperatures1"

# # evaluate ImBD
# run_experiment_ImBD () {
#   train_temps=$1
#   eval_temps=$2
#   for D in $datasets; do
#     train_dataset=""
#     for D1 in $datasets; do
#       if [ "$D1" = "$D" ]; then
#         continue  # 排除与测试集相同的 dataset
#       fi

#       for t2 in $train_temps; do
#         if [ -z "$train_dataset" ]; then
#             train_dataset="${data_path}/${D1}_${M}_${task}_${t2}"
#         else
#             train_dataset="${train_dataset}&${data_path}/${D1}_${M}_${task}_${t2}"
#         fi
#       done
#     done
#     echo "Train data: $train_dataset"
#     python scripts/detect_ImBD_task.py --datanum 500 --base_model "$M2" --train_dataset "$train_dataset" --save_trained

#     for t1 in $eval_temps; do
#       python scripts/detect_ImBD_task.py --eval_only --base_model "$M2" --eval_dataset "$data_path/${D}_${M}_${task}_${t1}" --output_file "$res_path/${D}_${M}_${task}_${t1}" --from_pretrained "$trained_model_path"
#     done
#   done
# }
# # run_experiment_ImBD "$target_temperatures1" "$target_temperatures2"
# run_experiment_ImBD "$target_temperatures2" "$target_temperatures1"

# # evaluate RAIDAR
# run_experiment_RAIDAR () {
#   train_temps=$1
#   eval_temps=$2
#   for D in $datasets; do
#     train_dataset=""
#     for D1 in $datasets; do
#       if [ "$D1" = "$D" ]; then
#         continue  # 排除与测试集相同的 dataset
#       fi

#       for t2 in $train_temps; do
#         if [ -z "$train_dataset" ]; then
#             train_dataset="${data_path}/${D1}_${M}_${task}_${t2}"
#         else
#             train_dataset="${train_dataset}&${data_path}/${D1}_${M}_${task}_${t2}"
#         fi
#       done
#     done
#     for t1 in $eval_temps; do
#       python scripts/detect_raidar.py --train_dataset ${train_dataset} --eval_dataset $data_path/${D}_${M}_${task}_${t1} --output_file $res_path/${D}_${M}_${task}_${t1}
#     done
#   done
# }
# run_experiment_RAIDAR "$target_temperatures1" "$target_temperatures2"
# run_experiment_RAIDAR "$target_temperatures2" "$target_temperatures1"