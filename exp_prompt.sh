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
exp_path=exp_prompt
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

gpu_device='cuda'

source_models="gpt-4o gemini-2.5-flash claude-3-5-haiku"
datasets="xsum squad writing"
tasks="expand rewrite polish"
# S='gemma-9b:gemma-9b-instruct'
S='gemma-9b-instruct:gemma-9b'
scoring_models="gemma-9b-instruct"

# preparing dataset
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      echo date, Preparing dataset ${D}_${M}_${task} ...
      $python_path scripts/data_builder_prompt.py \
        --dataset $D \
        --task $task \
        --n_samples 100 \
        --base_model $M \
        --output_file $data_path/${D}_${M}_${task} \
        --do_temperature  --temperature 0.8
    done
  done
done

# evaluate the rewrite-based method
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      for M1 in $scoring_models; do
        echo `date`, Evaluating Methods on ${D}_${M} ...
        python scripts/detect_fixdistance.py --base_model $M1 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
        python scripts/detect_bartscore.py --rewrite_model $M1 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
      done
    done
  done
done

# evaluate FastDetectGPT and Binoculars
IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
echo `date`, Evaluating Fast-DetectGPT with sampling model ${M1} and scoring model ${M2} ...
for M in $source_models; do
  for task in $tasks; do
    for D in $datasets; do
      python scripts/detect_gpt_fast.py --sampling_model_name $M1 --scoring_model_name $M2 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --discrepancy_analytic --device $gpu_device
      python scripts/detect_binoculars.py --model1_name $M1 --model2_name $M2 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
    done
  done
done

# evaluate fast baselines
for M in $source_models; do
  for task in $tasks; do
    for D in $datasets; do
      for M2 in $scoring_models; do
        echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
        python scripts/detect_lrr.py --scoring_model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
        python scripts/detect_ide.py --model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --solver 'MLE' --device $gpu_device
      done
    done
  done
done

# evaluate RADAR
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      echo `date`, Evaluating RADAR on ${D}_${M}_${task}  ...
      python scripts/detect_radar.py --dataset $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task}
    done
  done
done

# evaluate RoBerta
supervised_models="roberta-large-openai-detector"
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      for SM in $supervised_models; do
        echo `date`, Evaluating ${SM} on ${D}_${M}_${task} ...
        python scripts/detect_roberta.py --model_name $SM --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
      done
    done
  done
done

# evaluate AdaDetectGPT (cross-LLM training)
IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
echo `date`, Evaluating AdaDetectGPT with sampling model ${M1} and scoring model ${M2} ...
for M_test in $source_models; do
  for D in $datasets; do
    train_dataset=""

    for M_train in $source_models; do
      if [ "$M_train" = "$M_test" ]; then
        continue  # exclude LLMs that are identical to the LLM to be detected
      fi

      for D1 in $datasets; do
        if [ "$D1" = "$D" ]; then
          continue  # exclude datasets that are identical to the dataset to be detected
        fi

        for T1 in $tasks; do
          if [ -z "$train_dataset" ]; then
            train_dataset="${data_path}/${D1}_${M_train}_${T1}"
          else
            train_dataset="${train_dataset}&${data_path}/${D1}_${M_train}_${T1}"
          fi
        done
      done
    done

    for task in $tasks; do
      python scripts/detect_gpt_ada.py \
        --dataset $D --num_subsample 500 \
        --sampling_model_name ${M1} --scoring_model_name $M2 \
        --train_dataset ${train_dataset} \
        --dataset_file ${data_path}/${D}_${M_test}_${task} \
        --output_file ${res_path}/${D}_${M_test}_${task}
    done
  done
done

# evaluate RADIAR (cross-LLM training)
for M_test in $source_models; do
  for D in $datasets; do
    train_dataset=""

    for M_train in $source_models; do
      if [ "$M_train" = "$M_test" ]; then
        continue  # exclude LLMs that are identical to the LLM to be detected
      fi

      for D1 in $datasets; do
        if [ "$D1" = "$D" ]; then
          continue  # exclude datasets that are identical to the dataset to be detected
        fi

        for T1 in $tasks; do
          if [ -z "$train_dataset" ]; then
            train_dataset="${data_path}/${D1}_${M_train}_${T1}"
          else
            train_dataset="${train_dataset}&${data_path}/${D1}_${M_train}_${T1}"
          fi
        done
      done
    done

    for task in $tasks; do
      python scripts/detect_raidar.py \
        --train_dataset ${train_dataset} \
        --eval_dataset ${data_path}/${D}_${M_test}_${task} \
        --output_file ${res_path}/${D}_${M_test}_${task}
    done
  done
done

trained_ImBD_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
trained_AdaDist_path=./scripts/AdaDist/ckpt/
for M_test in $source_models; do
  for D in $datasets; do
    train_dataset=""

    for M_train in $source_models; do
      if [ "$M_train" = "$M_test" ]; then
        continue
      fi

      for D1 in $datasets; do
        if [ "$D1" = "$D" ]; then
          continue
        fi

        for T1 in $tasks; do
          if [ -z "$train_dataset" ]; then
            train_dataset="${data_path}/${D1}_${M_train}_${T1}"
          else
            train_dataset="${train_dataset}&${data_path}/${D1}_${M_train}_${T1}"
          fi
        done
      done
    done

    echo "Train data: $train_dataset"
    python scripts/detect_l2d.py --datanum 500 --base_model $scoring_models --train_dataset ${train_dataset} --save_trained
    python scripts/detect_ImBD_task.py --datanum 500 --base_model $scoring_models --train_dataset ${train_dataset} --save_trained

    for task in $tasks; do
      python scripts/detect_l2d.py --eval_only --base_model $scoring_models --eval_dataset ${data_path}/${D}_${M_test}_${task} --output_file ${res_path}/${D}_${M_test}_${task} --from_pretrained $trained_AdaDist_path
      python scripts/detect_ImBD_task.py --eval_only --base_model $scoring_models --eval_dataset $data_path/${D}_${M_test}_${task} --output_file $res_path/${D}_${M_test}_${task} --from_pretrained $trained_ImBD_path
    done
  done
done