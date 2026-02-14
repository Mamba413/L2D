#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_diverse
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

source_models="Llama-3-70B GPT-3-Turbo Gemini-1.5-Pro GPT-4o"
datasets="AcademicResearch EducationMaterial FoodCusine MedicalText ProductReview TravelTourism ArtCulture Entertainment GovernmentPublic NewsArticle Religious Business Environmental LegalDocument OnlineContent Sports Code Finance LiteratureCreativeWriting PersonalCommunication TechnicalWriting"

settings='gemma-9b:gemma-9b-instruct'
scoring_models="gemma-9b-instruct"
gpu_device='cuda'

# evaluate the rewrite-based method
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating Methods on ${D}_${M} ...
    python scripts/detect_fixdistance.py --base_model_name $scoring_models --dataset $D --dataset_file $data_path/${D}_${M}  --output_file $res_path/${D}_${M}  --device $gpu_device
  done
done

# evaluate Fast-DetectGPT and Binoculars
for M in $source_models; do
  for D in $datasets; do
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_gpt_fast.py --sampling_model_name $M1 --scoring_model_name $M2 --dataset $D --dataset_file $data_path/${D}_${M}  --output_file $res_path/${D}_${M}  --discrepancy_analytic --device $gpu_device
      python scripts/detect_binoculars.py --model1_name $M1 --model2_name $M2 --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} --device $gpu_device
    done
  done
done

# evaluate PHD
for D in $datasets; do
  for M in $source_models; do
    for M2 in $scoring_models; do
      python scripts/detect_ide.py --model_name $M2 --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} --solver 'MLE' --device $gpu_device
    done
  done
done

# evaluate LRR
for M in $source_models; do
  for D in $datasets; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
      python scripts/detect_lrr.py --scoring_model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} --device $gpu_device
    done
  done
done

# evaluate supervised detectors
supervised_models="roberta-large-openai-detector"
for M in $source_models; do
  for D in $datasets; do
    for SM in $supervised_models; do
      echo `date`, Evaluating ${SM} on ${D}_${M} ...
      python scripts/detect_roberta.py --model_name $SM --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} --device $gpu_device
    done
  done
done

# evaluate RADAR
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating RADAR on ${D}_${M}   ...
    python scripts/detect_radar.py --dataset $data_path/${D}_${M}  --output_file $res_path/${D}_${M} --device $gpu_device
  done
done

data_split_2="AcademicResearch EducationMaterial FoodCusine MedicalText ProductReview TravelTourism ArtCulture Entertainment GovernmentPublic NewsArticle"
data_split_1="Religious Business Environmental LegalDocument OnlineContent Sports Code Finance LiteratureCreativeWriting PersonalCommunication TechnicalWriting"
train_model_1="Llama-3-70B"
eval_models_1="GPT-3-Turbo GPT-4o Gemini-1.5-Pro"
train_model_2="GPT-3-Turbo"
eval_models_2="Llama-3-70B"

# evaluate RADIAR 
for model_setup in 1 2; do
  if [ "$model_setup" -eq 1 ]; then
    train_model="$train_model_1"
    eval_models="$eval_models_set_1"
  else
    train_model="$train_model_2"
    eval_models="$eval_models_set_2"
  fi
  for setting in 1 2; do
    if [ "$setting" -eq 1 ]; then
      train_dataset=$data_split_1
      eval_datasets=$data_split_2
    else
      train_dataset=$data_split_2
      eval_datasets=$data_split_1
    fi
    
    my_train_dataset_str=""
    for D1 in $train_dataset; do
      if [ -z "$my_train_dataset_str" ]; then
        my_train_dataset_str="${data_path}/${D1}_${train_model}"
      else
        my_train_dataset_str="${my_train_dataset_str}&${data_path}/${D1}_${train_model}"
      fi
    done
    echo $my_train_dataset_str

    for D in $eval_datasets; do
      for M in $eval_models; do
        python scripts/detect_raidar.py --train_dataset ${my_train_dataset_str} --eval_dataset $data_path/${D}_${M}  --output_file $res_path/${D}_${M}
      done
    done
  done

# evaluate the ada-rewrite-based method
trained_model_path=scripts/AdaDist/ckpt
for setting in 1 2; do
  if [ "$setting" -eq 1 ]; then
    train_dataset=$data_split_1
    eval_datasets=$data_split_2
  else
    train_dataset=$data_split_2
    eval_datasets=$data_split_1
  fi

  my_train_dataset_str=""
  for D1 in $train_dataset; do
    if [ "$D1" = "Code" ] || [ "$D1" = "LiteratureCreativeWriting" ]; then
      echo "Skipping dataset: $D1"
      continue
    fi
  
    if [ -z "$my_train_dataset_str" ]; then
      my_train_dataset_str="${data_path}/${D1}_${train_model}"
    else
      my_train_dataset_str="${my_train_dataset_str}&${data_path}/${D1}_${train_model}"
    fi
  done

  echo "Train data: $my_train_dataset_str"
  python scripts/detect_l2d.py --datanum 500 --base_model "$scoring_models" --train_dataset "$my_train_dataset_str" --save_trained

  for D in $eval_datasets; do
    for M in $eval_models; do
      python scripts/detect_l2d.py --eval_only --base_model "$scoring_models" --eval_dataset "$data_path/${D}_${M}" --output_file "$res_path/${D}_${M}" --from_pretrained "$trained_model_path"
    done
  done
done

# evaluate ImBD
trained_model_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
for setting in 1 2; do
  if [ "$setting" -eq 1 ]; then
    train_dataset=$data_split_1
    eval_datasets=$data_split_2
  else
    train_dataset=$data_split_2
    eval_datasets=$data_split_1
  fi

  my_train_dataset_str=""
  for D1 in $train_dataset; do
    if [ "$D1" = "Code" ] || [ "$D1" = "LiteratureCreativeWriting" ]; then
      echo "Skipping dataset: $D1"
      continue
    fi
  
    if [ -z "$my_train_dataset_str" ]; then
      my_train_dataset_str="${data_path}/${D1}_${train_model}"
    else
      my_train_dataset_str="${my_train_dataset_str}&${data_path}/${D1}_${train_model}"
    fi
  done

  echo "Train data: $my_train_dataset_str"
  python scripts/detect_ImBD.py --datanum 500 --base_model "$scoring_models" --train_dataset "$my_train_dataset_str" --save_trained

  for D in $eval_datasets; do
    for M in $eval_models; do
      python scripts/detect_ImBD.py --eval_only --base_model "$scoring_models" --eval_dataset "$data_path/${D}_${M}" --output_file "$res_path/${D}_${M}" --from_pretrained "$trained_model_path"
    done
  done
done

# evaluate AdaDetectGPT
for setting in 1 2; do
  if [ "$setting" -eq 1 ]; then
    train_dataset=$data_split_1
    eval_datasets=$data_split_2
  else
    train_dataset=$data_split_2
    eval_datasets=$data_split_1
  fi

  my_train_dataset_str=""
  for D1 in $train_dataset; do
    if [ "$D1" = "Code" ] || [ "$D1" = "LiteratureCreativeWriting" ]; then
      echo "Skipping dataset: $D1"
      continue
    fi
  
    if [ -z "$my_train_dataset_str" ]; then
      my_train_dataset_str="${data_path}/${D1}_${train_model}"
    else
      my_train_dataset_str="${my_train_dataset_str}&${data_path}/${D1}_${train_model}"
    fi
  done

  echo "Train data: $my_train_dataset_str"
  for D in $eval_datasets; do
    for M in $eval_models; do
      python scripts/detect_gpt_ada.py --sampling_model_name "$scoring_models" --scoring_model_name "$scoring_models" --train_dataset ${my_train_dataset_str} --num_subsample 500 --subsample_opt 'random' --dataset_file "$data_path/${D}_${M}" --output_file "$res_path/${D}_${M}"
    done
  done
done