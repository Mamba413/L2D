import torch
from torch import nn
import sys

from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import os
import torch.nn.functional as F
from .utils_spo import calculate_reconstruct_loss1, calculate_reconstruct_loss2
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')

model_fullnames = {  
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'qwen-7b': 'Qwen/Qwen2.5-7B',
    'mistralai-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma-9b-instruct': 'google/gemma-2-9b-it',
    'gemma-1b': 'google/gemma-3-1b-pt',
    'mistralai-8b-instruct': 'mistralai/Ministral-8B-Instruct-2410',
}
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'qwen-7b', 'mistralai-7b', 'mistralai-7b-instruct', 'llama3-8b-instruct', 'gemma-9b-instruct', 'llama3-8b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b', 'pythia-12b', 'falcon-7b', 'falcon-7b-instruct', 'gemma-9b', 'mistralai-8b-instruct']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


def get_logp_statistics(logits, labels, pad_id):
    lprobs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    # gather true-token logprobs
    labels_expanded = labels.unsqueeze(-1)  # [B, T, 1]
    token_logprobs = lprobs.gather(dim=-1, index=labels_expanded).squeeze(-1)  # [B, T]
    mask = (labels != pad_id).float()  # [B, T]
    total = (token_logprobs * mask).sum(dim=1)  # [B]
    
    return total


class ComputeScore(nn.Module):
    def __init__(self, scoring_model_name, reference_model_name, SPOtrained=True, SPO_beta=0.5, dataset='xsum', device='cuda', cache_dir='./models'):
        super().__init__()
        self.device = device
        self.scoring_model_name = get_model_fullname(scoring_model_name)
        self.beta = SPO_beta
        
        def load_model(model_name, device, cache_dir, SPOtrained=True):
            model_fullname = get_model_fullname(model_name)
            print(f'Loading model {model_fullname}...')
            model_kwargs = {}
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if 'gpt-j' in model_name:
                model_kwargs.update(dict(revision='float16'))
            if SPOtrained:
                model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            else: # Load ablation finetuned model
                model = from_pretrained(AutoPeftModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            print('Moving model to GPU...', end='', flush=True)
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')
            return model
        
        # load model
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
        scoring_model = load_model(scoring_model_name, device, cache_dir, SPOtrained)
        if scoring_model_name in ['gemma-1b', 'mistralai-8b-instruct']:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=4,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        else:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1, 
            )

        if SPOtrained: 
            self.scoring_model = get_peft_model(scoring_model, self.peft_config)
        else: 
            self.scoring_model = scoring_model
            
        self.forward = self.forward_SPO

        total = sum(p.numel() for p in self.scoring_model.parameters())
        trainable = sum(p.numel() for p in self.scoring_model.parameters() if p.requires_grad)
        print(f"Trainable / total (parameters): {trainable}/{total}={trainable/total}")

    def set_criterion_fn(self, criterion_fn):
        if criterion_fn == "mean_gap":
            self.criterion = 'mean_gap'
            self.criterion_fn = calculate_reconstruct_loss1
        elif criterion_fn == "auc":
            self.criterion = 'auc'
            self.criterion_fn = calculate_reconstruct_loss2
        else:
            raise ValueError(f"Unknown criterion function: {criterion_fn}")

    def get_SPO_input(self, tokenized, labels, pad_id, training_module=False):
        """
        计算整个序列 input_ids 的 total log-prob
        """
        lengths = tokenized.attention_mask.sum(dim=1).clamp(min=1).float()  # [B]
        if training_module:
            logits = self.scoring_model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # [B, T, V]
            logp = get_logp_statistics(logits, labels, pad_id)
        else:
            with torch.no_grad():
                logits = self.scoring_model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # [B, T, V]
                logp = get_logp_statistics(logits, labels, pad_id)
        avg_logp = logp / lengths
        return avg_logp  # [B]

    def forward_SPO(self, texts):
        """
        计算一批文本的 total log-prob（sum over tokens） under the base model.
        返回 tensor shape [len(texts)]，每个是 log p(text).
        """
        original_text = texts[0]
        sampled_text = texts[1]
        pad_id = self.scoring_tokenizer.pad_token_id

        tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        train_sampled_crit = self.get_SPO_input(tokenized, labels, pad_id, training_module=True)

        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        train_original_crit = self.get_SPO_input(tokenized, labels, pad_id, training_module=True)
        
        try:
            original_rewrite_text = [x[0] for x in texts[2]]
            tokenized = self.scoring_tokenizer(original_rewrite_text, return_tensors="pt", padding=True).to(self.device)
            labels = tokenized.input_ids[:, 1:] 
            train_original_regen_crit = self.get_SPO_input(tokenized, labels, pad_id, training_module=True)

            sampled_rewrite_text = [x[0] for x in texts[3]]
            tokenized = self.scoring_tokenizer(sampled_rewrite_text, return_tensors="pt", padding=True).to(self.device)
            labels = tokenized.input_ids[:, 1:] 
            train_sampled_regen_crit = self.get_SPO_input(tokenized, labels, pad_id, training_module=True)
        except torch.OutOfMemoryError:
            print("=================== long texts ===================")
            print(sampled_rewrite_text)

        if self.criterion == 'auc':
            train_original_crit_opt = torch.abs(train_original_crit - train_original_regen_crit).mean()
            train_sampled_crit_opt = torch.abs(train_sampled_crit - train_sampled_regen_crit).mean()
        elif self.criterion == 'auc2':
            train_original_crit_opt = torch.abs(train_original_crit - train_sampled_crit).mean()
            train_sampled_crit_opt = torch.abs(train_sampled_crit - train_sampled_regen_crit).mean()
        else:
            train_original_crit_opt = train_original_crit.mean()
            train_sampled_crit_opt = train_sampled_crit.mean()
        MMDloss = self.criterion_fn(train_original_crit_opt, train_sampled_crit_opt)

        train_original_crit = torch.abs(train_original_crit - train_original_regen_crit).mean()
        train_sampled_crit = torch.abs(train_sampled_crit - train_sampled_regen_crit).mean()
        output = dict(crit=[train_original_crit.detach(), train_original_crit, train_sampled_crit.detach(), train_sampled_crit], loss=MMDloss)
        return output

    def print_gradient_requirement(self):
        for name, param in self.named_parameters():
            gradient_requirement = 'Requires Grad' if param.requires_grad else 'Does not require grad'
            color_code = '\033[92m' if param.requires_grad else '\033[91m'  # Green for requires grad, red for does not require grad
            reset_color = '\033[0m'  # Reset color after printing
            print(f"{name}: {color_code}{gradient_requirement}{reset_color}")

    def register_no_grad(self, module_names):
        for name, param in self.named_parameters():
            for selected_module in module_names:
                # print(selected_module, name)
                if selected_module in name:
                    param.requires_grad = False

    def save_pretrained(self, save_directory):
        """
        Save the model's state_dict to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))

    def from_pretrained(self, load_directory):
        """
        Load the model's state_dict from the specified directory.
        """
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")

        print("[NOTE] Load pretrained model from: ", os.path.join(load_directory, "model.bin"))

        self.load_state_dict(torch.load(os.path.join(load_directory, "model.bin"), map_location=self.device))
