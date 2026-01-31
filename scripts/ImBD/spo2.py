import torch
from torch import nn
import sys

from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import os
import torch.nn.functional as F
from .utils_spo import calculate_MMD_loss
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
    'mistralai-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'gemma-9b': 'google/gemma-2-9b',
}
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'qwen-7b', 'mistralai-7b', 'llama3-8b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b', 'pythia-12b', 'falcon-7b', 'falcon-7b-instruct', 'gemma-9b']

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


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).clamp_min(0.0001).sqrt()
    
    return discrepancy, log_likelihood.sum(dim=-1)

def get_variance_stat(logits_ref, logits_score, labels):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    discrepancy = log_likelihood.var(dim=-1, unbiased=True) - mean_ref.var(dim=-1, unbiased=True)
    
    return discrepancy, log_likelihood.sum(dim=-1)

def get_approx_variance_stat(logits_ref, logits_score, labels):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    discrepancy = log_likelihood.square().mean(dim=-1) - mean_ref.square().mean(dim=-1)

    return discrepancy, log_likelihood.sum(dim=-1)

class ComputeScore(nn.Module):
    def __init__(self, scoring_model_name, reference_model_name, SPOtrained=True, SPO_beta=0.5, dataset='xsum', device='cuda', cache_dir='./models'):
        super().__init__()
        self.device = device
        self.reference_model_name = get_model_fullname(reference_model_name)
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

        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        reference_model = load_model(reference_model_name, device, cache_dir, SPOtrained)
        
        if SPOtrained: 
            self.scoring_model = get_peft_model(scoring_model, self.peft_config)
            # self.scoring_model = scoring_model
            self.reference_model = reference_model
        else: 
            self.scoring_model = scoring_model
            self.reference_model = reference_model
            
        
        self.reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)

        for p in self.reference_model.parameters():
            p.requires_grad = False
        # for p in self.scoring_model.parameters():
        #     p.requires_grad = False
        # for p in self.scoring_model.lm_head.parameters():
        #     p.requires_grad = True

        self.forward = self.forward_SPO

        total = sum(p.numel() for p in self.scoring_model.parameters())
        trainable = sum(p.numel() for p in self.scoring_model.parameters() if p.requires_grad)
        print(f"Trainable / total (parameters): {trainable}/{total}={trainable/total}")

    def set_criterion_fn(self, criterion_fn):
        if criterion_fn == "mean":
            self.criterion = 'mean'
            self.criterion_fn = get_sampling_discrepancy_analytic
        elif criterion_fn == "variance":
            self.criterion = 'variance'
            self.criterion_fn = get_variance_stat
        elif criterion_fn == "approx_variance":
            self.criterion = 'approx_variance'
            self.criterion_fn = get_approx_variance_stat
        else:
            raise ValueError(f"Unknown criterion function: {criterion_fn}")
        
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

        self.load_state_dict(torch.load(os.path.join(load_directory, "model.bin"), map_location=self.device))

    def get_SPO_input(self, tokenized=None ,text=[""], labels=[""], training_module=False):
        if training_module:
            logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
            if self.reference_model_name != self.scoring_model_name:
                tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False, add_special_tokens=True, return_attention_mask=True).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = self.reference_model(tokenized.input_ids).logits[:,:-1,:]
            crit, SPO_input  = self.criterion_fn(logits_ref, logits_score, labels)
        else:
            with torch.no_grad(): # get reference
                logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # shape: [bsz, sentence_len, dim]
                if self.reference_model_name != self.scoring_model_name:
                    tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False ,add_special_tokens=True, return_attention_mask=True).to(self.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(tokenized.input_ids).logits[:,:-1,:]
                crit, SPO_input = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, SPO_input, logits_score

    def forward_SPO(self, text):
        original_text = text[0]
        sampled_text = text[1]
        
        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        train_original_crit, train_disprefered_logprob, train_original_logits_score = self.get_SPO_input(tokenized,original_text,labels,training_module=True)
        
        tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        train_sampled_crit, train_prefered_logprob, train_sampled_logits_score = self.get_SPO_input(tokenized,sampled_text,labels,training_module=True)
        
        MMDloss = calculate_MMD_loss(train_original_crit, train_sampled_crit)
        output = dict(crit=[train_original_crit.detach(), train_original_crit, train_sampled_crit.detach(), train_sampled_crit], loss=MMDloss)
        return output


