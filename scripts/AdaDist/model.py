import torch
from torch import nn

from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def calculate_reconstruct_loss(original_crit, sample_crit):
    mmd_loss = sample_crit - original_crit
    return mmd_loss

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
    'gemma-9b-instruct': 'google/gemma-2-9b-it',
    'gemma-1b': 'google/gemma-3-1b-pt',
}
float16_models = ['gemma-9b-instruct', 'gemma-1b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
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

class AdaDist(nn.Module):
    def __init__(self, model_name, dataset='xsum', device='cuda', cache_dir='./models'):
        super().__init__()
        self.device = device
        self.model_name = get_model_fullname(model_name)

        def load_model(model_name, device, cache_dir):
            model_fullname = get_model_fullname(model_name)
            print(f'Loading model {model_fullname}...')
            model_kwargs = {}
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if 'gpt-j' in model_name:
                model_kwargs.update(dict(revision='float16'))
            model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            print('Moving model to GPU...', end='', flush=True)
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')
            return model

        # load model
        self.scoring_tokenizer = load_tokenizer(model_name, dataset, cache_dir)
        scoring_model = load_model(model_name, device, cache_dir)

        if model_name in ['gemma-1b']:
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
        self.scoring_model = get_peft_model(scoring_model, self.peft_config)

        total = sum(p.numel() for p in self.scoring_model.parameters())
        trainable = sum(p.numel() for p in self.scoring_model.parameters() if p.requires_grad)
        print(f"Trainable / total (parameters): {trainable}/{total}={trainable/total}")

    def set_criterion_fn(self, criterion_fn):
        if criterion_fn == "auc":
            self.criterion = 'auc'
            self.criterion_fn = calculate_reconstruct_loss
        else:
            raise ValueError(f"Unknown criterion function: {criterion_fn}")

    def get_logp(self, tokenized, labels, pad_id, training_module=False):
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

    def forward(self, texts, training_module=True):
        original_text = texts[0]
        sampled_text = texts[1]
        pad_id = self.scoring_tokenizer.pad_token_id

        tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        train_sampled_crit = self.get_logp(tokenized, labels, pad_id, training_module=training_module)

        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        train_original_crit = self.get_logp(tokenized, labels, pad_id, training_module=training_module)

        try:
            original_rewrite_text = [x[0] for x in texts[2]]
            tokenized = self.scoring_tokenizer(original_rewrite_text, return_tensors="pt", padding=True).to(self.device)
            labels = tokenized.input_ids[:, 1:] 
            train_original_regen_crit = self.get_logp(tokenized, labels, pad_id, training_module=training_module)

            sampled_rewrite_text = [x[0] for x in texts[3]]
            tokenized = self.scoring_tokenizer(sampled_rewrite_text, return_tensors="pt", padding=True).to(self.device)
            labels = tokenized.input_ids[:, 1:] 
            train_sampled_regen_crit = self.get_logp(tokenized, labels, pad_id, training_module=training_module)
        except torch.OutOfMemoryError:
            print("=================== long texts ===================")
            print(sampled_rewrite_text)

        if self.criterion == 'auc':
            train_original_crit_opt = torch.abs(train_original_crit - train_original_regen_crit).mean()
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

    def save_pretrained(self, save_directory: str):
        """
        Save the scoring model (with LoRA adapter) to reduce memory.
        """
        os.makedirs(save_directory, exist_ok=True)

        scoring_dir = os.path.join(save_directory, "scoring_model")
        self.scoring_model.save_pretrained(scoring_dir, safe_serialization=True)

        print(f"✅ Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, *args, **kwargs):
        """
        Load the scoring model, reference model, and all null_distr buffers.
        """
        model = cls(*args, **kwargs)

        scoring_dir = os.path.join(load_directory, "scoring_model")
        model.scoring_model = AutoPeftModelForCausalLM.from_pretrained(
            scoring_dir, 
            device_map="auto", 
            low_cpu_mem_usage=True, 
            use_safetensors=True, 
            cache_dir=kwargs['cache_dir'],
        )

        print(f"✅ Model loaded from {load_directory}")
        return model

