import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
import custom_datasets
from model import load_tokenizer, load_model

PROMPT1 = "You are a rewriting expert and you would rewrite the text without missing the original details. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary. \n\n Original text: \"{}\" Here is the rewritten version: \n\n"

PROMPT2 = "Revise the following text: \"{}\""

class PrefixSampler:
    def __init__(self, args, rewrite_prompt='l2d'):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.rewrite_model, args.cache_dir)
        self.base_model = load_model(args.rewrite_model, args.device, args.cache_dir)
        if rewrite_prompt == 'l2d':
            self.prompt = PROMPT1
        elif rewrite_prompt == 'bartscore':
            self.prompt = PROMPT2

        # self.pipe = pipeline("text-generation", model=self.base_model, tokenizer=self.base_tokenizer, device=torch.cuda.current_device())

    def _sample_rewrite_text_from_model(self, texts):
        texts_num_tokens = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False)['input_ids'].shape[1]
        prompt_texts = [self.prompt.format(o) for o in texts] 

        self.base_model.eval()
        decoded = ['' for _ in range(len(texts))]

        sampling_kwargs = {'temperature': self.args.temperature}
        if self.args.do_top_p:
            sampling_kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            sampling_kwargs['top_k'] = self.args.top_k

        sampling_kwargs['min_new_tokens'] = int(0.5*texts_num_tokens)
        sampling_kwargs['max_new_tokens'] = int(1.5*texts_num_tokens)
        sampling_kwargs["eos_token_id"] = self.base_tokenizer.eos_token_id
        sampling_kwargs['pad_token_id'] = self.base_tokenizer.eos_token_id

        all_encoded = self.base_tokenizer(prompt_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        prompt_lens = all_encoded['input_ids'].shape[1]
        outputs = self.base_model.generate(**all_encoded, do_sample=True, **sampling_kwargs)
        gen_ids = outputs[:, prompt_lens:]
        decoded = self.base_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return decoded

    def generate_samples(self, raw_data, batch_size):
        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],
            "sampled": [],
        }

        assert len(raw_data) % batch_size == 0
        for batch in range(len(raw_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = self._sample_rewrite_text_from_model(original_text)

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(custom_datasets.SEPARATOR, ' ')

                # add to the data
                data["original"].append(o)
                data["sampled"].append(s)

        return data
    
def get_regen_samples(sampler, text):
    data = [text] * sampler.args.regen_number
    data = sampler.generate_samples(
        data, batch_size=min(sampler.args.batch_size, sampler.args.regen_number)
    )
    return data['sampled']