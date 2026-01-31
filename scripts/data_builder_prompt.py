from transformers import pipeline
import torch
import argparse
import numpy as np
import random
import json
import tqdm
import time
import re
import custom_datasets
from model import load_tokenizer, load_model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# torch._dynamo.config.disable = True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_data(args, dataset, key):
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())

    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)
    else:
        data = custom_datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the base model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.shuffle(data)
    data = data[:5_000]

    return data

def openai_sampler(original_texts, task, args):
    from openai import OpenAI
    client = OpenAI()
    n_samples = len(original_texts)
    # n_samples = 2

    # kwargs = {"max_tokens": 500}
    kwargs = {"model": 'gpt-4o'}
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    elif args.do_top_k:
        kwargs['top_k'] = args.top_k
    elif args.do_temperature:
        kwargs['temperature'] = args.temperature

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text.'
        user_prompts = ['Please rewrite:'] * n_samples
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. '
        with open("./data/polish_prompt.json","r") as p:
            user_prompts = json.load(p)['out_prompt']
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. '
        with open("./data/expand_prompt.json","r") as p:
            user_prompts = json.load(p)['prompt']

    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        prompt = user_prompts[idx].strip()
        original_text = original_texts[idx]
        print(f"Original text: {original_text}")
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'{prompt}\n{original_text}'},
        ]
        kwargs["messages"] = messages
        response = client.chat.completions.create(**kwargs)
        output = response.choices[0].message.content
        print(f">>> OpenAI response: {output}")
        response_list.append(output)

    return response_list

def claude_sample(original_texts, task, args) -> str:
    def _clean_claude_generated_text(text: str) -> str:
        """
        移除类似 "Here's xxx:" 这种前缀提示语，只保留正文
        """
        # 匹配 "Here's ..." 或 "Here is ..." 后跟冒号的部分
        cleaned = re.sub(r"(?i)\bhere(?:'s| is)\s+[^:：]+[:：]\s*", "", text)
        return cleaned.strip()

    from anthropic import Anthropic
    client = Anthropic()
    model_full_name_list = {'claude-3-5-haiku': "claude-3-5-haiku-20241022"}
    model_full_name = model_full_name_list[args.model_name]
    n_samples = len(original_texts)

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * n_samples
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        with open("./data/polish_prompt.json","r") as p:
            user_prompts = json.load(p)['out_prompt']
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        with open("./data/expand_prompt.json","r") as p:
            user_prompts = json.load(p)['prompt']
    req = {
        "system": system_prompt,
        "temperature": args.temperature if args.do_temperature else None,
        "top_p": args.top_p if args.do_top_p else None,
        "top_k": args.top_k if args.do_top_k else None,
    }

    retries = 10
    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        original_text = original_texts[idx]
        print(f"Original text: {original_text}")
        prompt = user_prompts[idx].strip()
        for i in range(retries):
            try:
                response = client.messages.create(
                    model=model_full_name, 
                    max_tokens=1000,
                    messages=[{"role": "user", "content": f'{prompt} {original_text}'}],
                    **{k: v for k, v in req.items() if v is not None}
                )
                continue
            except Exception as e:
                wait_time = (2 ** i) + random.uniform(0, 1)
                print(f"Request failed ({e}), retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        response = response.content[0].text.strip()
        output = _clean_claude_generated_text(response)
        print(f">>> Claude response: {output}")
        response_list.append(output)

    return response_list

def gemini_sample(original_texts, task, args) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    n_samples = len(original_texts)

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * len(original_texts)
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        with open("./data/polish_prompt.json","r") as p:
            user_prompts = json.load(p)['out_prompt']
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        with open("./data/expand_prompt.json","r") as p:
            user_prompts = json.load(p)['prompt']

    max_retries = 5
    base_delay = 2
    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        prompt = user_prompts[idx].strip()
        original_text = original_texts[idx]
        print(f"Original text: {original_text}")
        params = {"model": args.model_name, "contents": f'{prompt}\n{original_text}',}
        response = None
        for i in range(max_retries):
            try:
                response = client.models.generate_content(
                    **params,
                    config=types.GenerateContentConfig(
                        top_p=args.top_p if args.do_top_p else None,
                        top_k=args.top_k if args.do_top_k else None,
                        temperature=args.temperature if args.do_temperature else None,
                        seed=args.seed,
                        candidate_count=1,
                        system_instruction=system_prompt,
                    ),
                )
                break
            except Exception as e:
                print(f"Error: {e}, retry {i+1}/{max_retries}")
                time.sleep(base_delay * (2 ** i))  # exponential backoff
        if response is None:
            raise RuntimeError(f"Failed after {max_retries} retries for sample {idx}")
        
        output = response.text.strip()
        print(f">>> Gemini response: {output}")
        response_list.append(output)

    return response_list

def generate_response_rewrite(original_texts, model, tokenizer, args):

    response =[]
    if "gemma" in args.model_name:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=torch.cuda.current_device(),
            model_kwargs={"attn_implementation": "eager"},
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=torch.cuda.current_device()
        )
    
    if "Llama" in args.model_name or "llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "min_new_tokens": (args.max_new_tokens >> 1),
        "max_new_tokens": args.max_new_tokens,
        "return_full_text": False,
        "eos_token_id": terminators,
        "do_sample": True,
    }
    if args.do_temperature:
        generation_args["temperature"] = args.temperature
    if args.do_top_k:
        generation_args["top_k"] = args.top_k
    if args.do_top_p:
        generation_args["top_p"] = args.top_p       
    
    for original_text in tqdm.tqdm(original_texts):
        if "Mistral" in args.model_name or "mistralai" in args.model_name or "Deepseek" in args.model_name: # not implementation of system prompt
            messages = [
            {"role": "user", "content": f"You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Original text:{original_text}"},
            {"role": "assistant", "content": f"Here is the rewritten paragraph: "},
        ]
        elif "gemma" in args.model_name:
            messages = [{"role": "user", "content": f"You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text.\n\nPlease rewrite: {original_text}"}]
        else:
            messages = [
                {"role": "system", "content": "You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text."},
                {"role": "user", "content": f"{original_text}"},
            ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        response.append(output)
        print(output)    
    return response

def generate_response_polish(original_texts, model, tokenizer, args):
    with open("./data/polish_prompt.json","r") as p:
        prompts = json.load(p)['out_prompt']
    response =[]
    if "gemma" in args.model_name:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=torch.cuda.current_device(),
            model_kwargs={"attn_implementation": "eager"},
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=torch.cuda.current_device()
        )
    
    if "Llama" in args.model_name or "llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "min_new_tokens": (args.max_new_tokens >> 1),
        "max_new_tokens": args.max_new_tokens,
        "return_full_text": False,
        "eos_token_id": terminators,
        "do_sample": True,
    }
    if args.do_temperature:
        generation_args["temperature"] = args.temperature
    if args.do_top_k:
        generation_args["top_k"] = args.top_k
    if args.do_top_p:
        generation_args["top_p"] = args.top_p   
    
    complete_prompts = []
    for idx, original_text in enumerate(tqdm.tqdm(original_texts)):
        prompt = prompts[idx].strip()
        if "gemma" in args.model_name:
            complete_prompt = f"{prompt}\n{original_text}.\n\nHere is the polished paragraph: "
            complete_prompts.append(complete_prompt)
            # print("Complete prompt: ", complete_prompt)
            messages = [{"role": "user", "content": complete_prompt}]
        else:
            messages = [
                {"role": "user", "content": f"{prompt}\n{original_text}"},
                {"role": "assistant", "content": f"Here is the polished paragraph: "},
            ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        response.append(output.replace("\n\n"," "))
        print(output)    
    
    # prompt_data_file = f"./exp_prompt/data/{args.dataset}_{args.model_name}_{args.task}_prompts.json"
    # with open(prompt_data_file, "w") as fout:
    #     json.dump(complete_prompts, fout, indent=4)
    #     print(f"Prompt data written into {prompt_data_file}")
    return response

def generate_response_expand(original_texts, model, tokenizer, args):
    with open("./data/expand_prompt.json","r") as p:
        prompts = json.load(p)['prompt']
    response =[]
    if "gemma" in args.model_name:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=torch.cuda.current_device(),
            model_kwargs={"attn_implementation": "eager"},
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=torch.cuda.current_device()
        )
    
    if "Llama" in args.model_name or "llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "min_new_tokens": (args.max_new_tokens >> 1),
        "max_new_tokens": args.max_new_tokens,
        "return_full_text": False,
        "eos_token_id": terminators,
        "do_sample": True,
    }
    if args.do_temperature:
        generation_args["temperature"] = args.temperature
    if args.do_top_k:
        generation_args["top_k"] = args.top_k
    if args.do_top_p:
        generation_args["top_p"] = args.top_p   
    
    for idx, original_text in enumerate(tqdm.tqdm(original_texts)):
        prompt = prompts[idx]
        if "gemma" in args.model_name:
            messages = [{"role": "user", "content": f"{prompt}\n{original_text}.\n\nHere is the expanded paragraph: "}]
        else:
            messages = [
                {"role": "user", "content": f"{prompt}\n{original_text}"},
                {"role": "assistant", "content": "Here is the expanded paragraph: "}
            ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        response.append(output)
        print(output)    
    return response

def save_data(output_file, args, data):
    # write args to file
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

def _trim_to_shorter_length(texta, textb, textc=None):
    # truncate to shorter of o and s (optional for textc)
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    if textc is not None:
        shorter_length = min(shorter_length, len(textc.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    if textc is not None:
        textc = ' '.join(textc.split(' ')[:shorter_length])
        return texta, textb, textc
    else:
        return texta, textb

def forward(args):
    print(f'Loading dataset {args.dataset}...')
    dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document'}

    original_texts = load_data(args, args.dataset, dataset_keys[args.dataset] if args.dataset in dataset_keys else None)

    if args.model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku']:
        tokenizer = load_tokenizer('gpt-neo-2.7B', cache_dir=args.cache_dir)
    else:
        tokenizer = load_tokenizer(args.model_name, cache_dir=args.cache_dir)
        model = load_model(args.model_name, device=args.device, cache_dir=args.cache_dir)

    # keep only examples with <= 512 tokens according to base_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = tokenizer(original_texts)
    original_texts = [x for x, y in zip(original_texts, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remaining data
    print(f"Total number of samples: {len(original_texts)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in original_texts])}")

    original_texts = original_texts[:min(args.n_samples, len(original_texts))]
    if args.model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku']:
        if args.model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']:
            sampled_texts = openai_sampler(original_texts, args.task, args)
        elif args.model_name == 'gemini-2.5-flash':
            sampled_texts = gemini_sample(original_texts, args.task, args)
        elif args.model_name == 'claude-3-5-haiku':
            sampled_texts = claude_sample(original_texts, args.task, args)
    else:
        if args.task == "rewrite":
            sampled_texts = generate_response_rewrite(original_texts, model, tokenizer, args)
        elif args.task == "polish":
            sampled_texts = generate_response_polish(original_texts, model, tokenizer, args)
        elif args.task == "expand":
            sampled_texts = generate_response_expand(original_texts, model, tokenizer, args)
        else:
            pass

    data = {"original": [], "sampled": [],}
    for o, s in zip(original_texts, sampled_texts):
        o, s = _trim_to_shorter_length(o, s)

        # add to the data
        data["original"].append(o)
        data["sampled"].append(s)

    save_data(args.output_file, args, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_prompt/data/xsum_gemma-9b-instruct_polish")
    parser.add_argument('--task', type=str, default="polish", choices=["rewrite", "polish", "expand", "generation"])
    parser.add_argument('--dataset', type=str, default="xsum", choices=['xsum', 'squad', 'writing', 'pubmed', 'essay'])
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--base_model_name', type=str, default="gemma-9b-instruct", choices=["mistralai-7b-instruct", "llama3-8b-instruct", "gemma-9b-instruct", "gpt-4o", "gemini-2.5-flash", "claude-3-5-haiku"])
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--n_prompts', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    args.model_name = args.base_model_name

    set_seed(args.seed)

    forward(args)
    
    