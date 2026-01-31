from torch.utils.data import Dataset
import json
from utils import load_training_data, load_rewrite_data
from copy import deepcopy

class CustomDataset(Dataset):
    def __init__(self, data_json_dir):
        with open(data_json_dir, 'r') as f:
            data_json = json.load(f)
        self.data = self.process_data(data_json)

    def __len__(self):
        return len(self.data['original'])

    def __getitem__(self, index):
        original_text = self.data['original'][index]
        sampled_text = self.data['sampled'][index]

        return {
            'text': [original_text, sampled_text],
            'label': [0, 1]  # Original label is 0, Sampled label is 1
        }

    def process_data(self, data_json):
        processed_data = {
            'original': data_json['original'],
            'sampled': data_json['sampled']
        }

        return processed_data

class CustomDataset_rewrite(Dataset):
    def __init__(self, data_json_dir):
        self.data_json_dir = data_json_dir
        if "&" in data_json_dir:
            data_name_list = data_json_dir.split('&')
            data_json = load_training_data(data_name_list)
        else:
            with open(data_json_dir+'.raw_data.json', 'r') as f:
                data_json = json.load(f)
        self.data = self.process_data(data_json)

    def __len__(self):
        return len(self.data['original'])

    def __getitem__(self, index):
        original_text = self.data['original'][index]
        sampled_text = self.data['sampled'][index]

        return original_text, sampled_text

    def process_data(self, data_json):
        if "pubmed" in self.data_json_dir:
            processed_data = {
                'original': [qa.split("Answer:")[1].strip() for qa in data_json['original']],
                'sampled': [qa.split("Answer:")[1].strip() for qa in data_json['sampled']]
            }
        else:
            processed_data = {
                'original': data_json['original'],
                'sampled': data_json['sampled']
            }

        return processed_data

class CustomDatasetRewrite(Dataset):
    def __init__(self, data_json_dir, args):
        self.data_json_dir = data_json_dir
        if "&" in data_json_dir:
            data_name_list = data_json_dir.split('&')
            data_json = load_training_data(data_name_list)
            rewrite_data_list = [x.replace("/data/", "/results/") + f".rewrite_{args.regen_number}" for x in data_name_list]
            rewrite_data_json = load_rewrite_data(rewrite_data_list)
        else:
            with open(data_json_dir+'.raw_data.json', 'r') as f:
                data_json = json.load(f)
            rewrite_data_dir = deepcopy(data_json_dir)
            rewrite_data_dir = rewrite_data_dir.replace("/data/", "/results/") + f".rewrite_{args.regen_number}.json"
            with open(rewrite_data_dir, 'r') as f:
                rewrite_data_json = json.load(f)
                print(f"Raw rewritten eval data loaded from {rewrite_data_dir}")
            rewrite_data_json = {
                'rewrite_original': [x['rewrite_original'] for x in rewrite_data_json], 
                'rewrite_sampled': [x['rewrite_sampled'] for x in rewrite_data_json]
            }
        self.data = self.process_data(data_json, rewrite_data_json)

    def __len__(self):
        return len(self.data['original'])

    def __getitem__(self, index):
        original_text = self.data['original'][index]
        sampled_text = self.data['sampled'][index]
        rewrite_original_text = self.data['original_rewrite'][index]
        rewrite_sampled_text = self.data['sampled_rewrite'][index]

        return original_text, sampled_text, rewrite_original_text, rewrite_sampled_text

    def process_data(self, data_json, rewrite_data_json):
        if "pubmed" in self.data_json_dir:
            processed_data = {
                'original': [qa.split("Answer:")[1].strip() for qa in data_json['original']],
                'sampled': [qa.split("Answer:")[1].strip() for qa in data_json['sampled']],
            }
        else:
            processed_data = {
                'original': data_json['original'],
                'sampled': data_json['sampled'], 
                'original_rewrite': rewrite_data_json['rewrite_original'],
                'sampled_rewrite': rewrite_data_json['rewrite_sampled']
            }

        return processed_data

class CustomDataset_split(Dataset):
    def __init__(self, data_json_dir, split='train', val_ratio=0.2):
        with open(data_json_dir, 'r') as f:
            data_json = json.load(f)
        self.data = self.process_data(data_json)
        
        total_size = len(self.data['original'])
        
        if val_ratio == 0: 
            self.indices = [i for i in range(total_size)]
            return
            
        # Compute step size for stratified sampling
        step_size = int(1 / val_ratio)

        val_indices = list(range(0, total_size, step_size))
        train_indices = [i for i in range(total_size) if i not in val_indices]
        # print(val_indices)
        # print(train_indices)
        if split == 'train':
            self.indices = train_indices
        elif split == 'val':
            self.indices = val_indices
        else:
            raise ValueError("split must be either 'train' or 'val'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        original_text = self.data['original'][actual_index]
        sampled_text = self.data['sampled'][actual_index]
        return original_text, sampled_text

    def process_data(self, data_json):
        processed_data = {
            'original': data_json['original'],
            'sampled': data_json['sampled']
        }

        return processed_data
