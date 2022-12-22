import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch

import random
import re
import datetime

import logging
import os
import sys

import datasets
from IPython.display import display, HTML
import random
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

nltk.download('punkt')

import wandb
wandb.login()

action_mapper = {i: chr(i+97) for i in range(10)}
rev_action_mapper = {v:k for k,v in action_mapper.items()}


import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch

import random
import re
import datetime

import logging
import os
import sys

import datasets
from IPython.display import display, HTML
import random
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

nltk.download('punkt')

import wandb
wandb.login()
action_mapper = {i: chr(i+97) for i in range(10)}
rev_action_mapper = {v:k for k,v in action_mapper.items()}

device = 'cuda'
max_input_length=512


prompt = "The goal is to chose the action which maximizes the reward."


def get_reward(chosen_action, out='out'):
    p = 0.95
    if chosen_action % 2 == 0: p = 0.05
    if out != 'out': p = 1 - p
    return np.random.choice(2, p=[1 - p, p])


def pre(text, strategy):
    if strategy == 'normal': return text
    if strategy == 'explicit': return text + 'action '
    else: return prompt + " " + text + 'action '


def post(text, chosen_action, reward, strategy):
    if strategy == 'normal': 
        return f'{text} {chosen_action} {reward}'
    else: 
        return f'{text} {chosen_action} reward {reward} action'

def generate(model, tokenizer, start, strategy='normal', out='out', leng=500):
    model.eval()
    data = []

    if out == 'in':
        leng = 100
    else:
        leng = 2500

    text = start
    text = pre(text, strategy)
    with torch.no_grad():

        while len(data) < leng:
            text = text.strip()
            encodings_dict = tokenizer(text, return_tensors='pt', padding=True).to(device)

            input_ids = encodings_dict['input_ids'][:, -max_input_length:]
            attention_mask=encodings_dict['attention_mask'][:, -max_input_length:]

            length = len(input_ids[0]) + 1

            sample_outputs = model.generate(input_ids = input_ids, attention_mask=attention_mask, max_length=length, num_beams=2, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)[0]
            chosen_action = decoded.strip()[-1]

            if chosen_action not in rev_action_mapper:
                print('Done! ->', decoded, "*", chosen_action)
                return data
            # print(text, decoded, chosen_action)
            chosen_action_int = rev_action_mapper[chosen_action]
            reward = get_reward(chosen_action_int, out=out)
            data.append([chosen_action_int, reward])
            text = post(text, chosen_action, reward, strategy)

            if len(data) % 100 == 0:
                print(f'[INFO] {len(data)} SAMPLES COMPLETED!')

    return data


texts = "Start: "
# data_sizes = [25, 100, 500, 2000]
data_sizes = [5000]

for size in data_sizes:
    # names = ['normal', 'explicit', 'prompt']
    names = ['prompt']
    # if size > 200: names = names[:1]
    for name in names:
        for out in ['out']:
            TRAIN_PARAMS = f'{name}_{size}'
            TRAIN_FILE_PATH = f"data/train_{TRAIN_PARAMS}.csv"
            VAL_FILE_PATH = f"data/test_{TRAIN_PARAMS}.csv"
            BASE_PATH = "models/"

            SAVE_PATH = BASE_PATH + f"gpt_{TRAIN_PARAMS}/"
            LOGGING_PATH = BASE_PATH + f"gpt_{TRAIN_PARAMS}/"
            SAVE_MODEL_PATH = BASE_PATH + f"gpt_{TRAIN_PARAMS}_model/"

            device = "cuda" if torch.cuda.is_available() else "cpu"

            MODEL_NAME = SAVE_MODEL_PATH

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

            max_input_length = 512
            max_target_length = 512
            tokenizer.pad_token = tokenizer.eos_token

            data = generate(model, tokenizer, texts, name, out)
            data = np.array(data)

            fname = f'{TRAIN_PARAMS}_{out}.npy'
            fpath = os.path.join('data', 'arrays', fname)
            np.save(fpath, data)

            print(f'Expected Reward for {TRAIN_PARAMS}: {np.mean(data[:, -1], axis=-1)}')

            del model
            torch.cuda.empty_cache()