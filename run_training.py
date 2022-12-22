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


data_sizes = [5000]
ep_sizes = {25: 150, 100: 50, 500: 15, 5000:15}

for size in data_sizes:
    names = ['normal', 'explicit', 'prompt']
    # if size > 200: names = names[:1]
    if size > 3000: names = names[-1:]
    for name in names:
        TRAIN_PARAMS = f'{name}_{size}'
        TRAIN_FILE_PATH = f"data/train_{TRAIN_PARAMS}.csv"
        VAL_FILE_PATH = f"data/test_{TRAIN_PARAMS}.csv"
        BASE_PATH = "models/"

        SAVE_PATH = BASE_PATH + f"gpt_{TRAIN_PARAMS}/"
        LOGGING_PATH = BASE_PATH + f"gpt_{TRAIN_PARAMS}/"
        SAVE_MODEL_PATH = BASE_PATH + f"gpt_{TRAIN_PARAMS}_model/"
        TRAIN_FILE_PATH, SAVE_MODEL_PATH
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL_NAME = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
        max_input_length = 512
        max_target_length = 512
        tokenizer.pad_token = tokenizer.eos_token
        input_col = "history"
        train_qns = pd.read_csv(TRAIN_FILE_PATH).loc[:, input_col]
        val_qns = pd.read_csv(VAL_FILE_PATH).loc[:, input_col]
        class HistoryDataset(Dataset):
            def __init__(self, txt_list, tokenizer, max_length):
                self.input_ids = []
                self.attn_masks = []
                self.labels = []
                self.action_tokens = [257, 275, 269, 288, 304, 277, 308, 289, 1312, 474]
                for txt in txt_list:
                    encodings_dict = tokenizer(txt, padding="max_length", max_length=max_input_length, truncation=True, return_tensors='pt')
                    self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                    self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                    input_ids = encodings_dict['input_ids']
                    labels = [[l if l in self.action_tokens else -100 for l in input_id] for input_id in input_ids]
                    self.labels.append(torch.tensor(labels))

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

        train_dataset = HistoryDataset(train_qns, tokenizer, max_length=max_input_length)
        val_dataset = HistoryDataset(val_qns, tokenizer, max_length=max_input_length)
        epochs = ep_sizes[size]
        lr = 8e-5
        batch_size = 16
        training_args = TrainingArguments(
            output_dir=SAVE_PATH,
            learning_rate=lr,
            do_train = True,
            do_eval = True,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            # load_best_model_at_end=True,
            num_train_epochs=epochs,
            # fp16=True,
            logging_dir=LOGGING_PATH,
            logging_steps=50,
            save_steps=100,
            report_to = "wandb",
            # hub_model_id='rg089/gpt2_mwp'
            )
        wandb_run = wandb.init(
            project="rl_project_ad",
            config={
                "per_device_train_batch_size": batch_size,
                "learning_rate": lr})

        now = datetime.datetime.now()
        current_time = now.strftime("%d.%b.%Y-%-I:%M:%S%p")

        run_name = SAVE_PATH.rstrip("/").split("/")[-1] + "-" + current_time 
        wandb_run.name = run_name
        print(run_name)
        trainer = Trainer(
                    model=model,  
                    args=training_args, 
                    train_dataset=train_dataset, 
                    eval_dataset=val_dataset, 
                    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                    'attention_mask': torch.stack([f[1] for f in data]),
                                                                    'labels': torch.stack([f[2] for f in data])})

        print(trainer.evaluate())
        trainer.train()
        import math
        eval_results = trainer.evaluate()
        print(eval_results)
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        eval_results
        trainer.save_model(SAVE_MODEL_PATH)
        wandb_run.finish()
        del model, tokenizer, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()


print("************************************** Training Complete ******************************************************")