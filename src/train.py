# +
import os
import sys
import time
import math
import random
import re
import json

import numpy as np
import pandas as pd

from tqdm.auto import tqdm, trange
from torch.cuda.amp import autocast

import spacy
from spacy.symbols import ORTH
from spacy.lang.en import English

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim import *
from torch.nn.modules.loss import *
from torch.optim.lr_scheduler import * 

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import EvalPrediction
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel


# -
def create_optimizer_and_scheduler(
    model,
    total_steps,
    weight_decay: float = 0.0,
    learning_rate: float = 5e-5,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    warmup_steps = 0,
):
    """
    Setup the optimizer and the learning rate scheduler.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )
    num_training_steps = total_steps - warmup_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler


class PoetryDataset(Dataset):
    
    def __init__(self, filename, tokenizer):
        self.input_ids = []
        self.filename = filename
        self.tokenizer = tokenizer
        self._get_dataset(filename)
        
    def _get_dataset(self, filename: str):
        dataset = pd.read_csv(self.filename)
        dataset = dataset['0'].to_list()
        for row in dataset:
            encoding = self.tokenizer(row, add_special_tokens=True)
            self.input_ids.extend([encoding["input_ids"]])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.input_ids[i], dtype=torch.long)


class Trainer:
    
    _BATCH_SIZE = 16
    _DATALOADER_DROP_LAST = True
    _NUM_EPOCHS = 10
    _MAX_GRAD_NORM = 1.0
    _device = torch.device('cuda:2')
    
    def __init__(self, params=None, warm_start=False):
        self.base_dir = pd.Timestamp.now().strftime('../experiments/%d-%m_%H:%M_checkpoint')
        self._special_tokens_dict = {
            'bos_token': '<BOS>', 
            'eos_token': '<EOS>', 
            'pad_token': '<PAD>',
        }
        self._tokens_dict = [
            '<BOV>',
            '<BOR>',
            '<EOV>',
            '<EOR>',
        ]
        self._load_main_properties(params, warm_start)
        
    def _load_main_properties(self, params, warm_start):
        self.tokenizer = GPT2Tokenizer.from_pretrained(params['tokenizer'])
        self.model = GPT2LMHeadModel.from_pretrained(params['model'])
        if not warm_start:
            self.tokenizer.add_special_tokens(self._special_tokens_dict)
            self.tokenizer.add_tokens(self._tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.train_iterator = self.get_iterator(
            '../data/train.csv',
        )
        self.optimizer, self.lr_scheduler = create_optimizer_and_scheduler(
            self.model,
            len(self.train_iterator) * self._NUM_EPOCHS,
        )
        if warm_start:
            self.optimizer.load_state_dict(
                torch.load(
                    params['optimizer'], 
                    map_location=self._device,
                )
            )
            self.lr_scheduler.load_state_dict(torch.load(params['scheduler']))
        
    def _save_checkpoint(self, n_epoch):
        os.makedirs(self.base_dir, exist_ok=True)
        output_dir = f"{self.base_dir}/"
        if not isinstance(self.model, PreTrainedModel):
            state_dict = self.model.state_dict()
            torch.save(state_dict, f"{output_dir}pytorch_model.bin")
        else:
            self.model.save_pretrained(output_dir)
        torch.save(self.optimizer.state_dict(), f"{output_dir}optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), f"{output_dir}scheduler.pt")
        self.tokenizer.save_pretrained(output_dir)
        with open(f'{output_dir}data.txt', 'w') as outfile:
            json.dump({
                'n_epoch': n_epoch + 1,
                'batch_size': self._BATCH_SIZE,
                'total_epoch': self._NUM_EPOCHS,
                'warm_start_params': {
                    'optimizer': f"{output_dir}optimizer.pt",
                    'scheduler': f"{output_dir}scheduler.pt",
                    'model': f"{output_dir}pytorch_model.bin",
                    'tokenizer': output_dir,
                },
            }, outfile)

    def get_sampler(self, dataset, is_eval=False):
        self.model = self.model.to(self._device)
#         return SequentialSampler(dataset)
        return RandomSampler(dataset) 
        
    def get_iterator(self, filename):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = self.tokenizer, 
            mlm = False, 
            mlm_probability = 0.15
        )
        dataset = PoetryDataset(filename, self.tokenizer)
        sampler = self.get_sampler(dataset)
        return DataLoader(    
            dataset,
            sampler=sampler,
            batch_size=self._BATCH_SIZE,
            collate_fn=data_collator,
            drop_last=self._DATALOADER_DROP_LAST,
        )
    
    def _prepare_inputs(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._device)
        
    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        self.model.zero_grad()
        epochs = trange(0, int(self._NUM_EPOCHS), desc="Epoch")
        for epoch in epochs:
            epoch_iterator = tqdm(self.train_iterator, desc="Iteration")
            for step, inputs in enumerate(epoch_iterator):
                self.model.train()
                self._prepare_inputs(inputs)
                with autocast():        
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self._MAX_GRAD_NORM
                )
                scaler.step(self.optimizer)
                scaler.update()
                self.lr_scheduler.step()
                self.model.zero_grad()
                torch.cuda.empty_cache()
            print(f'Train {epoch} ==> {torch.exp(loss)}')
            self._save_checkpoint(epoch)
        self.evalute()
            
    def evalute(self):
        max_length = self.model.config.n_positions
        stride = 512
        df = pd.read_csv('../data/valid.csv')
        evalset = ' '.join(df['0'].to_list())
        encodings = self.tokenizer(evalset, return_tensors='pt')
        losses = []
        iterator = tqdm(
            range(0, encodings.input_ids.size(1), stride), 
            desc="Iteration",
        )
        for i in iterator:
            self.model.eval()
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:,begin_loc:end_loc].to(test._device)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100
            self.model.eval()
            with torch.no_grad():
                with autocast():
                    outputs = test.model(input_ids, labels=target_ids)
                    log_likelihood = outputs[0] * trg_len
            losses.append(log_likelihood)
        self.ppl = torch.exp(torch.stack(losses).sum() / end_loc)
        print(self.ppl)


test = Trainer(
    params={
        'model': 'gpt2',
        'tokenizer': 'gpt2',
    },
    warm_start=False,
)
test.train()


def predict(trainer, prompt_text, args):
    input_ids = trainer.tokenizer.encode(prompt_text, return_tensors="pt")
    input_ids = input_ids.to(trainer._device)
    output_sequences = trainer.model.generate(input_ids=input_ids, **args)
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
        
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = trainer.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        text = text[: text.index('<EOS>')]
        begin_squence = len(
            trainer.tokenizer.decode(
                input_ids[0], 
                clean_up_tokenization_spaces=True,
            )
        )
        total_sequence = (prompt_text + text[begin_squence:])
    for pattern, repl_str  in (
        ('<BOS>', '\n\n\n'),
        ('<EOS>', '\n\n\n'),
        ('<BOV>', '\n\n'),
        ('<EOV>', '\n\n'),
        ('<BOR>', '\n'),
        ('<EOR>', '\n'),
    ):
        total_sequence = total_sequence.replace(pattern, repl_str)
    print(total_sequence)


prompt_text = '<BOS><BOV><BOR>Magic'
predict(
    trainer = test,
    prompt_text = prompt_text,
    args = dict(
        max_length=1024,
        temperature=1.,
        top_k=10,
        top_p=.8,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
    ),
)











prompt_text = '<BOS><BOV><BOR>Magic'
predict(
    trainer = test,
    prompt_text = prompt_text,
    args = dict(
        max_length=500, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        num_return_sequences=5, 
        early_stopping=True,
    ),
)







df = pd.read_csv('../data/train.csv')



data



# +
with open('../experiments/21-01_20:23_checkpoint/data.txt', 'r') as file:
    params = json.load(file)['warm_start_params']

params['model'] = '../experiments/21-01_20:23_checkpoint/'
test = Trainer(
    params=params,
    warm_start=True,
)
test.train()
# -

















# +
if self.is_world_process_zero():
            

            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021






#     def get_sampler(self, dataset, is_eval=False):
#         if is_eval:
#             num_processes = torch.distributed.get_world_size()
#             process_index = torch.distributed.get_rank()
#             self.model = torch.nn.DataParallel(self.model)
#             return DistributedSampler(
#                 dataset, 
#                 num_replicas=num_processes, 
#                 rank=process_index
#             )
#         return SequentialDistributedSampler(dataset)



# -







