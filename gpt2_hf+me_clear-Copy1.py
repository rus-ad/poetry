# -*- coding: utf-8 -*-
# +
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling

from tqdm.auto import tqdm, trange

import os
import sys
import time
import math
import random
import re

import spacy
from spacy.symbols import ORTH
from spacy.lang.en import English

from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim import *
from torch.nn.modules.loss import *
from torch.optim.lr_scheduler import * 
from transformers.optimization import get_linear_schedule_with_warmup

from transformers.trainer_utils import EvalPrediction


# +
USE_GPU = 1
if USE_GPU:
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')

print('using device:', device)

# +
from packaging import version

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast
    
_use_native_amp, _use_apex
# -

if _use_native_amp:
    scaler = torch.cuda.amp.GradScaler()

BATCH_SIZE = 1
DATALOADER_DROP_LAST = True
MODEL_NAME = 'gpt2'
NUM_EPOCHS = 10
MIN_LENGTH_FOR_SHORT_ROW = 3
MAX_GRAD_NORM = 1.0

# +
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
special_tokens_dict = {
    'bos_token': '<BOS>', 
    'eos_token': '<EOS>', 
    'pad_token': '<PAD>',

}
tokens_dict = [
    '<BOV>',
    '<BOR>',
    '<EOV>',
    '<EOR>',
]

tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.add_tokens(tokens_dict)

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, 
    mlm = False, 
    mlm_probability = 0.15
)


# -

def read_whitman_poem(filename: str) -> list:
    """We bring poems to a unified form for marking with additional tokens."""
    with open(filename, 'rb') as file:
        raw_text = file.read().decode(errors='replace').replace('�', '').replace('\ufeff', '').split('\n\n\n')
        
    raw_dataset = []
    for poem in raw_text:
        poem_by_verse = poem.split('\n\n')
        poem_by_rows = [row for row in poem_by_verse if '\n' in row]
        if not poem_by_rows: continue
        raw_dataset.append(poem_by_rows)
    return raw_dataset


class Preparer:
    
    def __init__(self):
        self.dataset = []
        self.max_len_token = 20
    
    def tagged(self, poem: list) -> str:
        """Data markup implementation."""
        result = ' <BOS>'
        for idx, verse in enumerate(poem):
            bov_token = ' <BOV>' if idx else '<BOV>'
            result += bov_token
            rows = ''
            for row in verse.split('\n'):
                rows += f' <BOR>{row.lstrip().rstrip()}<EOR> '
            result += rows.rstrip().lstrip() + '<EOV>'
        result += '<EOS> '
        result = re.sub('\s+', ' ', result)
        return result.rstrip().lstrip()
    
    def get_index_end_of_string(self, text: list):
        for n, word in enumerate(text):
            if not '<EO' in word:
                continue
            return n
    
    def preprocessing(self, read_raw_dataset, filenames: list):
        """Preparing data for training.
        
        Raw text =>
        [Poem level
            [Verse level
                Rows level
                [I MET a seer,],
                [Passing the hues and objects of the world,],
                [The fields of art and learning, pleasure, sense,],
                [To glean eidólons.],
            ]
        ]
        
        """
        token = []
        for filename in filenames:
            raw_text = read_raw_dataset(filename)
            for poem in raw_text:
                text = self.tagged(poem)
                words = text.split(' ')
                for word in words:
                    token.append(word)
                    if len(token) < self.max_len_token:
                        continue
                        
                    if '<EOR>' in word:
                        print(len(token))
                        self.dataset.append(' '.join(token))
                        token = []


test = Preparer()


class PoetryDataset(Dataset):
    
    def __init__(self):
        self.input_ids = []
        
    def _get_dataset(self, file_path: str, is_eval: bool=False):
        with open(file_path, 'rb') as file:
            text = file.read().decode(errors='replace').replace('�', '').replace('\ufeff', '').split('\n\n\n')
        dataset = []
        for poem_raw in text:
            poem = poem_raw.split('\n\n')
            poem = [row for row in poem if '\n' in row]
            if not poem: continue
            result = [' <BOS> ']
            for verse in poem:
                result.extend([' <BOV> '])
                result.extend([f' <BOR> {row.lstrip().rstrip()} <EOR> ' for row in verse.split('\n')])
                result.extend([' <EOV> '])
            result.extend([' <EOS> '])
            dataset.append(''.join(result))

        encoding = tokenizer(dataset, add_special_tokens=True)
        self.input_ids.extend([list(input_id) for input_id in encoding["input_ids"]])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.input_ids[i], dtype=torch.long)


# +
train_dataset = PoetryDataset()
train_dataset._get_dataset('data/whitman/input.txt')

# for foldername in os.listdir('data/'):
#     train_dataset._get_dataset(f'data/{foldername}/input.txt')

train_sampler = RandomSampler(train_dataset)
train_iterator = DataLoader(    
    train_dataset,
    sampler=train_sampler,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    drop_last=DATALOADER_DROP_LAST,
)

# eval_dataset = PoetryDataset('data/whitman/input.txt', is_eval=True)
# eval_sampler = SequentialSampler(eval_dataset)
# eval_iterator = DataLoader(
#     eval_dataset,
#     sampler=eval_sampler,
#     batch_size=BATCH_SIZE,
#     collate_fn=data_collator,
#     drop_last=DATALOADER_DROP_LAST,
# )
# -

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.resize_token_embeddings(len(tokenizer))


def create_optimizer_and_scheduler(
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
    num_training_steps = len(train_iterator) * NUM_EPOCHS - warmup_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler


optimizer, lr_scheduler = create_optimizer_and_scheduler()

if _use_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

# +
model.zero_grad()
epochs = trange(0, int(NUM_EPOCHS), desc="Epoch")

stop_point = 0 
for epoch in epochs:
    perplexity = 0
    epoch_iterator = tqdm(train_iterator, desc="Iteration")
    for step, inputs in enumerate(epoch_iterator):
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
                
        if _use_native_amp:
            with autocast():        
                outputs = model(**inputs)
                loss = outputs[0]
        else:
            outputs = model(**inputs)
            loss = outputs[0]

        if _use_native_amp:
            scaler.scale(loss).backward()
        elif _use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if _use_native_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        elif _use_apex:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), MAX_GRAD_NORM)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
        if _use_native_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr_scheduler.step()
        model.zero_grad()
        torch.cuda.empty_cache()
        
    print(f"Train loss {loss / len(train_iterator)}")
# -





prompt_text = '<BOS><BOV><BOR>Magic'
input_ids  = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
input_ids  = input_ids.to(device)

# +
output_sequences = model.generate(
        input_ids=input_ids,
        max_length=500,
        temperature=1,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
    )

# Remove the batch dimension when returning multiple sequences
if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

# +
generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[:None]

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = (
        prompt_text + text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :]
    )

    generated_sequences.append(total_sequence)
    print(total_sequence)
# -

result = """
<BOS><BOV><BOR>Magic, the unseen world of things and lands—the real world, <EOR> <BOR> The past entire in its myriad forms. <EOR> <EOV> <BOV> <BOR> I see a vast undulating sphere beyond all time except now known as the <EOR> <BOR> Unknown Past; <EOR> <BOR> It is not Time or Space only but it also includes Reality, Space itself, <EOR> <BOR> Dimensions untold yet untellable before we are born, <EOR> <BOR> And that which was once unknown to us by any other means has become <EOR> <BOR> indetermin'd there on this face of ours. <EOR> <EOV> <BOV> <BOR> Myself I am aware myself for my own sake, <EOR> <BOR> For what purpose have you drawn me here? <EOR> <BOR> What do they want from life here then O soul? <EOR> <BOR> O who knows if Life could be nothing without Death? <EOR> <BOR> Who knowest thou art destin'd out among these States? <EOR> <BOR> Who knew so long ago whether Life can ever again exist without Death? <EOR> <BOR> (If Life were no more than an illusion, if Death could never exist without <EOR> <BOR> Death?) <EOR> <EOV> <EOS>
""".replace(
    '<BOS>', '\n\n\n').replace(
    '<EOS>', '\n\n\n').replace(
    '<BOV>', '\n\n').replace(
    '<EOV>', '\n\n').replace(
    '<BOR>', '\n').replace(
    '<EOR>', '\n')

print(result)



# Sleep
#
# I know I cannot lie awake all night and be content with myself, 
# But that the light falls on my face as now upon others' faces, 
# And from them comes their own melodious song. 
#
# O how sweet it is for me! 
# How much more so when you are alone in your room or at 
# the dinner table; 
# It seems Nature has given us a little something out of its workings, 
# and we have not yet fully comprehend'd what it brings. 

# Magic
#
# Magic, the unseen world of things and lands—the real world, 
# The past entire in its myriad forms. 
#
# I see a vast undulating sphere beyond all time except now known as the 
# Unknown Past; 
# It is not Time or Space only but it also includes Reality, Space itself, 
# Dimensions untold yet untellable before we are born, 
# And that which was once unknown to us by any other means has become 
# indetermin'd there on this face of ours. 
#
# Myself I am aware myself for my own sake, 
# For what purpose have you drawn me here? 
# What do they want from life here then O soul? 
# O who knows if Life could be nothing without Death? 
# Who knowest thou art destin'd out among these States? 
# Who knew so long ago whether Life can ever again exist without Death? 
# (If Life were no more than an illusion, if Death could never exist without 
# Death?) 





output_dir = 'clear_model/'

import os

tokenizer.save_vocabulary('clear_vocab/')

model.save_pretrained(output_dir)
torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))







# +
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("The Manhattan bridge")
context = torch.tensor([generated])
past = None

output, past = model(context, past=past)

token = torch.argmax(output[..., -1, :])
context = token.unsqueeze(0)

probs = torch.softmax(output, axis=-1)
probs = probs[0][-1]
probs = probs.cpu().detach().numpy()
# -

torch.softmax(output, axis=-1)[0][-1].argmax()

probs.shape
stoi[word]

# +
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("The Manhattan bridge")
context = torch.tensor([generated])
past = None

for i in range(5):
    print(i)
    output, past = model(context, past=past)
    token = torch.argmax(output[..., -1, :])

    generated += [token.tolist()]
    context = token.unsqueeze(0)


# -

def evaluate():
    model.eval()
    eval_losses: List[float] = []
    preds: torch.Tensor = None
    label_ids: torch.Tensor = None
        
    for inputs in tqdm(eval_iterator):
        loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
        if loss is not None:
            eval_losses.append(loss)
        if logits is not None:
            preds = logits if preds is None else torch.cat((preds, logits), dim=0)
        if labels is not None:
            label_ids = labels if label_ids is None else torch.cat((label_ids, labels), dim=0)
            
        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)


def prediction_step(
    self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Perform an evaluation step on :obj:`model` using obj:`inputs`.
    Subclass and override to inject custom behavior.
    Args:
        model (:obj:`nn.Module`):
            The model to evaluate.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        prediction_loss_only (:obj:`bool`):
            Whether or not to return the loss only.
    Return:
        Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        A tuple with the loss, logits and labels (each being optional).
    """
    has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

    inputs = self._prepare_inputs(inputs, model)

    with torch.no_grad():
        outputs = model(**inputs)
        if has_labels:
            loss, logits = outputs[:2]
            loss = loss.mean().item()
        else:
            loss = None
            logits = outputs[0]
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

    if prediction_loss_only:
        return (loss, None, None)

    labels = inputs.get("labels")
    if labels is not None:
        labels = labels.detach()
    return (loss, logits.detach(), labels)






