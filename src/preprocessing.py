# -*- coding: utf-8 -*-
# +
import pandas as pd
from sklearn.model_selection import train_test_split

from data_collator import Preparer


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


with open('../raw_data/whitman/input.txt', 'rb') as file:
    raw_text = file.read().decode(errors='replace').replace('�', '').replace('\ufeff', '').split('\n\n\n')

raw_dataset = []
for poem in raw_text:
    poem_by_verse = poem.split('\n\n')
    poem_by_rows = [row for row in poem_by_verse if '\n' in row]
    if not poem_by_rows: continue
    raw_dataset.append(poem_by_rows)

raw_dataset







data = Preparer(max_len_token=20)
data.preprocessing([
    ('../raw_data/whitman/input.txt', read_whitman_poem),
])

train, valid = train_test_split(
    pd.Series(data.dataset), 
    test_size=0.20, 
    random_state=42,
    shuffle=False,
)

train.to_csv('../data/train.csv', index=False)
valid.to_csv('../data/valid.csv', index=False)


