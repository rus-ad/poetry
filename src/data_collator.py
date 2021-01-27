# -*- coding: utf-8 -*-
from math import ceil
import re
from tqdm import tqdm


class Preparer:
    
    def __init__(self, max_len_token):
        self.dataset = []
        self.max_len_token = max_len_token
    
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
    
    def preprocessing(self, sources: list):
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
        for filename, read_raw_dataset in sources:
            print(f'Reading file: {filename}')
            raw_text = read_raw_dataset(filename)
            for poem in tqdm(raw_text, total=len(raw_text)):
                text = self.tagged(poem)
                words = text.split(' ')
                for word in words:
                    token.append(word)
                    if len(token) < self.max_len_token:
                        continue
                        
                    if '<EOR>' in word:
                        self.dataset.append(' '.join(token))
                        token = []
