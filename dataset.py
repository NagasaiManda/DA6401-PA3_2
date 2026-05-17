import torch
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
import spacy
import os

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.dataset = load_dataset('bentrevett/multi30k', split=split)
        
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download de_core_news_sm")
            os.system("python -m spacy download en_core_web_sm")
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        from collections import Counter
        from torchtext.vocab import Vocab
        
        counter_de = Counter()
        for example in self.dataset:
            counter_de.update(self.tokenize_de(example['de']))
            
        counter_en = Counter()
        for example in self.dataset:
            counter_en.update(self.tokenize_en(example['en']))
            
        specials = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.vocab_de = Vocab(counter_de, specials=specials)
        self.vocab_en = Vocab(counter_en, specials=specials)
        return self.vocab_de, self.vocab_en

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        if not hasattr(self, 'vocab_de'):
            self.build_vocab()
            
        data = []
        for example in self.dataset:
            de_tokens = self.tokenize_de(example['de'])
            en_tokens = self.tokenize_en(example['en'])
            
            de_indices = [self.vocab_de['<sos>']] + [self.vocab_de[token] for token in de_tokens] + [self.vocab_de['<eos>']]
            en_indices = [self.vocab_en['<sos>']] + [self.vocab_en[token] for token in en_tokens] + [self.vocab_en['<eos>']]
            
            data.append((torch.tensor(de_indices), torch.tensor(en_indices)))
        return data