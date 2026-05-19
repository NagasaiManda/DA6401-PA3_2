import torch
from datasets import load_dataset
import os
import re

class SimpleVocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
        
    def __contains__(self, token):
        return token in self.stoi
        
    def __len__(self):
        return len(self.stoi)
        
    def get_itos(self):
        return self.itos

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.dataset = load_dataset('bentrevett/multi30k', split=split)

    def tokenize_de(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def tokenize_en(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        from collections import Counter
        
        counter_de = Counter()
        for example in self.dataset:
            counter_de.update(self.tokenize_de(example['de']))
            
        counter_en = Counter()
        for example in self.dataset:
            counter_en.update(self.tokenize_en(example['en']))
            
        specials = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        # Build DE vocab: sort by frequency descending
        sorted_tokens_de = sorted(counter_de.items(), key=lambda x: (-x[1], x[0]))
        itos_de = specials + [token for token, freq in sorted_tokens_de if token not in specials]
        stoi_de = {token: idx for idx, token in enumerate(itos_de)}
        
        # Build EN vocab: sort by frequency descending
        sorted_tokens_en = sorted(counter_en.items(), key=lambda x: (-x[1], x[0]))
        itos_en = specials + [token for token, freq in sorted_tokens_en if token not in specials]
        stoi_en = {token: idx for idx, token in enumerate(itos_en)}
        
        self.vocab_de = SimpleVocab(stoi_de, itos_de)
        self.vocab_en = SimpleVocab(stoi_en, itos_en)
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

def load_vocab(vocab_path="vocab.pt", gdrive_id="1dPR7kDXuLyQH8e3lxqnfg3D8pTKgQEcr"):
    import os
    import gdown
    import torch
    if not os.path.exists(vocab_path):
        if gdrive_id and gdrive_id != "YOUR_GDRIVE_ID_HERE":
            print(f"Downloading vocab from Google Drive ({gdrive_id})...")
            try:
                result = gdown.download(id=gdrive_id, output=vocab_path, quiet=False)
                if result is None or not os.path.exists(vocab_path):
                    raise RuntimeError("Download returned None or file does not exist")
            except Exception as e:
                print("failed")
                raise e
        else:
            raise FileNotFoundError(f"{vocab_path} not found and no valid Google Drive ID provided.")
    
    try:
        vocab_data = torch.load(vocab_path, map_location='cpu', weights_only=False)
    except TypeError:
        vocab_data = torch.load(vocab_path, map_location='cpu')
        
    if isinstance(vocab_data, dict) and 'de_stoi' in vocab_data:
        vocab_de = SimpleVocab(vocab_data['de_stoi'], vocab_data['de_itos'])
        vocab_en = SimpleVocab(vocab_data['en_stoi'], vocab_data['en_itos'])
        return vocab_de, vocab_en
    else:
        return vocab_data