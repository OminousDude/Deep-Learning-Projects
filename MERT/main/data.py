import dask.dataframe as dd
from torch.utils.data import Dataset, DataLoader
from dask import delayed
from fastparquet import ParquetFile
import glob
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
import random
import pandas as pd

class BertDataset(Dataset):
    def __init__(self, file_dict, max_seq_len, vocab_size):
        self.path = file_dict

        # Read files one by one
        files = glob.glob(file_dict)

        ddf = dd.from_delayed([self.load_chunk(f) for f in files])
        self.data = ddf.compute()

        # Load tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
        bert_tokenizer = self.tokenizer.train_new_from_iterator(text_iterator=self.batch_iterator(), vocab_size=vocab_size)
        bert_tokenizer.save_pretrained("tokenizer")
        self.tokenizer = bert_tokenizer

        self.vocab = self.tokenizer.get_vocab()
        self.pad_i = self.vocab['[PAD]']
        self.mask_i = self.vocab['[MASK]']

    @delayed
    def load_chunk(self, pth) -> pd.DataFrame:
        x = ParquetFile(pth).to_pandas()
        return x

    def batch_iterator(self):
        for i in tqdm(range(0, len(self.data), self.max_seq_len)):
            yield self.data[i : i + self.max_seq_len ]["text"]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.data['text'][idx]

        # Encode the sentence
        sentence = []
        label_sentence = []
        encoding = self.tokenizer.encode(text, max_length = self.max_seq_len, return_special_tokens_mask=True, truncation=True)

        for token in encoding:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    sentence.append(self.mask_i)
                elif prob < 0.9:
                    sentence.append(random.randrange(len(self.vocab)))
                else:
                    sentence.append(token)

                label_sentence.append(token)
            else:
                sentence.append(token)
                label_sentence.append(0)

        padding = [self.pad_i for _ in range(self.max_seq_len - len(sentence))]
        sentence.extend(padding)
        label_sentence.extend(padding)

        sentence = torch.as_tensor(sentence)
        label_sentence = torch.as_tensor(label_sentence)

        return sentence, label_sentence