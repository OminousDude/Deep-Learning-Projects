import util
import data
import attention
from constant import Constants
import embedding
import model
import train

import torch
from transformers import BertTokenizerFast, AutoTokenizer

tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

LM = torch.load('saves/BERT_time: 11|04|2024 21:50:17|step: 4000.pt')

vocab = dict((v,k) for k,v in tokenizer.get_vocab().items())

trainer = train.BertTrainer(LM, vocab=vocab, device=Constants.DEVICE.value)

text = "I [MASK] my horse!"

trainer.test(text, tokenizer, tokenizer[0], Constants.MAX_SEQ_LEN.value)