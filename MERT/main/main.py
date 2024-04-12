# Import 3rd party dependencies
import dask.dataframe as dd
from dask import delayed
from fastparquet import ParquetFile
import glob
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
import random
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd

# Import my own classes
import util
import data
import attention
from constant import Constants
import embedding
import model
import train

torch.set_float32_matmul_precision('high')

# Set random seed
util.random_seed(Constants.RANDOM_SEED.value)

# Create dataset and dataloader
_data = data.BertDataset('/media/maxim/DataSets/BERT/BERT-DATA/', vocab_size=Constants.VOCAB_SIZE.value, max_seq_len=Constants.MAX_SEQ_LEN.value)
tokenizer = _data.tokenizer
loader = DataLoader(_data, batch_size=Constants.BATCH_SIZE.value, shuffle=True)

# Print dataset vocab size to make sure that my dataset initialized correctly
util.print(len(_data.vocab), "Dataset Vocab Length: ", "\n")

# Print the first sentence in my dataset decoded
util.print(tokenizer.decode(next(iter(loader))[0][0]), "First sentence data: ", "\n")
# Print the first label sentence in my dataset
util.print(next(iter(loader))[1][0], "First label sentence data: ", "\n")

# Set vocab reverse so that I can input the id and get word
vocab = dict((v,k) for k,v in tokenizer.get_vocab().items())

bert = model.Bert(Constants.VOCAB_SIZE.value, Constants.EMB_DIM.value, Constants.NUM_LAYERS.value, Constants.NUM_HEADS.value, Constants.FEED_FORWARD_DIM.value, Constants.MAX_SEQ_LEN.value, 0.1)

LM = model.BertLM(bert, Constants.EMB_DIM.value, Constants.VOCAB_SIZE.value)
# LM = torch.load('saves/BERT_time: 11|04|2024 13:07:19|step: 60000.pt')

trainer = train.BertTrainer(LM, vocab=vocab, device=Constants.DEVICE.value)
trainer.train(loader, 5)