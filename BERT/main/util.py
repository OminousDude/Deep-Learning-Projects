import torch
import numpy as np
import random

def random_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# Set base print to _print so that i can call my print the same as normal
_print = print

def print(val: object, str_start: str = None, str_end: str = None) -> None:
    str_val = str(val)
    _print(str_start + str_val + str_end)

class TrainArguments():
    def __init__(self, batch_size, max_seq_len, vocab_size, num_heads, emb_dim, num_layers, pos_enc_len = None, feed_forward_dim = None) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.pos_enc_len = pos_enc_len
        self.emb_dim = emb_dim
        self.feed_forward_dim = feed_forward_dim
        self.num_layers = num_layers

        if not feed_forward_dim:
            self.feed_forward_dim = emb_dim * 4
        if not pos_enc_len:
            self.pos_enc_len = self.max_seq_len