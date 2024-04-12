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