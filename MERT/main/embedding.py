import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device="cuda"):
        super().__init__()

        pe = torch.zeros(max_len, d_model, device=device).float()
        pe.require_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x) -> torch.Tensor:
        return self.pe
    
class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, device="cuda", dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len, device=device)
        self.dropout = nn.Dropout(p=dropout)
       
    def forward(self, sequence) -> torch.Tensor:
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)