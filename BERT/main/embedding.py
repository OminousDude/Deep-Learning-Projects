import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: str = "cuda"):
        super().__init__()

        # Create tensor of zeroes
        pe = torch.zeros(max_len, d_model, device=device).float()
        pe.require_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                # Add the sin and cos to the embedding
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self) -> torch.Tensor:
        return self.pe
    
class BertEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, seq_len: int, device: str = "cuda", dropout: float = 0.1):
        super().__init__()
        # Combine the nn.Embedding with the PositionalEmbedding class above
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len, device=device)
        self.dropout = nn.Dropout(p=dropout)
       
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.token(sequence) + self.position()
        return self.dropout(x)