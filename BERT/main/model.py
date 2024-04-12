import torch
import torch.nn as nn
from datetime import datetime

import util
from attention import MultiHeadedAttention
from embedding import BertEmbedding

class FeedForward(nn.Module):
    def __init__(self, d_model: int, middle_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 2 Linear layers in Feed Forward with Gelu in between
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, feed_forward_hidden: int, dropout: float = 0.1):
        super().__init__()

        # Combine the MultiHeadedAttention with the FeedForward layer
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)

        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    
class Bert(nn.Module):
    def __init__(self, train_args: util.TrainArguments, device: str = "cuda", dropout: float = 0.1):
        super().__init__()
        self.d_model = train_args.emb_dim
        self.n_layers = train_args.num_layers
        self.heads = train_args.num_heads
        self.max_seq_len = train_args.max_seq_len
        self.vocab_size = train_args.vocab_size
        self.feed_forward_hidden = train_args.feed_forward_dim
        self.emb_dim = train_args.emb_dim
        
        self.embedding = BertEmbedding(vocab_size=self.vocab_size, embed_size=self.d_model, seq_len=self.max_seq_len)

        # Create a list of length n_layers of my Encoder layers
        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(self.d_model, self.heads, self.feed_forward_hidden, dropout) for _ in range(self.n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create attention mask by checking what vvalues are greater than 0 (padding token)
        mask = (x>0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1).to('cuda')

        x = self.embedding(x)

        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
class BertLM(nn.Module):
    def __init__(self, bert: Bert):
        super().__init__()
        self.bert = bert

        self.linear = torch.nn.Linear(self.bert.emb_dim, self.bert.vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def save(self, step: int, epoch: int = None) -> None:
        now = datetime.now()

        dt_string = now.strftime("%d|%m|%Y %H:%M:%S")
        
        if epoch == None:
            torch.save(self, "saves/BERT_time: " + dt_string + "|step: " + step.__str__()  + ".pt")
        else:
            torch.save(self, "saves/BERT_time: " + dt_string + "|epoch: " + epoch.__str__()  + ".pt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out