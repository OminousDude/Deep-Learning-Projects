import torch
import torch.nn as nn
from datetime import datetime

from attention import MultiHeadedAttention
from embedding import BertEmbedding

class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, feed_forward_hidden, dropout=0.1):
        super().__init__()

        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask) -> torch.Tensor:
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)

        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    
class Bert(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, feed_forward_dim, seq_len, device="cuda", dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        self.feed_forward_hidden = feed_forward_dim
        
        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=seq_len)

        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(d_model, heads, feed_forward_dim, dropout) for _ in range(n_layers)])

    def forward(self, x) -> torch.Tensor:
        mask = (x>0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1).to('cuda')

        x = self.embedding(x)

        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
class BertLM(nn.Module):
    def __init__(self, bert: Bert, emb_dim, vocab_size):
        super().__init__()
        self.bert = bert
        self.linear = torch.nn.Linear(emb_dim, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def save(self, step, epoch = None) -> None:
        now = datetime.now()

        dt_string = now.strftime("%d|%m|%Y %H:%M:%S")
        
        if epoch == None:
            torch.save(self, "saves/BERT_time: " + dt_string + "|step: " + step.__str__()  + ".pt")
        else:
            torch.save(self, "saves/BERT_time: " + dt_string + "|epoch: " + epoch.__str__()  + ".pt")

    def forward(self, x) -> torch.Tensor:
        x = self.bert(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out