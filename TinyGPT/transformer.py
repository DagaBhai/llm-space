import torch
import torch.nn as nn

from attention import SelfAttention
from tokenizer import Tokenizer
from embeddings import PositionalEmbeddings, Embeddings

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        super().__init__()

        self.embedding = Embeddings(vocab_size, d_model)
        self.pos_embedding = PositionalEmbeddings(max_seq_len, d_model)

        self.attention = SelfAttention(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding(x)
        x, weights = self.attention(x)
        logits = self.lm_head(x)
        return logits
