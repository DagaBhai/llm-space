import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        init_weight = torch.randn(vocab_size,emb_dim) * (1 / (emb_dim ** 0.5))
        self.weight = nn.Parameter(init_weight)

    def forward(self, x):
        return self.weight[x]

class PositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        init_weight = torch.randn(vocab_size,emb_dim) * (1 / (emb_dim ** 0.5))
        self.weight = nn.Parameter(init_weight)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device)
        return self.weight[position]

if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size, embed_dim = 100, 756
    max_seq_len = 50 
    
    # Instantiate both embedding layers
    my_token_embedding = Embeddings(vocab_size, embed_dim)
    my_pos_embedding = PositionalEmbeddings(max_seq_len, embed_dim)

    # Input sequence of token IDs (batch of 2 sequences, 3 tokens each)
    input_ids = torch.tensor([[1, 4, 9], [2, 8, 4]]) 
    
    # 1. Get token embeddings
    token_vectors = my_token_embedding(input_ids)
    
    # 2. Get positional embeddings
    pos_vectors = my_pos_embedding(input_ids)
    
    # 3. Combine them (PyTorch automatically broadcasts the sequence length across the batch)
    final_embeddings = token_vectors + pos_vectors

    print("Input Shape:           ", input_ids.shape)
    print("Token Embedding Shape: ", token_vectors.shape)
    print("Pos Embedding Shape:   ", pos_vectors.shape)
    print("Final Output Shape:    ", final_embeddings.shape)