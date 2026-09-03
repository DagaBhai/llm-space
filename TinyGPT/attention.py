import torch.nn as nn
import math
import torch

class SelfAttention(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.Wq = nn.Linear(dimension,dimension,bias=False)
        self.Wk = nn.Linear(dimension,dimension,bias=False)
        self.Wv = nn.Linear(dimension,dimension,bias=False)
        
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = Q @ K.transpose(-2,-1) #Only swaps the last two dimensions 
        scores = scores / math.sqrt(x.size(-1))
        seq_len = scores.size(-1)
        mask =  torch.tril(torch.ones(seq_len,seq_len,device=scores.device))
        scores = scores.masked_fill(mask == 0 , float('-inf'))
        attention_weights = nn.Softmax(dim=-1)(scores)
        output = attention_weights @ V
        
        return output, attention_weights

if __name__ == '__main__':
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    embed_dim = 8

    x = torch.randn(batch_size, seq_len, embed_dim)
    attn_layer = SelfAttention(dimension=embed_dim)
    output, weights = attn_layer(x)

    print("--- Test Output Shapes ---")
    print(f"Input shape:             {x.shape}")
    print(f"Output shape:            {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch!"
    assert weights.shape == (batch_size, seq_len, seq_len), "Weights shape mismatch!"

    print("\n--- Testing Causal Mask ---")
    upper_triangle_sum = torch.triu(weights[0], diagonal=1).sum().item()
    print(f"Sum of upper triangle attention (should be 0.0): {upper_triangle_sum:.4f}")
    assert math.isclose(upper_triangle_sum, 0.0, abs_tol=1e-6), "Causal mask failed!"

    print("\nAll tests passed successfully!")
