import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learnable projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        """
        X: (batch_size, seq_len, d_model)
        """

        batch_size, seq_len, _ = X.shape

        # 1️ Linear projections
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2️ Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 3️ Move heads forward
        Q = Q.transpose(1, 2)  # (batch, heads, seq, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4️ Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)

        # 5️ Concatenate heads
        out = out.transpose(1, 2)  # (batch, seq, heads, d_k)
        out = out.contiguous().view(batch_size, seq_len, self.d_model)

        # 6️ Final linear layer
        out = self.W_O(out)

        return out
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        return self.linear2(self.relu(self.linear1(X)))



class EncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=8, d_ff=256, dropout=0.1):
        super().__init__()

        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        
        # 1 multi head attention
        attn_output = self.mha(X)

        # 2 Residual connection + layerNorm
        X = self.norm1(X + self.dropout(attn_output))

        # 3 feed forward network
        ffn_output = self.ffn(X)

        # 4 Residual + LayerNorm
        X = self.norm2(X + self.dropout(ffn_output))

        return X


x = torch.randn(1, 12, 64)  # batch=1, seq_len=12
encoder = EncoderBlock(d_model=64, num_heads=8)

out = encoder(x)

print(out.shape)