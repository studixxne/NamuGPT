import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.head_num = config.head_num
        self.d_head = int(self.d_model / self.head_num)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.final_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size))
        )

        self.Wq = nn.Linear(self.d_model, self.head_num*self.d_head, bias=False)
        self.Wk = nn.Linear(self.d_model, self.head_num*self.d_head, bias=False)
        self.Wv = nn.Linear(self.d_model, self.head_num*self.d_head, bias=False)
        self.Wo = nn.Linear(self.head_num*self.d_head, self.d_model, bias=False)

    def forward(self, x):
        # x: (b, n, d_model)
        n = x.shape[1]

        # d_attn: head_num * head_dim
        Q = rearrange(self.Wq(x), 'b n (h d) -> b h n d', h=self.head_num)
        K = rearrange(self.Wk(x), 'b n (h d) -> b h n d', h=self.head_num)
        V = rearrange(self.Wv(x), 'b n (h d) -> b h n d', h=self.head_num)

        weight = (Q @ K.transpose(-1, -2) / (math.sqrt(self.d_head))).masked_fill(self.mask[:, :, :n, :n] == 0, float('-inf'))
        weight_sum = self.attention_dropout(F.softmax(weight, dim=-1))
        out = self.final_dropout(self.Wo(rearrange(weight_sum, 'b h n d -> b n (h d)')))
        return out
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.W1 = nn.Linear(config.d_model, 4*config.d_model)
        self.W2 = nn.Linear(4*config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.W1(x))
        x = self.W2(x)
        out = self.dropout(x)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = CausalSelfAttention(config)
        self.pre_ln = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
        self.post_ln = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.attention(self.pre_ln(x))
        out = x + self.mlp(self.post_ln(x))
        return out