import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    # 타입 어노테이션이 있을 경우에는 Instance Variable
    # 타입 어노테이션이 없는 경우에는 Class Variable (모든 Instance가 공유)
    vocab_size: int = 51200   # skt/kogpt2-base-v2 vocabulary size
    block_size: int = 256     # 최대 context 길이 (position embedding 크기)
    d_model: int = 256        # embedding / hidden 차원
    n_layer: int = 6          # TransformerBlock 개수
    head_num: int = 8         # attention head 수 → d_head = d_model / head_num = 32
    dropout: float = 0.1
    
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
        attn = self.attention_dropout(F.softmax(weight, dim=-1))  # (b, h, n, n)

        # attn @ V를 통해서 각 단어에 대한 평균 Value를 구해준다.
        # (n, n) @ (n, d) -> (n, d)
        # 그 후 Wo를 통해 Concat 진행 후 Dropout 적용
        out = self.final_dropout(self.Wo(rearrange(attn @ V, 'b h n d -> b n (h d)')))
        return out
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.W1 = nn.Linear(config.d_model, 4*config.d_model)
        self.W2 = nn.Linear(4*config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 연산 효율을 위하여 gelu를 적용할 때 복잡한 Phi 대신 근사적인 연산으로 tanh를 사용한다.
        # 차원을 4배로 늘린 후, gelu를 통해 필요없는 feature들을 정리 후 다시 차원 축소!
        x = F.gelu(self.W1(x), approximate='tanh')
        out = self.dropout(self.W2(x))
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