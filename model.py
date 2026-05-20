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

        self.config = config
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

        """
        ** 직접 구현한 Pure Causal Attention **

        weight = (Q @ K.transpose(-1, -2) / (math.sqrt(self.d_head))).masked_fill(self.mask[:, :, :n, :n] == 0, float('-inf'))
        attn = self.attention_dropout(F.softmax(weight, dim=-1))  # (b, h, n, n)

        # attn @ V를 통해서 각 단어에 대한 평균 Value를 구해준다.
        # (n, n) @ (n, d) -> (n, d)
        out_attn = attn @ V
        """

        # 더 효율적인 학습을 위해 Flash Attention 사용
        out_attn = F.scaled_dot_product_attention(Q, K, V, 
                                                  attn_mask=None, 
                                                  dropout_p=self.config.dropout if self.training else 0.0,
                                                  is_causal=True)

        # 그 후 Wo를 통해 Concat 진행 후 Dropout 적용
        out = self.final_dropout(self.Wo(rearrange(out_attn, 'b h n d -> b n (h d)')))
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
    
class NanoGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config  # generate()에서 block_size를 참조하기 위해 저장

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_ln = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        # x: (batch, n)
        device = x.device
        b, n = x.shape

        x = self.token_embed(x) + self.position_embed(torch.arange(0, n, dtype=torch.long, device=device))
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        # (b, n, d)
        x = self.final_ln(x)

        # 학습 중일 때는 모든 토큰들에 대한 logits을 계산
        if targets is not None:
            logits = self.lm_head(x) # (b, n, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        else:
            # 추론의 경우에는 마지막 토큰에 대해서만 연산을 진행해 계산량을 줄임.
            # 이 때 x[:, [-1], :]로 인덱싱 함으로써 (b, 1, vocab_size)로 차원 유지 (Keepdims=True)
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        tokens: (b, n) - 프롬프트 토큰 인덱스
        max_new_tokens: int -> 새로 생성될 수 있는 최대 토큰 개수
        temperature: float -> 낮을수록 Greedy와 가까워지고 높아질수록 다양성 증가
        top_k: int -> 계산된 확률의 상위 k개의 토큰만 후보로 남기고 최종적으로 토큰을 선정함
        """

        for _ in range(max_new_tokens):
            # Context Length가 Block Size를 초과하면 가장 최근의 토큰만 읽어냄
            tokens_cond = tokens[:, -self.config.block_size:]

            logits, _ = self(tokens_cond)           # (b, 1, vocab_size)
            logits = logits[:, -1, :] / temperature # (b, vocab_size)

            # 상위 K개의 토큰만 선정할 수 있도록 하여 필요없는 꼬리 확률들의 단어들을 모두 잘라냄
            if top_k is not None:
                # 상위 k개의 값들을 가져옴
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # (b, top_k)

                # top_k value 중에서 가장 작은 값보다 작은 logit들을 -inf로 마스킹
                logits[logits < topk_vals[:, [-1]]] = float('-inf')

            # top_k에 속해있는 Vocab에 대해서만 확률 재정규화
            probs = F.softmax(logits, dim=-1)
            # probs에 기반하여 1개 샘플링
            new_token = torch.multinomial(probs, num_samples=1)  # (b, 1)
            # 새로 샘플링한 토큰과 누적된 토큰을 concat 해줌
            tokens = torch.cat([tokens, new_token], dim=-1)

        return tokens