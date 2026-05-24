# NanoGPT

한국어 NanoGPT-2 구현과 사전학습부터 사후학습까지

---

## Model

| Parameter | Value |
|---|---|
| Total Parameter | 125.1M |
| Number of Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Context length | 1024 |
| Vocab size | 51,200 (skt/kogpt2-base-v2) |

---

## Pre-training

| Hyperparameter | Value |
|---|---|
| Dataset | 나무위키 덤프 (heegyu/namuwiki-extracted) |
| Training Tokens | 2.32B |
| Total steps | 141,492 |
| Batch size | 131,072 tokens/update |
| Optimizer | AdamW (fused) |
| LR Schedule | Warmup + Cosine Decay |
| Mixed Precision | BF16 |

### Loss

![loss](histories/loss.png)

---