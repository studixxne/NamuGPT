import json
import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from utils import get_device, autocast_ctx

@dataclass
class TrainConfig:
    batch_size:    int        = 32              # Batch Size
    epochs:        int        = 2               # Epoch 횟수
    max_lr:        float      = 6e-4            # Learning Rate
    weight_decay:  float      = 0.1             # Weight Decay 강도    
    betas:         tuple      = (0.9, 0.95)     # AdamW beta 설정
    warmup_steps:  int        = 500             # warmup step 횟수
    min_lr_ratio:  float      = 0.1             # 최소로 보장되는 학습 비율 (ratio * lr이 최종 학습률)
    grad_clip:     float      = 1.0             # Gradient Cliping 기준
    log_interval:  int        = 50              # 로그 기록 간격
    eval_interval: int        = 1000            # Evaluation 간격
    ckpt_dir:      str        = 'checkpoints'   # CheckPoint 파일 저장할 Dir 위치
    history_dir:   str        = 'histories'     # Loss History를 기록할 Dir 위치
    val_ratio:     float      = 0.005           # Validation 데이터에 사용할 비율
    grad_accum:    int        = 4               # Grad accumulation 횟수
    max_articles:  int | None = None            # int이거나 None (학습에 사용할 article의 개수)

def configure_optimizer(model, train_cfg: TrainConfig, device):
    """
    [GPT를 더 효율적으로 학습하기 위해 Optimizer를 설정]

    1. 왜 AdamW인가?
        - Adam의 경우 Loss를 계산할 때 L2 정규화를 포함하여 계산한다.
        - 그렇기 때문에 dLoss에서 L2 정규화의 미분 결과인 lambda * weight 항이 남아 있게 되고 이는 Adam의 m과 v를 갱신할 때 정규화 항의 기울기까지 반영하게 되는 문제가 존재한다.
        - 이를 해결하기 위해서 AdamW는 L2 정규화를 Loss 계산에서 분리하고 weight 업데이트할 때 따로 빼줌으로써 순수하게 dLoss만 구할 수 있도록 해준다.
        - 결과적으로 m, v의 업데이트가 순수 모델의 dLoss로만 이뤄지기 때문에 대부분의 경우 AdamW가 Adam보다 더 좋은 성능을 낸다. (Weight Decay가 없을 땐 차이 X)

    2. ** Model의 파라미터를 체크해서 1D의 경우에는 Weight Decay를 적용하지 않는다. **
        - Weight Decay는 기본적으로 파라미터의 값이 매우 커지지 않도록 패널티를 주는 시스템이다.
        - Parameter의 값이 매우 커지게 되면 작은 노이즈에도 쉽게 반응하게 되며 Overfitting 되는 현상이 발생한다.
        - 그렇기 때문에 실제 데이터와 곱해지게 되는 2D 이상의 파라미터들은 이를 해결하기 위해서 Weight Decay를 적용해야 한다.

        - 반면 1D의 경우에는 곱셈 연산을 하는 것이 아니라, Shift/Scale 작업을 위한 파라미터다.
        - 최적의 크기와 위치를 학습해야 하는데 Weight Decay를 적용하게 되면 계속해서 0으로 끌어당겨지게 되어 올바른 학습이 어려워진다.
        - 예를 들어 LayerNorm에서 γ가 0에 가까워지게 되면 분산이 0에 가까워지고 이는 학습을 멈추게 한다.
        - 혹은 Bias의 경우 0에 가까워지게 되면 발화점을 능동적으로 조절할 수 없는 문제가 발생한다.
    """

    decay, no_decay = [], []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 2차원 이상은 weight decay 적용
        if param.dim() >= 2:
            decay.append(param)

        # 1차원의 경우는 weight decay 적용 X
        else:
            no_decay.append(param)
    
    param_groups = [
        {'params': decay,    'weight_decay': train_cfg.weight_decay},
        {'params': no_decay, 'weight_decay': 0}
    ]

    # 만약 CUDA 연산이 가능하다면 Kernel Fusion을 적용한다.
    # Kernel Fusion은 필요한 연산들을 하나의 커널로 통합해서 GPU에서의 VRAM과 코어 사이의 이동 횟수를 줄여주는 하드웨어 최적화 기법이다.
    # 트레이드 오프 없이 효율성 증대가 크기 때문에 사용이 가능하다면 반드시 사용하는 것이 좋다!
    use_fused = (device == 'cuda')
    optimizer = torch.optim.AdamW(param_groups, lr=train_cfg.max_lr, betas=train_cfg.betas, fused=use_fused)
    return optimizer

def get_lr_scheduler(optimizer, total_steps, train_cfg: TrainConfig):
    """
        Cosine Decay를 바로 적용하기 이전에 Warmup을 적용함으로써 lr을 작게 설정해 초기 학습을 안정화시켜준다.

        1. 사전학습 초기에는 Weight가 무작위 분포이기 때문에 Gradient가 불안정함으로 Warmup 단계가 필요하다. (하지만 Pre-LN 구조일 때 (1)의 경우 WarmUp이 없어도 학습이 안정화된다.)
        2. Optimizer Adam은 초기 v의 값이 0이기 때문에 지정한 lr보다 매우 커짐으로 이를 예방하기 위해 초기에는 lr을 매우 작게 설정해줘야 한다.

        -> 이를 위해서 매 step에 따라 직접 비율을 조절할 수 있는 lr_scheduler.LambdaLR 사용
    """

    # learning rate에 곱해질 비율 (Ratio, 0.0 ~ 1.0)을 반환하는 함수
    def lr_lambda(step):
        # Warmup 구간 (step < warmup_steps)
        # step이 0에서 warmup_steps로 올라가면서 ratio가 0.0에서 1.0으로 선형적으로 증가하게 된다.
        if step < train_cfg.warmup_steps:
            return step / train_cfg.warmup_steps

        # 총 학습 스텝이 모든 끝난 경우, 즉 추가 학습 시에는 미세 조정을 위해 최소 비율을 적용한다.
        if step > total_steps:
            return train_cfg.min_lr_ratio

        # Warmup 구간 <= step <= total_steps 일 때는 Cosine Decay를 적용한다.
        # 막 Warmup이 끝났을 때에는 진척도가 0.0이며, 전체 학습이 끝나가는 시점에는 진척도가 1.0이 된다.
        progress = (step - train_cfg.warmup_steps) / (total_steps - train_cfg.warmup_steps)

        # progress는 [0, 1]임으로 cos은 [-1, 1]이 된다.
        # 최소 학습 비율 이상을 반드시 유지하되 진척도가 커질수록 ratio가 1에서부터 시작하여 천천히 감소하도록 한다.
        ratio = train_cfg.min_lr_ratio + 0.5 * (1 - train_cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return ratio
    
    # 매 스텝마다 optimizer의 정해진 lr에 lr_lambda로 얻어진 ratio를 곱해서 구한 lr로 학습을 진행시키는 Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

# dataset을 가져와서 max_articles의 개수만큼의 text를 encode하여 flat한 torch로 반환
def build_corpus(dataset, tokenizer, max_articles=None):
    if max_articles is not None:
        print(f'# {max_articles} 개의 문서를 활용해서 인코딩을 수행합니다.')
        dataset = dataset.select(range(min(max_articles, len(dataset))))
    else:
        print(f'# 전체 데이터셋인 {len(dataset)} 개의 문서를 활용해서 인코딩을 수행합니다.')

    def process_function(examples):
        outputs = tokenizer(examples['text'], add_special_tokens=False)
        return {"ids": outputs["input_ids"]}
    
    '''
    ** dataset.map **

    dataset의 문서들에 process_function 작업 수행
    batched=True -> batch로 묶어서 연산 수행
    num_proc = 코어 개수
    remove_columns = 함수 적용이 끝나면 제거할 원본 column 
    (dataset.column_names를 통해 'title', 'text' 등 모든 열 제거)

    '''
    print(f'# 토크나이징을 수행합니다.')
    tokenized_dataset = dataset.map(
        process_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc='Tokenizing'
    )

    print(f'# 토큰을 단일 배열로 통합합니다.')
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, 'eos_id는 반드시 존재해야 합니다'

    # ids를 리스트로 보관하면 Memory 요구량이 급격하게 증가함으로 np.array로 보관
    chunks = []
    for row in tqdm(tokenized_dataset, desc="Merging tokens"):
        chunks.append(np.array(row['ids']+[eos_id], dtype=np.int32))
    
    final_tokens = np.concatenate(chunks)
    print(f'# 인코딩 완료! 최종 토큰 개수: {len(final_tokens)/1e6:1f}M')

    return torch.from_numpy(final_tokens.astype(np.int64))

class GPTDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index):
        data = self.tokens[index*self.block_size : index*self.block_size + self.block_size + 1]
        x = data[:-1]
        y = data[1:]
        return x, y

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, cnt = 0.0, 0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device):
            _, loss = model(x, targets=y)
        total_loss += loss.item()
        cnt += 1

    model.train()
    return total_loss / cnt

def save_checkpoint(path, model, optimizer, scheduler, step, val_loss, train_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_cfg': raw_model.config,
        'train_cfg': train_cfg
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    check_point = torch.load(path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    scheduler.load_state_dict(check_point['scheduler_state_dict'])
    return check_point['step'], check_point['val_loss']

def _save_loss_plot(ckpt_dir):
    log_path = os.path.join(ckpt_dir, 'loss_log.json')

    if not os.path.exists(log_path):
        return
    
    with open(log_path) as f:
        logs = json.load(f)
    train_log = logs['train']
    val_log   = logs['val']

    fig, ax = plt.subplots()
    if train_log:
        steps, losses = zip(*train_log)
        ax.plot(steps, losses, label='train')
    if val_log:
        steps, losses = zip(*val_log)
        ax.plot(steps, losses, label='val')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(os.path.join(ckpt_dir, 'loss.png'))
    plt.close(fig)

def train(model, train_loader, val_loader, optimizer, scheduler, device, total_steps, train_cfg: TrainConfig, start_step=0, start_val_loss=float('inf')):
    model.train()
    step = start_step
    best_val_loss = start_val_loss
    train_log = []  # (step, loss)
    val_log   = []  # (step, val_loss)

    steps_per_epoch = len(train_loader)
    start_epoch = start_step // steps_per_epoch

    def save_model(current_step):
        nonlocal best_val_loss

        val_loss = evaluate(model, val_loader, device)
        val_log.append((current_step, val_loss))

        os.makedirs(train_cfg.history_dir, exist_ok=True)
        with open(os.path.join(train_cfg.history_dir, 'loss_log.json'), 'w') as f:
            json.dump({'train': train_log, 'val': val_log}, f)

        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(os.path.join(train_cfg.ckpt_dir, 'best_model.pt'),
                            model, optimizer, scheduler, current_step, best_val_loss, train_cfg)

        # 최신 모델 저장
        save_checkpoint(os.path.join(train_cfg.ckpt_dir, 'latest_model.pt'),
                        model, optimizer, scheduler, current_step, best_val_loss, train_cfg)

    print(f'# Training Start')
    for epoch in tqdm(range(start_epoch, train_cfg.epochs)):
        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)

            # 혼합 정밀도 BF16을 사용함으로써 연산 최적화 (CUDA만 적용)
            with autocast_ctx(device):
                _, loss = model(x, targets=y)
                # grad_accum 만큼 나눠준다.
                loss = loss / train_cfg.grad_accum

            # 미분값 누적
            loss.backward()

            # grad_accum만큼의 누적 횟수가 도달했을 때만 가중치를 업데이트한다.
            # 이러한 방식을 통해서 VRAM이 부족한 상황에서도 더 큰 batch_size를 사용하는 것과 동일한 효과를 냄으로써 좀 더 Robust해지고 학습 효율 증가!
            if step % train_cfg.grad_accum == 0:
                # 뒤에 _가 붙으면 inplace 연산
                # loss가 튈 때 gradient가 폭발하여 학습이 망가질 수 있음으로 이를 방지하기 위해서 Grad Cliping 도입
                # 만약 L2 Norm이 5.0 이라면 grad_clip이 1.0이므로 모든 기울기에 0.2를 곱함으로써 L2 Norm을 1로 맞춰준다
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % train_cfg.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                train_log.append((step, loss.item() * train_cfg.grad_accum))
                tqdm.write(f'epoch [{epoch+1}|{train_cfg.epochs}] | step [{step}|{total_steps}] | loss {loss.item() * train_cfg.grad_accum:.4f} | current_lr {current_lr:.2e}')

            # 일정 주기로 일반화 성능을 체크하기 위해 val_loss 계산 및 출력
            if step % train_cfg.eval_interval == 0:
                save_model(step)

        if step % train_cfg.grad_accum != 0:
            optimizer.zero_grad()

        save_model(step)

if __name__ == '__main__':
    import argparse
    from datasets import load_dataset
    from model import NanoGPT, GPTConfig

    # FP32 데이터의 행렬 곱셈 연산 방식을 TF32로 변경함으로써 최적화
    # 소수점 일부 반올림함으로써 가속 연산!
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load',          action='store_true', help='latest_model.pt에서 학습 재개')
    parser.add_argument('--batch_size',    type=int,   default=None)
    parser.add_argument('--epochs',        type=int,   default=None)
    parser.add_argument('--grad_accum',    type=int,   default=None)
    parser.add_argument('--max_lr',        type=float, default=None)
    parser.add_argument('--warmup_steps',  type=int,   default=None)
    parser.add_argument('--log_interval',  type=int,   default=None)
    parser.add_argument('--eval_interval', type=int,   default=None)
    parser.add_argument('--max_articles',  type=int,   default=None)
    args = parser.parse_args()

    # *=============================================*
    device    = get_device()
    model_cfg = GPTConfig()
    train_cfg = TrainConfig()
    # *=============================================*

    # Hyper Parameter 적용
    for key, val in vars(args).items():
        if key != 'load' and val is not None and hasattr(train_cfg, key):
            setattr(train_cfg, key, val)

    # Tokenizer 초기화
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        'skt/kogpt2-base-v2',
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )

    # Token Cache 경로 생성
    cache_dir = 'token_cache'
    os.makedirs(cache_dir, exist_ok=True)
    articles_num = 'full' if train_cfg.max_articles is None else train_cfg.max_articles
    token_cache_path = f"token_cache/tokens_{articles_num}.pt"

    # 만약 Token Cache가 존재한다면 cache를 불러온다.
    # 존재하지 않는 경우에는 Tokinizing 수행 후, Torch Tensor로 변환 후 Cache로 저장
    if os.path.exists(token_cache_path):
        tokens = torch.load(token_cache_path)
    else:
        wiki = load_dataset('heegyu/namuwiki-extracted', split='train').shuffle(seed=39)
        tokens = build_corpus(wiki, tokenizer, max_articles=train_cfg.max_articles)
        torch.save(tokens, token_cache_path)

    # val_ratio 만큼 train/val 데이터 분할
    split = int(len(tokens) * (1 - train_cfg.val_ratio))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    train_dataset = GPTDataset(train_tokens, model_cfg.block_size)
    val_dataset = GPTDataset(val_tokens, model_cfg.block_size)

    # pin_memory와 persistent_workers를 True로 설정함으로써 고속 메모리 구역에 고정시키고 프로세스를 유지함으로써 성능 개선
    # drop_last를 True로 설정함으로써 자투리 Batch 버린다. 이를 통해 Shape의 크기가 변하면서 발생할 수 있는 에러나 비효율성 예방
    pin_memory = (device == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin_memory, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin_memory, persistent_workers=True, drop_last=True)
    total_steps = train_cfg.epochs * len(train_loader)
    
    # Model, Optimizer, Scheduler 생성
    model     = NanoGPT(model_cfg).to(device)
    optimizer = configure_optimizer(model, train_cfg, device)
    scheduler = get_lr_scheduler(optimizer, total_steps // train_cfg.grad_accum, train_cfg)

    # Model Checkpoint Load
    start_step = 0
    best_val_loss = float('inf')
    if args.load:
        ckpt_path = os.path.join(train_cfg.ckpt_dir, 'latest_model.pt')
        assert os.path.exists(ckpt_path), f'# 체크포인트 파일이 존재하지 않습니다: {ckpt_path}'
        start_step, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        print(f'# 체크포인트 로드 완료: Step {start_step}에서 재개합니다.')

    # 모델 Compile, 학습 최적화
    # Compile시 Model이 래퍼로 감싸지면서 Key 앞에 _orig_mod라는 Prefix가 붙게 된다.
    # 고로 Save와 Load할 때 Key에 주의!
    if device == 'cuda':
        model = torch.compile(model)

    # Parameter, Token 크기 체크
    param_set = set(p for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in param_set)
    print(f'# Model Parameter: {n_params/1e6:.1f}M')
    print(f'# Total training steps: {total_steps}')
    print(f'# Tokens per step: {train_cfg.batch_size * model_cfg.block_size:,}')
    print(f'# Tokens per update: {train_cfg.grad_accum * train_cfg.batch_size * model_cfg.block_size:,}')
    print(f'# Total training tokens: {total_steps * train_cfg.batch_size * model_cfg.block_size / 1e9:.2f}B')    

    # 학습 시작
    train(model, train_loader, val_loader, optimizer, scheduler, device, total_steps, train_cfg, start_step, best_val_loss)

    # Loss graph 생성
    _save_loss_plot(train_cfg.history_dir)