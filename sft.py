import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from train import (
    TrainConfig, configure_optimizer, get_lr_scheduler,
    load_checkpoint, train
)

from utils import get_device, save_loss_plot
from model import NanoGPT

PROMPT_TEMPLATE = "<|user|>{instruction}<|answer|>"

class SFTDataset(Dataset):
    def __init__(self, samples):
        # samples: list of (x, y) tensor 쌍
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_sft_samples(dataset, tokenizer, block_size):
    eos_id  = tokenizer.eos_token_id
    samples = []
    skipped = 0

    for item in tqdm(dataset, desc='Building SFT samples'):
        # Prompt Template에 instruction을 삽입 후 토큰화
        prompt_ids   = tokenizer.encode(PROMPT_TEMPLATE.format(instruction=item['instruction']),
                                        add_special_tokens=False)
        
        # output 토큰화
        response_ids = tokenizer.encode(item['output'], add_special_tokens=False)

        # 하나의 list로 합치기
        full_ids = prompt_ids + response_ids + [eos_id]

        # Block Size 1024를 맞춰주기 위해서 길이를 넘어가는 경우 Cut
        if len(full_ids) > block_size + 1:
            full_ids = full_ids[:block_size + 1]

        # 잘린 후 Answer가 하나도 안 남아있는 경우에는 해당 Sample을 제거
        if len(full_ids) <= len(prompt_ids):
            skipped += 1
            continue

        # (x, y) 샘플 만들기
        # y는 full_ids를 1칸 shift해 x가 입력으로 들어올 때 바로 다음 토큰을 예측하도록 한다
        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:],  dtype=torch.long)

        # y의 Prompt 부분은 -100으로 Masking 처리 해, 따로 학습을 하지 않도록 함
        y[:len(prompt_ids)-1] = -100

        # 길이를 Block Size로 모두 통일 시켜준다.
        pad_len = block_size - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])

        samples.append((x, y))

    print(f'# SFT 샘플 생성 완료: {len(samples)}개 (스킵: {skipped}개)')
    return samples

if __name__ == '__main__':
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_ckpt',  type=str,   default='models/pt_best_model.pt')
    parser.add_argument('--load',           action='store_true', help='SFT 체크포인트에서 재개')
    parser.add_argument('--epochs',         type=int,   default=None)
    parser.add_argument('--max_lr',         type=float, default=None)
    parser.add_argument('--warmup_steps',   type=int,   default=None)
    parser.add_argument('--eval_interval',  type=int,   default=None)
    parser.add_argument('--log_interval',   type=int,   default=None)
    parser.add_argument('--grad_accum',     type=int,   default=None)
    parser.add_argument('--batch_size',     type=int,   default=None)
    args = parser.parse_args()

    # *=============================================*
    device = get_device()

    train_cfg = TrainConfig(
        epochs        = 3,
        max_lr        = 6e-5,
        warmup_steps  = 100,
        eval_interval = 200,
        log_interval  = 50,
        batch_size    = 4,
        grad_accum    = 8,
        ckpt_dir      = 'checkpoints',
        history_dir   = 'histories',
        train_type    = 'sft'
    )
    # *=============================================*

    # Arg Parsing 후 Hyper Parameter 설정
    for key, val in vars(args).items():
        if key not in ('pretrain_ckpt', 'load') and val is not None and hasattr(train_cfg, key):
            setattr(train_cfg, key, val)

    # 토크나이저 초기화
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        'skt/kogpt2-base-v2',
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )

    # 사전학습 체크포인트에서 모델 구조 및 파라미터 로드
    ckpt      = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt['model_cfg']
    model     = NanoGPT(model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'# 사전학습 모델 로드 완료: {args.pretrain_ckpt}')

    # KoAlpaca 로드 및 샘플 생성
    koalpaca = load_dataset('beomi/KoAlpaca-v1.1a', split='train').shuffle(seed=39)
    samples  = build_sft_samples(koalpaca, tokenizer, model_cfg.block_size)

    # train/val 95:5 분리
    split         = int(len(samples) * 0.95)
    train_dataset = SFTDataset(samples[:split])
    val_dataset   = SFTDataset(samples[split:])

    # Loader 초기화
    pin_memory   = (device == 'cuda')
    num_workers  = 4 if device == 'cuda' else 0
    persistent   = (num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=train_cfg.batch_size, shuffle=False,
                              num_workers=num_workers-2, pin_memory=pin_memory, persistent_workers=persistent, drop_last=True)

    # Optimizer, Scheduler 초기화
    total_steps  = train_cfg.epochs * len(train_loader)
    optimizer = configure_optimizer(model, train_cfg, device)
    scheduler = get_lr_scheduler(optimizer, total_steps // train_cfg.grad_accum, train_cfg)

    # 만약 SFT 체크포인트 존재 시 Load
    start_step    = 0
    best_val_loss = float('inf')
    if args.load:
        ckpt_path = os.path.join(train_cfg.ckpt_dir, train_cfg.train_type + '_latest_model.pt')
        assert os.path.exists(ckpt_path), f'# SFT 체크포인트가 없습니다: {ckpt_path}'
        start_step, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        print(f'# SFT 체크포인트 로드 완료: step {start_step}에서 재개')

    # Cuda 환경일 시 최적화를 위해서 Compile
    if device == 'cuda':
        model = torch.compile(model)

    # 학습 시작
    train(model, train_loader, val_loader, optimizer, scheduler,
          device, total_steps, train_cfg, start_step, best_val_loss)
    
    # Loss Graph 생성
    save_loss_plot(train_cfg.history_dir, 'sft_loss.json')