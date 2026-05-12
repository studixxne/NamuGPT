import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
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
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0}
    ]

    # 만약 CUDA 연산이 가능하다면 Kernel Fusion을 적용한다.
    # Kernel Fusion은 필요한 연산들을 하나의 커널로 통합해서 GPU에서의 VRAM과 코어 사이의 이동 횟수를 줄여주는 하드웨어 최적화 기법이다.
    # 트레이드 오프 없이 효율성 증대가 크기 때문에 사용이 가능하다면 반드시 사용하는 것이 좋다!
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=use_fused)
    return optimizer

def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
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
        if step < warmup_steps:
            return step / warmup_steps
        
        # 총 학습 스텝이 모든 끝난 경우, 즉 추가 학습 시에는 미세 조정을 위해 최소 비율을 적용한다.
        if step > total_steps:
            return min_lr_ratio
        
        # Warmup 구간 <= step <= total_steps 일 때는 Cosine Decay를 적용한다.
        # 막 Warmup이 끝났을 때에는 진척도가 0.0이며, 전체 학습이 끝나가는 시점에는 진척도가 1.0이 된다.
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        
        # progress는 [0, 1]임으로 cos은 [-1, 1]이 된다.
        # 최소 학습 비율 이상을 반드시 유지하되 진척도가 커질수록 ratio가 1에서부터 시작하여 천천히 감소하도록 한다.
        ratio = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return ratio
    
    # 매 스텝마다 optimizer의 정해진 lr에 lr_lambda로 얻어진 ratio를 곱해서 구한 lr로 학습을 진행시키는 Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size
        self.tokens = tokenizer.encode(text, return_tensors='pt')[0].long()
        
    def __len__(self):
        return (len(self.tokens)-1) // self.block_size
        
    def __getitem__(self, index):
        data = self.tokens[index*self.block_size:index*self.block_size+self.block_size+1]
        x = data[:-1]
        y = data[1:]
        return x, y

def train(model, data_loader, optimizer, scheduler, device, epochs, total_steps, grad_clip=1.0, log_interval=10):
    model.train()
    step = 0

    for epoch in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            _, loss = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()

            # 뒤에 _가 붙으면 inplace 연산
            # loss가 튈 때 gradient가 폭발하여 학습이 망가질 수 있음으로 이를 방지하기 위해서 Grad Cliping 도입
            # 만약 L2 Norm이 5.0 이라면 grad_clip이 1.0이므로 모든 기울기에 0.2를 곱함으로써 L2 Norm을 1로 맞춰준다
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            step += 1
            if step % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'epoch [{epoch+1}|{epochs}] | step [{step}|{total_steps}] | loss {loss.item():.4f} | current_lr {current_lr:.2e}')

if __name__ == '__main__':
    from model import NanoGPT, GPTConfig

    # *=============================================*
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')

    config = GPTConfig()
    block_size = config.block_size
    batch_size = 8
    epochs = 3
    max_lr = 1e-4
    warmup_steps = 100
    text = '안녕하세요? 두쫀쿠 좋아하세요? 저는 두쫀쿠를 먹어봤는데 그냥 크런키 맛나는데 이걸 8000원 넘게 줘가면서 먹을 필요가 있을까요?'
    # *=============================================*

    dataset = GPTDataset(text, tokenizer, block_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = epochs * len(data_loader)

    model = NanoGPT(config).to(device)
    optimizer = configure_optimizer(model, lr=max_lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    train(model, data_loader, optimizer, scheduler, device, epochs, total_steps)
