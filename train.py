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
        
if __name__ == '__main__':
    # *=============================================*
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
    block_size = 16
    batch_size = 8
    text = '안녕하세요? 두쫀쿠 좋아하세요? 저는 두쫀쿠를 먹어봤는데 그냥 크런키 맛나는데 이걸 8000원 넘게 줘가면서 먹을 필요가 있을까요?'
    # *=============================================*

    dataset = GPTDataset(text, tokenizer, block_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (x, y) in enumerate(data_loader):
        print(f'{tokenizer.decode(x)}')
        print(f'{tokenizer.decode(y)}')
        break