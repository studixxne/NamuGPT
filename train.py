import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size
        self.tokens = tokenizer.encode(text,
                                       return_tensors='pt',
                                       dtype=torch.long)[0]
        
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