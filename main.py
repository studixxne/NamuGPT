import json
import readline

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from model import GPTConfig, NanoGPT
from transformers import PreTrainedTokenizerFast

from utils import get_device

# * ============================================= *
HF_REPO_ID         = 'hinagiku/NanuGPT-125M-Instruct'
PROMPT_TEMPLATE    = '<|user|>{instruction}<|answer|>'
MAX_NEW_TOKENS     = 200
TEMPERATURE        = 0.8
TOP_K              = 80
TOP_P              = 0.9
REPETITION_PENALTY = 1.3
# * ============================================= *

device = get_device()
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2',
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>'
)

config_path  = hf_hub_download(repo_id=HF_REPO_ID, filename='config.json')
weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename='model.safetensors')

with open(config_path) as f:
    model_cfg = GPTConfig(**json.load(f))

model = NanoGPT(model_cfg).to(device)
loaded_state_dict = load_file(weights_path, device=device)
loaded_state_dict["token_embed.weight"] = loaded_state_dict["lm_head.weight"] # Weight Tyning 적용
model.load_state_dict(loaded_state_dict)
model.eval()

print('🌱 Welcome To NanuGPT-125M!\n')
print(f'☘️  NanuGPT: 무엇을 알려드릴까요?\n')

while True:
    prompt = input('👤 user: ').strip()

    if not prompt:
        continue

    input_ids = tokenizer.encode(PROMPT_TEMPLATE.format(instruction=prompt))
    tokens    = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(tokens, max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)

    result = tokenizer.decode(output_ids[0].tolist())
    # <|answer|> 이후 텍스트만 추출
    answer = result.split('<|answer|>')[-1].strip()

    # EOS 이전까지만 사용
    if '</s>' in answer:
        answer = answer.split('</s>')[0].strip()

    print(f'\n☘️  NanuGPT: {answer}\n')