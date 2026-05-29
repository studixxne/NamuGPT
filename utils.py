import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from contextlib import nullcontext

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def autocast_ctx(device):
    if device == 'cpu':
        return nullcontext()

    # cuda/mps 사용 중일 경우 BF16 사용
    return torch.amp.autocast(device, dtype=torch.bfloat16)

def save_loss_plot(ckpt_dir, file_name):
    log_path = os.path.join(ckpt_dir, file_name)

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
    save_name = os.path.splitext(file_name)[0] + '.png'
    fig.savefig(os.path.join(ckpt_dir, save_name))
    plt.close(fig)