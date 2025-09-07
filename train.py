# trainer script to train GPT models formatted properly for GPT.c
# currently works with tinyshakespeare dataset

# TODO: CLI mode.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import time
import math
from model_gpt import GPT, GPTModelArgs

# training config
class TrainConfig:
    # model architecture
    dim = 384
    n_layers = 6
    n_head = 6
    vocab_size = 5000
    max_seq_len = 256
    dropout = 0.2
    
    # training
    batch_size = 64
    learning_rate = 1e-3
    max_iters = 5000
    eval_interval = 100
    eval_iters = 200
    grad_clip = 1.0
    
    # optimizer
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.99
    
    # learning rate decay
    warmup_iters = 100
    lr_decay_iters = 5000
    min_lr = 1e-4
    
    # system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile = True
    
    # checkpointing
    out_dir = 'out'
    checkpoint_interval = 1000

# data preparation
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = text
        self.block_size = block_size
        self.stoi = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.itos = {i: ch for i, ch in enumerate(sorted(set(text)))}
        self.vocab_size = len(self.stoi)
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x_str = self.data[idx:idx+self.block_size]
        y_str = self.data[idx+1:idx+self.block_size+1]
        
        x = torch.tensor([self.stoi[ch] for ch in x_str], dtype=torch.long)
        y = torch.tensor([self.stoi[ch] for ch in y_str], dtype=torch.long)
        
        return x, y

def get_batch(batch_size, split='train'):
    data = train_dataset if split == 'train' else val_dataset
    indices = torch.randint(0, len(data), (batch_size,))
    x_batch = []
    y_batch = []
    for idx in indices:
        x, y = data[idx]
        x_batch.append(x)
        y_batch.append(y)
    return torch.stack(x_batch), torch.stack(y_batch)

# training functions
def save_checkpoint(model, optimizer, iter_num, config, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'config': config,
        'vocab_size': config.vocab_size,
        'stoi': train_dataset.stoi
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def get_lr(it):
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        split_losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(config.batch_size, split=split)
            x, y = x.to(config.device), y.to(config.device)
            with ctx:
                logits, _ = model(x, y)
                loss = model.last_loss
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean()
    model.train()
    return losses

# main training loop
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
ctx = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, dtype))

if __name__ == "__main__":
    config = TrainConfig()

    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
    response = requests.get(shakespeare_url)
    response.raise_for_status()  # check if download was successful
    text = response.text

    # split data: 90% train, 10% val
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = TextDataset(train_text, config.max_seq_len)
    val_dataset = TextDataset(val_text, config.max_seq_len)
    config.vocab_size = train_dataset.vocab_size  # same vocab for both

    print(f"Train dataset size: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Vocabulary size: {config.vocab_size}")
    
    model_args = GPTModelArgs(
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_head,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    model = GPT(model_args)
    model.to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=config.device
    )
    
    # compile model if enabled
    if config.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled for faster training")
    
    # training loop
    model.train()
    start_time = time.time()
    best_val_loss = float('inf')

    for iter_num in range(config.max_iters):
        # get batch
        x, y = get_batch(config.batch_size, split='train')
        x, y = x.to(config.device), y.to(config.device)
        
        # forward pass
        with ctx:
            logits, _ = model(x, y)
            loss = model.last_loss
        
        # backward pass
        scaler.scale(loss).backward()
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # gradient step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # print/log progress
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(f"Iter {iter_num:5d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Time: {elapsed:.2f}s")
            # save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint_path = f"{config.out_dir}/best_model.pt"
                save_checkpoint(model, optimizer, iter_num, config, checkpoint_path)
        
        # save periodic checkpoint
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            checkpoint_path = f"{config.out_dir}/ckpt_{iter_num}.pt"
            save_checkpoint(model, optimizer, iter_num, config, checkpoint_path)

    # final checkpoint
    final_path = f"{config.out_dir}/final_model.pt"
    save_checkpoint(model, optimizer, config.max_iters, config, final_path)
    print("Training completed!")