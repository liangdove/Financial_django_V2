from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  # Mixed precision for speedup

from models.GraphSAGE import GraphSAGE as Model
from utils.dgraphfin import load_data
from utils.evaluator import Evaluator

# Initialize mixed precision scaler for speeding up
scaler = GradScaler()

def train(model, data, train_idx, optimizer, cache_path, use_amp=True):
    model.train()
    optimizer.zero_grad()
    
    # Mixed precision training
    with autocast(enabled=use_amp):
        out = model(data.x, data.adj_t)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache_path)
        out = out[train_idx]
        loss = F.nll_loss(out, data.y[train_idx])

    # Backward pass with mixed precision scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def valid(model, data, split_idx, evaluator, cache_path, use_amp=True):
    if os.path.exists(cache_path):
        out = torch.load(cache_path)
    else:
        with torch.no_grad():
            model.eval()
            with autocast(enabled=use_amp):
                out = model(data.x, data.adj_t)
    y_pred = out.exp()
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])['auc']
    return eval_results, losses

def train_epoch(
    model, data, optimizer, evaluator, lr, min_valid_loss, epoch, model_desc, stop_count,
    use_early_stop=False, use_lr_scheduler=False, use_amp=True
):
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    cache_path = Path(f'./results/out-{model_desc}.pt')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    loss = train(model, data, data.train_mask, optimizer, cache_path, use_amp)
    eval_results, losses = valid(model, data, split_idx, evaluator, cache_path, use_amp)
    
    valid_loss = losses['valid']
    early_stop = False

    if valid_loss < min_valid_loss:
        stop_count = 0
        model_save_path = Path(f'results/model-{model_desc}.pt')
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        min_valid_loss = valid_loss
        if cache_path.exists():
            out = torch.load(cache_path)
            torch.save(out, cache_path.with_name(f'out-best-{model_desc}.pt'))
    else:
        stop_count += 1
        if stop_count == 5 and use_lr_scheduler:
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
        if stop_count == 10 and use_early_stop:
            early_stop = True

    train_log = {
        'epoch': epoch,
        't.loss': losses['train'],
        't.auc': eval_results['train'],
        'v.loss': losses['valid'],
        'v.auc': eval_results['valid'],
        'lr': lr,
        's.cnt': stop_count,
        'min.v.loss': min_valid_loss,
    }
    
    with open(f'results/train_log-{model_desc}.csv', 'a' if epoch > 0 else 'w', newline='') as f:
        pd.DataFrame({k: [v] for k, v in train_log.items()}).to_csv(f, header=f.tell() == 0, index=False)
    
    return min_valid_loss, lr, stop_count, early_stop, train_log

if __name__ == '__main__':
    # Set GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data and move to GPU
    data = load_data('./datasets/', 'DGraph', force_to_symmetric=True).to(device)

    lr = 0.005
    print(f'batch_size: all data, lr: {lr}')

    model_params = {
        "h_c": 16,
        "dropout": 0.0,
    }

    model = Model(
        in_c=17,
        out_c=2,
        **model_params
    ).to(device)

    model_desc = f'GraphSAGE-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    evaluator = Evaluator('auc')

    min_valid_loss = 1e10
    stop_count = 0

    epoch_iter = tqdm(range(0, 600))
    
    for epoch in epoch_iter:
        min_valid_loss, lr, stop_count, early_stop, train_log = train_epoch(
            model, data, optimizer, evaluator, lr, min_valid_loss, epoch, model_desc, stop_count, use_amp=True
        )
        epoch_iter.set_postfix(**train_log)
        if early_stop:
            break

    if early_stop:
        print(f'Early stop at epoch {epoch}')
    else:
        print(f'Training finished with {epoch} epochs')
