from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.GEARSage import GEARSage
from utils.dgraphfin import load_data, AdjacentNodesDataset
from utils.evaluator import Evaluator
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from torch_geometric.nn import SAGEConv


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('Fin\\financial_detection\\datasets', 'DGraphFin', force_to_symmetric=True)
data = data.to(device)

model = GEARSage(
    in_channels=data.x.size(-1),
        hidden_channels=96,
        out_channels=2,
        num_layers=2,
        dropout=0.3,
        activation="elu",
        bn=True,
)
# model_desc = f'GraphSAGE-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'Fin\\financial_detection\\results_fin\\GEARSage_model.pt' # fin
model.load_state_dict(torch.load(model_save_path, map_location=device))

# cache_path = f'./results_fin/out-best-{model_desc}.pt' # fin


def predict(data):

    with torch.no_grad():
        model.eval()
        out = model(data.x, data.adj_t)

    preds = out
    return preds


def main():
    out = predict(data)
    preds = out[data.valid_mask].cpu().numpy()
    y = data.y[data.valid_mask]
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\preds', preds)
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\y', y)
    
    # fin train:27万  val：11万  test：19万
    # no  train:85万  val: 18万  test：18万
    
if __name__ == "__main__":
    main()


