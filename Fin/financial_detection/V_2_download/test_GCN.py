from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.GCN import Model
from utils.dgraphfin import load_data, AdjacentNodesDataset
from utils.evaluator import Evaluator
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


class Model(torch.nn.Module):
    def __init__(
            self,
            in_c: int,
            h_c: int,
            out_c: int,
            n_layers: int = 2,
            dropout: float = 0.1,
            normalize: bool = True,
    ):
        super(Model, self).__init__()
        self.n_layers = n_layers

        self.convs = torch.nn.ModuleList([
            GCNConv(
                in_c if i == 0 else h_c,
                h_c if i != n_layers - 1 else out_c,
                normalize=normalize,
                cached=True
            )
            for i in range(n_layers)
        ])
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)
    

# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('C:\\E\\PycharmProject\\Financial_django\\Fin\\financial_detection\\datasets', 'DGraph', force_to_symmetric=True)
data = data.to(device)

model_params = {
    "h_c": 16,
    "n_layers": 2,
    "dropout": 0.1,
    "normalize": True,
}

model = Model(
    in_c=17,
    out_c=2,
    ** model_params
)
model_desc = f'GCN-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'C:\\E\PycharmProject\\Financial_django\\Fin\\financial_detection\\results\\model-{model_desc}.pt' 
model.load_state_dict(torch.load(model_save_path, map_location=device))

cache_path = f'C:\\E\PycharmProject\\Financial_django\\Fin\\financial_detection\\results\\out-best-{model_desc}.pt' # fin


def predict(data):
    if os.path.exists(cache_path):
        out = torch.load(cache_path, map_location=device)
    else:
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.adj_t)

    preds = out.exp()
    return preds

def draw(y, preds):
    fpr, tpr, thresholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    
     # 创建保存图片的文件夹
    save_dir = "Fin\\financial_detection\\results_plt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    # 保存图片到 plt 文件夹
    save_path = os.path.join(save_dir, "roc_curve_GCN.png")
    plt.savefig(save_path)
    
    plt.show()


def main():
    out = predict(data)
    preds = out[data.test_mask].cpu().detach().numpy()
    y = data.y[data.test_mask]
    
    draw(y, preds[:,1])
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\preds', preds)
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\y', y)
    
    # fin train:27万  val：11万  test：19万
    # no  train:85万  val: 18万  test：18万
    
if __name__ == "__main__":
    main()


