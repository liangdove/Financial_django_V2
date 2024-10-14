from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.GraphSAGE import GraphSAGE as Model
from utils.dgraphfin import load_data, AdjacentNodesDataset
from utils.evaluator import Evaluator
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int, dropout: float = 0.05):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_c, h_c)
        self.conv2 = SAGEConv(h_c, out_c)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj, **kwargs):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=-1)

# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('C:\\E\\PycharmProject\\Financial_django\\Fin\\financial_detection\\datasets', 'DGraph', force_to_symmetric=True)
data = data.to(device)

model_params = {
    "h_c": 16,
    "dropout": 0.0,
}

model = GraphSAGE(
    in_c=17,
    out_c=2,
    ** model_params
)
model_desc = f'GraphSAGE-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'C:\\E\PycharmProject\\Financial_django\\Fin\\financial_detection\\results\\model-{model_desc}.pt' # fin
model.load_state_dict(torch.load(model_save_path, map_location=device))

cache_path = f'./results/out-best-{model_desc}.pt' # fin


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
    save_path = os.path.join(save_dir, "roc_curve_GraphSage.png")
    plt.savefig(save_path)
    
    plt.show()


def main():
    out = predict(data)
    preds = out[data.test_mask].cpu().numpy()
    y = data.y[data.test_mask]
    draw(y, preds[:,1])
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\preds', preds)
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\y', y)
    
    # fin train:27万  val：11万  test：19万
    # no  train:85万  val: 18万  test：18万
    
if __name__ == "__main__":
    main()


