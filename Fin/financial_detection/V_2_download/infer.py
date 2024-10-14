import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph


from utils.feat_func_gear import data_process
from models import GEARSage             # 换成路径
from utils import DGraphFin             # 换成路径
from utils.evaluator import Evaluator   # 换成路径
from utils.utils import prepare_folder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


def set_seed(seed):
    np.random.seed(seed)             # 设置NumPy的随机种子
    random.seed(seed)                # 设置Python内置random模块的随机种子
    torch.manual_seed(seed)          # 设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)     # 设置PyTorch在CUDA上的随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置随机种子


@torch.no_grad()  # 禁用梯度计算，节省内存和计算资源
def test(model, data):
    model.eval()  # 将模型设置为评估模式

    # 传递整个图通过模型
    out = model(
        data.x, data.edge_index, data.edge_attr, data.edge_direct,
    )

    y_pred = out.exp()  # 计算预测值的指数，用于将输出转换为概率
    return y_pred

def draw(y, preds):
    fpr, tpr, thresholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    
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
    # 保存图像
    save_path = os.path.join(save_dir, "roc_curve_train.png")
    plt.savefig(save_path)
    
    plt.show()
    print('finish_plt')


def main():

    # # 创建一个解析器对象，用于处理命令行参数
    # parser = argparse.ArgumentParser(
    #     description="GEARSage for DGraphFin Dataset")

    # # 添加不同的命令行参数
    # parser.add_argument("--dataset", type=str, default="DGraphFin")
    # parser.add_argument("--model", type=str, default="GEARSage")
    # parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--epochs", type=int, default=500)
    # parser.add_argument("--hiddens", type=int, default=96)
    # parser.add_argument("--layers", type=int, default=2)
    # parser.add_argument("--dropout", type=float, default=0.3)

    # # 解析命令行参数
    # args = parser.parse_args()
    # print("args:", args)

    # 判断是否有可用的CUDA设备，如果有，使用CUDA设备；否则使用CPU
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("device:", device)

    # 设置随机种子，以确保实验可重复
    set_seed(42)

    # 加载DGraphFin数据集
    dataset = DGraphFin(root="Fin\\financial_detection\\datasets", name="DGraph")
    nlabels = 2  # 设置标签数量
    data = dataset[0]  # 获取数据

    # 展示一下大小
    print('data.test_mask.shape:', data.test_mask.shape)
    print('data.train_mask.shape:', data.train_mask.shape)
    print('data.valid_mask.shape:', data.valid_mask.shape)
    print('data.y.shape:', data.y.shape)

    # 划分数据集为训练集、验证集和测试集
    split_idx = {
        "train": data.train_mask,
        "valid": data.valid_mask,
        "test": data.test_mask,
    }

    # 对数据进行预处理并转移到指定设备
    data = data_process(data).to(device)

    # 获取训练集索引，并根据标签分为正负样本
    train_idx = split_idx["train"].to(device)

    data.train_pos = train_idx[data.y[train_idx] == 1]
    data.train_neg = train_idx[data.y[train_idx] == 0]

    # 初始化模型
    model = GEARSage(
        in_channels=data.x.size(-1),
        hidden_channels=96,
        out_channels=nlabels,
        num_layers=2,
        dropout=0.3,
        activation="elu",
        bn=True,
    ).to(device)

    print(f"Model GEARSage initialized")

    # model_file = './model_files/{}/{}/model.pt'.format(  # 换成路径
    #     args.dataset, args.model)
    model_file = 'Fin\\financial_detection\\results\\model_648.pt'
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file, map_location=device))

    out = test(model, data)

    # evaluator = Evaluator('auc')
    # preds_train, preds_valid, preds_test = out[data.train_mask], out[data.valid_mask], out[data.test_mask]
    # y_train, y_valid, y_test = data.y[data.train_mask], data.y[data.valid_mask], data.y[data.test_mask]
    # train_auc = evaluator.eval(y_train, preds_train)['auc']
    # valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
    # test_auc = evaluator.eval(y_test, preds_test)['auc']
    # print('train_auc:', train_auc)
    # print('valid_auc:', valid_auc)
    # print('valid_auc:', test_auc)

    preds = out[data.train_mask].cpu().numpy()
    y = data.y[data.train_mask].cpu().numpy()
    draw(y, preds[:,1])
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\preds', preds)
    np.save('Fin\\financial_detection\\V_2_download\\csv_npy\\y', y)
    print('finish_poj')


if __name__ == "__main__":
    main()
