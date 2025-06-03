# import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from Fin.financial_detection.V_2_download.utils.feat_func import data_process
from Fin.financial_detection.V_2_download.utils.dgraphfin import DGraphFin             # 换成路径
from Fin.financial_detection.V_2_download.utils.utils import prepare_folder
from Fin.financial_detection.V_2_download.models.GEARSage import GEARSage
from Fin.financial_detection.V_2_download.utils.dgraphfin import load_data, AdjacentNodesDataset
from Fin.financial_detection.V_2_download.utils.evaluator import Evaluator
from django.http import HttpResponse
from django.shortcuts import render


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
# print("device:", device)

# 设置随机种子，以确保实验可重复
set_seed(42)

# 加载DGraphFin数据集
dataset = DGraphFin(root="Fin\\financial_detection\\datasets", name="DGraphFin") # fin
nlabels = 2  # 设置标签数量
data = dataset[0]  # 获取数据

# 展示一下大小
# print('data.test_mask.shape:', data.test_mask.shape)
# print('data.train_mask.shape:', data.train_mask.shape)
# print('data.valid_mask.shape:', data.valid_mask.shape)
# print('data.y.shape:', data.y.shape)

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
    out_channels=2,
    num_layers=2,
    dropout=0.3,
    activation="elu",
    bn=True,
).to(device)

# print(f"Model {args.model} initialized")

model_file = 'Fin\\financial_detection\\results_fin\\GEARSage_model.pt'
# print('model_file:', model_file)
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))


def predict(data, node_id):
    
    with torch.no_grad():
        model.eval()
        out = test(model, data)

    pred = out[node_id]
    return pred.squeeze(0)

def GEARSage_main(request):
    result_gear = None  # 为 result_graphsage 提供默认值，防止 UnboundLocalError
    if request.method == 'POST':
        node_idx = request.POST.get('node_idx')  # 从 POST 请求中获取节点索引
        if node_idx is not None:
            try:
                node_idx = int(node_idx)  # 确保 node_idx 是整数
                # 调用预测函数并生成结果
                dic = {0: "正常用户", 1: "欺诈用户"}
                y_pred = predict(data, node_idx)
                label_idx = torch.argmax(y_pred).item() # 预测标签
                score = y_pred[1].item() # 置信度
                score_rounded = round(score, 3) # 保存3位小数
                result_gear = f'节点 {node_idx} 预测对应的标签为:{label_idx}, 为 {dic[label_idx]}, 欺诈置信度为 {score_rounded}, 原{y_pred}。'
            except ValueError:
                result_gear = "Invalid node index provided."
        else:
            result_gear = "No node index provided."

    # 渲染 HTML 模板，并将 result_graphsage 传递给前端
    return render(request, 'result_GEARSage.html', {'result_gear': result_gear})

from django.http import JsonResponse


def GEARSage_main_guard_tool_call(request):
    result_gear = None  # 默认值
    if request.method == 'POST':
        node_idx = request.POST.get('node_idx')  # 从 POST 请求中获取节点索引
        if node_idx is not None:
            try:
                node_idx = int(node_idx)
                dic = {0: "正常用户", 1: "欺诈用户"}
                y_pred = predict(data, node_idx)
                label_idx = torch.argmax(y_pred).item()  # 预测标签
                score = y_pred[1].item()  # 置信度
                score_rounded = round(score, 3)  # 保留3位小数
                result_gear = f'节点 {node_idx} 预测对应的标签为: {label_idx}\n\n 为 {dic[label_idx]}\n\n 欺诈置信度为 {score_rounded}\n\n 原 {y_pred}'
            except ValueError:
                result_gear = "Invalid node index provided."
        else:
            result_gear = "No node index provided."
    return JsonResponse({'result_gear': result_gear})
        
