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
import networkx as nx
# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('Fin\\financial_detection\\datasets', 'DGraph', force_to_symmetric=True)
data = data.to(device)

# 提取节点特征、标签和边
x = data['x']  # 17-dimensional node features
y = data['y']  # node labels (4 classes)
edge_index = data['edge_index']  # shape (4300999, 2)
# edge_type = data['edge_type']  # 11 types of edges
# edge_timestamp = data['edge_timestamp']  # timestamps for each edge
# train_mask = data['train_mask']  # training nodes mask
# valid_mask = data['valid_mask']  # validation nodes mask
# test_mask = data['test_mask']  # test nodes mask

# # 截取前 1000 个节点的特征和标签
# num_nodes_to_extract = 10
# x = x[:num_nodes_to_extract]  # 截取前 1000 个节点的特征
# y = y[:num_nodes_to_extract]  # 截取前 1000 个节点的标签
# train_mask = train_mask[:num_nodes_to_extract]
# valid_mask = valid_mask[:num_nodes_to_extract]
# test_mask = test_mask[:num_nodes_to_extract]
# # 获取与前 1000 个节点相关的边
# mask = (edge_index[:, 0] < num_nodes_to_extract) & (edge_index[:, 1] < num_nodes_to_extract)
# edge_index = edge_index[mask]  # 过滤出与前 1000 个节点相关的边
# print(edge_index.shape)

# # 获取与前 1000 个节点相关的边
# # mask = (edge_index[:, 0] < num_nodes_to_extract) & (edge_index[:, 1] < num_nodes_to_extract)
# # edge_index = edge_index[mask]  # 过滤出与前 1000 个节点相关的边
# # edge_type = edge_type[mask]  # 获取相关边的类型
# # edge_timestamp = edge_timestamp[mask]  # 获取相关边的时间戳

# # print("Features of first 1000 nodes:", x)
# # print("Labels of first 1000 nodes:", y)
# # print("train_mask:", train_mask)
# # print("valid_mask:", valid_mask)
# # print("test_mask:", test_mask)
# print("Edges between first 1000 nodes:", edge_index)

# # # 获取训练集中所有 Class 0 和 Class 1 的节点索引
# # train_class_0 = np.where((train_mask == True) & (y == 0))[0]
# # train_class_1 = np.where((train_mask == True) & (y == 1))[0]

# # # 获取验证集中所有 Class 0 和 Class 1 的节点索引
# # valid_class_0 = np.where((valid_mask == True) & (y == 0))[0]
# # valid_class_1 = np.where((valid_mask == True) & (y == 1))[0]

# # # 获取测试集中所有 Class 0 和 Class 1 的节点索引
# # test_class_0 = np.where((test_mask == True) & (y == 0))[0]
# # test_class_1 = np.where((test_mask == True) & (y == 1))[0]

# # # 输出这些节点的特征
# # train_class_0_features = x[train_class_0]
# # train_class_1_features = x[train_class_1]
# # # print("Train Class 0 Features:", train_class_0_features)
# # # print("Train Class 1 Features:", train_class_1_features)

# # # 获取训练集中 Class 0 节点的所有边
# # train_class_0_edges = edge_index[np.isin(edge_index[:, 0], train_class_0) | np.isin(edge_index[:, 1], train_class_0)]
# # # print("Edges for Train Class 0 Nodes:", train_class_0_edges)

# # # 获取 Class 0 节点相关边的类型和时间戳
# # train_class_0_edge_types = edge_type[np.isin(edge_index[:, 0], train_class_0) | np.isin(edge_index[:, 1], train_class_0)]
# # train_class_0_edge_timestamps = edge_timestamp[np.isin(edge_index[:, 0], train_class_0) | np.isin(edge_index[:, 1], train_class_0)]
# # # print("Edge Types for Train Class 0 Nodes:", train_class_0_edge_types)
# # # print("Edge Timestamps for Train Class 0 Nodes:", train_class_0_edge_timestamps)

# 使用 networkx 创建一个无向图
G = nx.Graph()

# 前1000个节点的标签和边
num_nodes_to_visualize = 3000
y_subset = y[:num_nodes_to_visualize]  # 获取前1000个节点的标签
edge_index_subset = edge_index[(edge_index[:, 0] < num_nodes_to_visualize) & (edge_index[:, 1] < num_nodes_to_visualize)]  # 相关边

# 添加节点到图中，并根据标签设置颜色
colors = []
for i in range(num_nodes_to_visualize):
    G.add_node(i)  # 确保只添加前1000个节点
    if y_subset[i] == 0:
        colors.append('blue')  # 正常用户
    elif y_subset[i] == 1:
        colors.append('red')  # 欺诈用户
    elif y_subset[i] == 2:
        colors.append('green')  # 背景用户
    elif y_subset[i] == 3:
        colors.append('yellow')  # 背景用户

# # 添加前1000个节点之间的边
# for edge in edge_index_subset:
#     if edge[0] < num_nodes_to_visualize and edge[1] < num_nodes_to_visualize:
#         G.add_edge(edge[0], edge[1])

# 绘制图
plt.figure(figsize=(12, 12))
pos = nx.random_layout(G, seed=42)  # spring_layout 用于节点布局
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=50, edge_color='gray', alpha=0.5)

# 显示图
plt.title("Visualization of first 1000 nodes and their edges")
plt.show()


