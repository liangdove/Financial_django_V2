from django.shortcuts import render
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from Fin.financial_detection.V_2_download.models.GraphSAGE import GraphSAGE as Model
from Fin.financial_detection.V_2_download.utils.dgraphfin import load_data, AdjacentNodesDataset
from Fin.financial_detection.V_2_download.utils.evaluator import Evaluator
from django.http import HttpResponse
from Fin.financial_detection.V_2_download import main_GCN, main_GraphSAGE, main_GEARSage
import json
import random
from django.http import JsonResponse
from django.shortcuts import render
import random

def index_show(request):
    return render(request, 'index_fraud.html')

def main_GCN_output(request):
    output = main_GCN.GCN_main(request)
    return HttpResponse(output)

def main_GraphSage_output(request):
    output = main_GraphSAGE.GraphSAGE_main(request)
    return HttpResponse(output)

def main_GEARSage_output(request):
    output = main_GEARSage.GEARSage_main(request)
    return HttpResponse(output)


def csv_view(request):
    # 读取CSV文件
    df = pd.read_csv('C:\\E\\PycharmProject\\Financial_django\\static\\show_node.csv')  # 替换为你的文件路径
    # 将数据转换为HTML表格
    table_html = df.to_html(classes='table table-striped', index=False)
    
    return render(request, 'graph.html', {'table_html': table_html})

def about_us(request):
    return render(request, 'about_us.html')


def graph_data(request):
    # 生成500个节点，初始红色节点数量较少
    nodes = [{'id': i, 'color': 'red' if i < 2 else 'blue'} for i in range(500)]
    edges = []

    # 创建随机边连接
    for i in range(500):
        for _ in range(random.randint(1, 2)):  # 每个节点连接1到3个其他节点
            target = random.randint(0, 499)
            if target != i:  # 防止自连接
                edges.append({'source': i, 'target': target})

    # 将数据传递给HTML模板
    data = {'nodes': nodes, 'edges': edges}
    return render(request, 'dgraph.html', {'data': data})


