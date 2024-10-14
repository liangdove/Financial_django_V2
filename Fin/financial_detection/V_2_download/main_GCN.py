from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from Fin.financial_detection.V_2_download.models.GCN import Model
from Fin.financial_detection.V_2_download.utils.dgraphfin import load_data, AdjacentNodesDataset
from Fin.financial_detection.V_2_download.utils.evaluator import Evaluator
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
# from utils.evaluator import Evaluator


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('C:\\E\\PycharmProject\\Financial_django\\Fin\\financial_detection\\datasets', 'DGraphFin', force_to_symmetric=True)
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
).to(device)
model_desc = f'GCN-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'Fin\\financial_detection\\results_fin\\model-{model_desc}.pt'
model.load_state_dict(torch.load(model_save_path, map_location=device))

cache_path = f'./results_fin/out-best-{model_desc}.pt'


def predict(data, node_id):
    if os.path.exists(cache_path):
        out = torch.load(cache_path, map_location=device)
    else:
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)

    pred = out[node_id].exp()
    return pred.squeeze(0)


def GCN_main(request):
    result_gcn = None  # 为 result_graphsage 提供默认值，防止 UnboundLocalError
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
                result_gcn = f'节点 {node_idx} 预测对应的标签为:{label_idx}, 为 {dic[label_idx]}, 欺诈置信度为 {score_rounded},原{y_pred}。'
            except ValueError:
                result_gcn = "Invalid node index provided."
        else:
            result_gcn = "No node index provided."

    # 渲染 HTML 模板，并将 result_graphsage 传递给前端
    return render(request, 'result_gcn.html', {'result_gcn': result_gcn})

