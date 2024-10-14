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
from django.shortcuts import render
from django.http import HttpResponse


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('Fin\\financial_detection\\datasets', 'DGraphFin', force_to_symmetric=True)
data = data.to(device)

model_params = {
    "h_c": 16,
    "dropout": 0.0,
}

model = Model(
    in_c=17,
    out_c=2,
    ** model_params
)
model_desc = f'GraphSAGE-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'Fin\\financial_detection\\results_fin\\model-{model_desc}.pt' # fin
model.load_state_dict(torch.load(model_save_path, map_location=device))

cache_path = f'./results_fin/out-best-{model_desc}.pt' # fin


def predict(data, node_id):
    if os.path.exists(cache_path):
        out = torch.load(cache_path, map_location=device)
    else:
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.adj_t)

    pred = out[node_id].exp()
    return pred.squeeze(0)


# def GraphSAGE_main(request):
#     result = None  # 初始化结果为空
#
#     if request.method == 'POST':
#         # 获取表单中用户输入的 node_idx
#         node_idx = int(request.POST.get('node_idx'))
#
#         # 模拟预测过程
#         dic = {0: "正常用户", 1: "欺诈用户"}
#         y_pred = predict(data, node_idx)
#         label_idx = torch.argmax(y_pred).item()
#         result_graphsage = f'节点 {node_idx} 预测对应的标签为: {label_idx}, 为 {dic[label_idx]}。'
#
#     # 无论是 GET 还是 POST 请求，都会渲染同一个模板
#     return render(request, 'input_and_result.html', {'result_graphsage': result_graphsage})


def GraphSAGE_main(request):
    result_graphsage = None  # 为 result_graphsage 提供默认值，防止 UnboundLocalError
    if request.method == 'POST':
        node_idx = request.POST.get('node_idx')  # 从 POST 请求中获取节点索引
        if node_idx is not None:
            try:
                node_idx = int(node_idx)  # 确保 node_idx 是整数
                # 调用预测函数并生成结果
                dic = {0: "正常用户", 1: "欺诈用户"}
                y_pred = predict(data, node_idx)
                score = y_pred[1].item() # 置信度
                score_rounded = round(score, 3) # 保存3位小数
                label_idx = torch.argmax(y_pred).item()
                result_graphsage = f'节点 {node_idx} 预测对应的标签为: {label_idx}, 为 {dic[label_idx]}, 欺诈置信度为{score_rounded},原{y_pred}。'
            except ValueError:
                result_graphsage = "Invalid node index provided."
        else:
            result_graphsage = "No node index provided."

    # 渲染 HTML 模板，并将 result_graphsage 传递给前端
    return render(request, 'result_graphsage.html', {'result_graphsage': result_graphsage})


