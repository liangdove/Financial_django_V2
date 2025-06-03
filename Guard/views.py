
# Create your views here.
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


def guard(request):
    return render(request, "guard.html")

def prevention(request):
    return render(request, "prevention.html")

from ollama import chat  # ollama 库并正确配置了模型

# 默认模型名称（可根据需要调整）
MODEL_NAME = 'qwen2.5:3B'
@csrf_exempt
def chat_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持 POST 请求'}, status=400)
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        if not user_message:
            return JsonResponse({'error': '消息为空'}, status=400)
        
        system_message = "你是一个友好耐心的助手，专门服务金融领域的客户。"
        
        # 从session中获取历史对话记录，若不存在则初始化
        history = request.session.get('chat_history', [])
        if not history:
            history.append({"role": "system", "content": system_message})
        
        # 添加用户消息
        history.append({"role": "user", "content": user_message})
        
        # 使用所有历史消息作为对话上下文
        stream = chat(
            model=MODEL_NAME,
            messages=history,
            stream=True
        )
        reply_chunks = []
        for chunk in stream:
            reply_chunks.append(chunk['message']['content'])
        reply = ''.join(reply_chunks)
        
        # 保存机器人回复到历史记录中
        history.append({"role": "assistant", "content": reply})
        request.session['chat_history'] = history
        
        return JsonResponse({'reply': reply})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

import random

@csrf_exempt
def broadcast_case(request):
    if request.method != 'GET':
        return JsonResponse({'error': '仅支持 GET 请求'}, status=400)
    try:
        # 根据实际路径调整文件路径
        with open(r"c:\E\Financial_django\static\samples.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return JsonResponse({'error': '数据为空'}, status=404)
        sample = random.choice(lines).strip()  # 随机选取一条
        parts = sample.split("-")
        response_data = {
            "date": parts[0] if len(parts) > 0 else "",
            "location": parts[1] if len(parts) > 1 else "",
            "victim": parts[2].replace("受害人：", "") if len(parts) > 2 else "",
            "amount": parts[3].replace("诈骗金额：", "") if len(parts) > 3 else "",
            "transaction_id": parts[4].replace("交易单号：", "") if len(parts) > 4 else ""
        }
        # 也可以返回完整信息
        response_data["full_message"] = sample
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
import os
from django.conf import settings

@csrf_exempt  # 或在urls中使用csrf_exempt装饰器（生产环境建议正确配置csrf）
def record_exception(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # 定义保存异常数据的 json 文件路径
            file_path = os.path.join(settings.BASE_DIR, 'exception_records.json')
            print(f"保存异常数据到: {file_path}")
            # 如果文件存在则读取原数据，不存在则初始化一个列表
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            else:
                records = []
            # 将新异常数据追加
            records.append(data)
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
            return JsonResponse({'status': 'success', 'data': data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': '仅支持POST请求'}, status=400)