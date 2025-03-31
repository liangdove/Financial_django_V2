
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