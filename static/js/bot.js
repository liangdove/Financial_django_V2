// 切换聊天窗口显示
document.getElementById('chatToggleBtn').addEventListener('click', function() {
    document.getElementById('chatBot').style.display = 'flex';
    this.style.display = 'none';
});
document.getElementById('chatCloseBtn').addEventListener('click', function() {
    document.getElementById('chatBot').style.display = 'none';
    document.getElementById('chatToggleBtn').style.display = 'block';
});

// 发送消息事件绑定
document.getElementById('chatSendBtn').addEventListener('click', sendChatMessage);
document.getElementById('chatInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
});

function sendChatMessage() {
    var chatInput = document.getElementById('chatInput');
    var message = chatInput.value.trim();
    if (!message) return;
    
    appendMessage('user', message);
    chatInput.value = '';
    
    // 调用后端 API（调用 ollama 接口）获取回复
    fetch('/Guard/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        if(data.reply) {
            appendMessage('bot', data.reply);
        } else {
            appendMessage('bot', '抱歉，未收到回复。');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage('bot', '请求失败，请稍后重试。');
    });
}

function appendMessage(sender, text) {
    var chatMessages = document.getElementById('chatMessages');
    var msgDiv = document.createElement('div');
    msgDiv.style.margin = '8px 0';
    msgDiv.style.padding = '8px';
    msgDiv.style.borderRadius = '4px';
    if (sender === 'user') {
        msgDiv.style.backgroundColor = '#e1f5fe';
        msgDiv.style.alignSelf = 'flex-end';
    } else {
        msgDiv.style.backgroundColor = '#f1f1f1';
        msgDiv.style.alignSelf = 'flex-start';
    }
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
