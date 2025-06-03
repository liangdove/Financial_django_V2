function calculateRiskScore() {
    const amount = parseFloat(document.getElementById('transactionAmount').value) || 0;
    const frequency = parseFloat(document.getElementById('transactionFrequency').value) || 0;
    const accountAge = parseFloat(document.getElementById('accountAge').value) || 0;
    const locationRisk = parseFloat(document.getElementById('locationRisk').value) || 0;

    if (amount <= 0 || frequency <= 0 || accountAge <= 0 || locationRisk <= 0) {
        document.getElementById('riskScoreResult').innerHTML = '<div class="alert alert-warning">请输入所有有效数据</div>';
        return;
    }

    // 简单风险评分公式：权重加权求和
    const riskScore = (amount * 0.4) + (frequency * 0.3) + ((6 - accountAge / 12) * 0.2) + (locationRisk * 0.1);

    let riskLevel, riskClass;
    if (riskScore > 80) {
        riskLevel = "高风险";
        riskClass = "danger";
    } else if (riskScore > 50) {
        riskLevel = "中等风险";
        riskClass = "warning";
    } else {
        riskLevel = "低风险";
        riskClass = "success";
    }

    document.getElementById('riskScoreResult').innerHTML = `
        <div class="alert alert-${riskClass}">
            <h5>${riskLevel}</h5>
            <p>风险评分：${riskScore.toFixed(2)}</p>
        </div>
    `;
}