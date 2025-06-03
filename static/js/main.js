// function checkID() {
//     const idInput = document.getElementById('idInput');
//     const idResult = document.getElementById('idResult');
//     if (!idInput || !idResult) return;

//     const id = idInput.value.trim();
//     if (!/^\d{6}$/.test(id)) {
//         idResult.innerHTML = '<div class="alert alert-warning mt-2">请输入有效的6位数字ID号</div>';
//         return;
//     }

//     // 简单规则引擎：假设ID以特定数字开头或满足某些条件时为高风险
//     const isRisk = id.startsWith('9') || parseInt(id) % 7 === 0;

//     idResult.innerHTML = `
//         <div class="alert alert-${isRisk ? 'danger' : 'success'} mt-2">
//             <h5>${isRisk ? '<i class="fas fa-exclamation-triangle"></i> 高风险' : '<i class="fas fa-check-circle"></i> 安全'}</h5>
//             <p>${isRisk ? '该ID号存在潜在风险，请进一步核实。' : '该ID号未发现异常。'}</p>
//         </div>
//     `;
// }
document.addEventListener('DOMContentLoaded', function() {
    
    const typeCtx = document.getElementById('fraudTypeChart').getContext('2d');
    const typeChart = new Chart(typeCtx, {
        type: 'pie',
        data: {
            labels: ['投资理财诈骗', '冒充客服诈骗', '网购退款诈骗', '冒充公检法诈骗', '刷单诈骗', '其他'],
            datasets: [{
                data: [35, 20, 15, 18, 10, 2],
                backgroundColor: [
                    '#dc3545', '#fd7e14', '#ffc107', 
                    '#20c997', '#0d6efd', '#6c757d'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                },
                title: {
                    display: true,
                    text: '诈骗类型占比(%)',
                    padding: {
                        bottom: 10
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(0,0,0,0.8)',
                    borderWidth: 1
                }
            }
        }
    });

    const trendCtx = document.getElementById('fraudTrendChart').getContext('2d');
    const trendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
            datasets: [{
                label: '诈骗案件数(起)',
                data: [120, 135, 180, 170, 190, 165],
                borderColor: 'var(--primary-color)',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.3,
                fill: true,
                pointBackgroundColor: 'var(--primary-color)',
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: '挽回损失(万元)',
                data: [85, 95, 120, 110, 130, 115],
                borderColor: 'var(--success-color)',
                backgroundColor: 'rgba(25, 135, 84, 0.1)',
                tension: 0.3,
                fill: true,
                pointBackgroundColor: 'var(--success-color)',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '近半年诈骗趋势',
                    padding: {
                        bottom: 10
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(0,0,0,0.8)',
                    borderWidth: 1
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0,0,0,0.05)',
                        borderDash: [2, 3]
                    }
                }
            }
        }
    });
    
    const mapChart = echarts.init(document.getElementById('fraudMap'));
    
    const mapData = [
        {name: '北京市', value: 85},
        {name: '天津市', value: 65},
        {name: '上海市', value: 95},
        {name: '重庆市', value: 75},
        {name: '河北省', value: 45},
        {name: '河南省', value: 55},
        {name: '云南省', value: 35},
        {name: '辽宁省', value: 40},
        {name: '黑龙江省', value: 30},
        {name: '湖南省', value: 50},
        {name: '安徽省', value: 40},
        {name: '山东省', value: 60},
        {name: '新疆维吾尔自治区', value: 25},
        {name: '江苏省', value: 70},
        {name: '浙江省', value: 80},
        {name: '江西省', value: 45},
        {name: '湖北省', value: 55},
        {name: '广西壮族自治区', value: 40},
        {name: '甘肃省', value: 30},
        {name: '山西省', value: 35},
        {name: '内蒙古自治区', value: 25},
        {name: '陕西省', value: 40},
        {name: '吉林省', value: 35},
        {name: '福建省', value: 50},
        {name: '贵州省', value: 30},
        {name: '广东省', value: 90},
        {name: '青海省', value: 20},
        {name: '西藏自治区', value: 15},
        {name: '四川省', value: 65},
        {name: '宁夏回族自治区', value: 25},
        {name: '海南省', value: 30},
        {name: '台湾省', value: 45},
        {name: '香港特别行政区', value: 55},
        {name: '澳门特别行政区', value: 20}
    ];

    const mapOption = {
        title: {
            text: '全国诈骗案件分布',
            left: 'center',
            textStyle: {
                    color: '#333',
                    fontSize: 16,
                    fontWeight: 600
            }
        },
        tooltip: {
            trigger: 'item',
            formatter: function(params) {
                const dataItem = params.data;
                const value = dataItem ? dataItem.value : 0;
                const name = params.name || (dataItem ? dataItem.name : '未知区域');
                return name + '：' + (value || 0) + '起案件';
            },
                backgroundColor: 'rgba(0,0,0,0.7)',
                borderColor: '#333',
                textStyle: {
                    color: '#fff'
                }
        },
        visualMap: {
            min: 0,
            max: 100,
            left: 'left',
            top: 'bottom',
            text: ['高', '低'],
            calculable: true,
            inRange: {
                color: ['#ccebff', '#007bff']
            },
            textStyle: {
                    color: '#333'
            }
        },
        series: [{
            name: '诈骗案件',
            type: 'map',
            map: 'china',
            roam: true,
            itemStyle: {
                areaColor: '#f0f2f5',
                borderColor: '#fff',
                borderWidth: 0.5
            },
            emphasis: {
                label: {
                    show: true,
                    color: '#333'
                },
                itemStyle: {
                    areaColor: '#a8d5ff',
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.2)'
                }
            },
            select: {
                    label: {
                        show: true,
                        color: '#fff'
                    },
                    itemStyle: {
                        areaColor: '#0056b3'
                    }
            },
            data: mapData,
            label: {
                show: false,
                color: '#333'
            }
        }]
    };

    fetch('https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json')
        .then(response => response.json())
        .then(geoJson => {
            echarts.registerMap('china', geoJson);
            mapChart.setOption(mapOption);
        })
        .catch(error => {
            console.error('加载地图数据失败:', error);
            const mapElement = document.getElementById('fraudMap');
            if (mapElement) {
                mapElement.innerHTML =
                    '<div class="alert alert-danger m-3">地图数据加载失败，请刷新重试或检查网络连接。</div>';
            }
        });

    window.addEventListener('resize', function() {
        if (this.resizeTimeout) {
            clearTimeout(this.resizeTimeout);
        }
        this.resizeTimeout = setTimeout(() => {
                typeChart.resize();
                trendChart.resize();
                mapChart.resize();
        }, 200);
    });

    

    const counterElements = document.querySelectorAll('.stat-card-value');
    counterElements.forEach(counter => {
            const textContent = counter.textContent || '';
            const match = textContent.match(/[¥]?([\d,]+)/);
            if (!match || !match[1]) return;

            const target = parseFloat(match[1].replace(/,/g, ''));
            const prefix = textContent.includes('¥') ? '¥' : '';
            let count = 0;
            const duration = 1500;
            const frameDuration = 1000 / 60;
            const totalFrames = Math.round(duration / frameDuration);
            const increment = target / totalFrames;

            function updateCount() {
                count += increment;
                if (count < target) {
                    counter.textContent = prefix + Math.floor(count).toLocaleString('en-US');
                    requestAnimationFrame(updateCount);
                } else {
                    counter.textContent = prefix + target.toLocaleString('en-US');
                }
            }

            if (!isNaN(target) && target > 0) {
                requestAnimationFrame(updateCount);
            } else {
                counter.textContent = prefix + target.toLocaleString('en-US');
            }
    });
});

function checkRisk() {
    const amountInput = document.getElementById('amountInput');
    const riskResult = document.getElementById('riskResult');
    if (!amountInput || !riskResult) return;

    const amount = parseFloat(amountInput.value);
    
    if (isNaN(amount) || amount <= 0) {
        riskResult.innerHTML = '<div class="alert alert-warning mt-2">请输入有效的金额</div>';
        return;
    }
    
    let riskLevel, riskMessage, riskClass;
    
    if (amount > 50000) {
        riskLevel = "高风险";
        riskMessage = "大额交易请务必确认对方身份，建议通过官方渠道核实";
        riskClass = "danger";
    } else if (amount > 10000) {
        riskLevel = "中等风险";
        riskMessage = "请确认交易目的，谨慎验证对方信息";
        riskClass = "warning";
    } else {
        riskLevel = "低风险";
        riskMessage = "风险较低，但仍需注意交易安全";
        riskClass = "success";
    }
    
    riskResult.innerHTML = `
        <div class="alert alert-${riskClass} alert-dismissible fade show mt-2" role="alert">
            <h5 class="alert-heading">${riskLevel}</h5>
            <p>${riskMessage}</p>
            <hr>
            <p class="mb-0">
                <button class="btn btn-sm btn-outline-${riskClass}" onclick="showRiskDetail()">查看详情</button>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </p>
        </div>
    `;
}

function showRiskDetail() {
    alert('风险详情：\n1. 陌生账户转账风险\n2. 非官方渠道交易风险\n3. 大额资金安全建议');
}

function scanPhone() {
    const phoneInput = document.getElementById('phoneScan');
    const result = document.getElementById('scanResult');
    if (!phoneInput || !result) return;

    const phone = phoneInput.value;
    if (!phone || !/^\d+$/.test(phone)) {
        result.innerHTML = '<div class="alert alert-warning mt-2">请输入有效的电话号码</div>';
        return;
    }

    const isRisk = Math.random() > 0.7;
    result.innerHTML = `
        <div class="alert alert-${isRisk ? 'danger' : 'success'} mt-2">
            <h5>${isRisk ? '<i class="fas fa-exclamation-triangle"></i> 高风险' : '<i class="fas fa-check-circle"></i> 安全'}</h5>
            <p>${isRisk ? '该号码已被标记为可疑号码，请谨慎对待。' : '该号码未发现异常记录。'}</p>
        </div>
    `;
}

function scanURL() {
    const urlInput = document.getElementById('urlScan');
    const result = document.getElementById('scanResult');
    if (!urlInput || !result) return;

    const url = urlInput.value;
    if (!url || !url.includes('.')) {
        result.innerHTML = '<div class="alert alert-warning mt-2">请输入有效的网址</div>';
        return;
    }

    const isRisk = Math.random() > 0.7;
    result.innerHTML = `
        <div class="alert alert-${isRisk ? 'danger' : 'success'} mt-2">
                <h5>${isRisk ? '<i class="fas fa-exclamation-triangle"></i> 高风险' : '<i class="fas fa-check-circle"></i> 安全'}</h5>
            <p>${isRisk ? '该网址可能存在安全风险，建议谨慎访问。' : '该网址未发现异常。'}</p>
        </div>
    `;
}

function scanAccount() {
    const accountInput = document.getElementById('accountScan');
    const result = document.getElementById('scanResult');
    if (!accountInput || !result) return;

    const account = accountInput.value;
    if (!account || !/^\d+$/.test(account)) {
        result.innerHTML = '<div class="alert alert-warning mt-2">请输入有效的银行账号</div>';
        return;
    }

    const isRisk = Math.random() > 0.7;
    result.innerHTML = `
        <div class="alert alert-${isRisk ? 'danger' : 'success'} mt-2">
                <h5>${isRisk ? '<i class="fas fa-exclamation-triangle"></i> 高风险' : '<i class="fas fa-check-circle"></i> 安全'}</h5>
            <p>${isRisk ? '该账号存在异常交易记录，请谨慎对待。' : '该账号未发现异常。'}</p>
        </div>
    `;
}