// fraudData.js

const fraudData = {
    years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    fraudCases: [450, 520, 610, 720, 900, 1100, 1350, 1600],
    fraudAmounts: [1.5, 2.0, 2.7, 3.2, 4.0, 5.3, 6.8, 7.5, 9.3], // 单位：百万美元
    ageDistribution: [25, 35, 20, 10, 10], // 各年龄段受害者百分比
    ageGroups: ['18-25', '26-35', '36-45', '46-60', '60+']
};

// 验证数据
const totalPercentage = fraudData.ageDistribution.reduce((a, b) => a + b, 0);
if (totalPercentage !== 100) {
    console.error('Age distribution percentages do not sum to 100!');
}