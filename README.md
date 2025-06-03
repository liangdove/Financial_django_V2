## 大创原型系统
### 参考项目：
https://github.com/ZhouShichan/Financial-anomaly-detection    
https://github.com/Zhu-Shatong/DynamicSocialNetworkFraudDetection
https://github.com/DGraphXinye/2022_finvcup_baseline

### 项目算法：
- 我们这个项目研究主题是使用图神经网络研究社交网络的变化
- 具体来讲，我们选用了金融领域的社交数据集（[DGraph：一个金融领域的难样本分类数据集](https://dgraph.xinye.com/)），使用了图神经网络的经典模型（GCN、GraphSAGE）和新型的图神经网络模型（[GEARSage](https://github.com/Zhu-Shatong/DynamicSocialNetworkFraudDetection/tree/main)）,来完成图异常检测任务。
- 我们对3个模型在DGraph数据集上做了完整训练与验证，验证指标为准确率和AUC。（使用模型：http://10.29.16.132:8000/Fin/1/）。
- 实验不足：对单样本分类依旧困难，只有群体聚合的分类效果。

### 项目前端：
- 此外，为了**适合大创项目的展示需要**，我们使用Django开发框架完成了项目的前后端搭建，有了具体的演示效果（pc端访问端口：http://10.29.16.132:8000/Guard/guard）。
- 产品定位：一个简易的金融交易监控系统
- 实现功能：金融监控仪表盘、异常数据捕获、数分、风险评分、快速预测
- 只有“快速预测”模块实际调用了后端GNN模型，其余功能均为展示效果。

### 结题：
结题需要提交：
- 项目结题报告：https://kdocs.cn/l/co9BP1AVJ9rG
- 项目报告（计划）书：https://kdocs.cn/l/ckeAL7QTQTK6
- 答辩PPT（我写）
- 演示视频（我写）

需要在在线文档中标注自己完成了哪些内容（用自己名字的缩写标记“ljl”）,在线文档编辑不方便的话，可以标注好自己想完成哪一部分，然后将编辑好的本地文档发群里。
可以借助deepseek，结合国家战略需求写就行了~