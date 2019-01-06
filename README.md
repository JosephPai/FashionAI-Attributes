# Fashion AI Attributes Recognition of Apparel

* 天池大数据竞赛——FashionAI全球挑战赛—服饰属性标签识别
* 每个任务单独训练 + 多任务联合训练融合
* 比赛对模型数目有限制，本仓库仅限学习
* 一些经验放在[博客](https://blog.csdn.net/JosephPai/article/details/85923454)，欢迎大家交流学习

# Getting Started 

代码均在Python3.6下运行，Python2暂未优化测试

## 环境

* Ubuntu16.04
* Keras==2.2
* tensorflow==1.11
* opencv-python==3.4

Python Packages可以通过如下方式快速安装

> git clone https://github.com/JosephPai/FashionAI-Attributes.git
>
> cd FashionAI-Attributes-master
>
> pip3 install -r requirement.txt

## 使用

* datasets ---------------------- 存放Annotations标记文件
* result ------------------------ 保存模型预测结果
* weights ----------------------- 保存模型训练所得权重
* config.py --------------------- 配置文件，包括数据集目录等
* dataset.py -------------------- 数据预处理
* single_task_predict.py -------- 单任务训练脚本
* single_task_train.py ---------- 单任务测试脚本
* multitask_predict.py ---------- 单任务训练脚本
* multitask_train.py ------------ 单任务测试脚本

## 思路分享

[阿里天池服装标签识别比赛新人赛练习经验心得](https://blog.csdn.net/JosephPai/article/details/85923454)

## TODO
- [ ] multitask train and predict
- [ ] imgaug库进行data augmentation