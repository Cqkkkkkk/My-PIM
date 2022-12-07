# 细粒度图像分类

本项目将基于CUB-200-2011数据集[1]，实践一些细粒度图像分类任务的算法与模型，包括ResNet[2], Swin-Transformer[3]以及PIM[4]，并在其此基础上进行了一些新的模型架构的修改和尝试。CUB-200-2011包含200种鸟的子类，共含11783张3x384x384的RGB彩色图片，其中训练集5994张图片，测试集5789张图片。图片的标签为图像类别标注，以及其他的有关bounding box的标注。本项目只关注图像类别标签，不使用bounding box作为额外输入。作为分类任务，其最终的指标为分类准确率。


## 环境搭建

创建环境：
```
conda env create -f environment.yml
```
## 模型
可用的模型包括：

- ResNet50 + Linear Layer
- Swin-Transformer + Linear Layer
- Swin-Transformer + PIM(Original 1-hop GCN [5])
- Swin-Transformer + PIM(2-hop GCN)
- Swin-Transformer + PIM(APPNP [6])
- Swin-Transformer + PIM(GPR-GNN [7])

## 实验结果

|      | Res50 | Swin-T | Swin-T + PIM | Swin-T + PIM(2-hop) | Swin-T + PIM(APPNP) | Swin-T + PIM(GPR) |
| ---- | ----- | ------ | ------------ | ------------ | ------------ | ------------ |
| Acc  | 81.36 | 90.31  | 91.74        | 91.77        | 91.81        | 91.95        |



## 训练
训练所有模型的指令均已包含在`train.sh`中，相应的配置文件在`configs`文件目录下。训练完毕后，最佳的模型将被存储。
```
bash train.sh
```

## Inference

模型的inference可以通过`infer.sh`中的指令实现，其中inference只实现了基础PIM，PIM-2Hop，PIM-APPNP以及PIM-GPR这几个准确率最好的模型。

```
bash infer.sh
```

---

[1] The Caltech-ucsd Birds-200-2011 Dataset. 

[2] Deep Residual Learning for Image Recognition, CVPR 2016

[3] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, CVPR 2021

[4] A Novel Plug-in Module for Fine-Grained Visual Classification, CVPR 2022

[5] Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017

[6] Predict then Propagate: Graph Neural Networks meet Personalized PageRank, ICLR 2019

[7] Adaptive Universal Generalized PageRank Graph Neural Network, ICLR 2021