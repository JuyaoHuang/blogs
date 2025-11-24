---
title: 'nn.Layers模块'
author: Alen
published: 2025-10-27
description: "PyTorch神经网络网络层构建工具箱nn.layer的介绍"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['ML','DL']
draft: false
---


# torch.nn.Layers 概述

torch.nn 模块中的各种网络层 (Layers) 是构建神经网络的核心组件，它们都是 nn.Module 的子类，负责执行特定的数学运算

nn.Layers 可以看作是预先定义好的、带有可学习参数的函数，每个层接收一个或多个输入张量  (Tensor)，并产生一个或多个输出张量。它们是构建复杂神经网络的工具组件

------

## 1. 卷积层  Convolutional Layers

主要用于处理具有网格结构的数据，如图像（2D网格）和时间序列（1D网格）

**nn.Conv2d**

- **作用**：  对2D输入（如图像）执行2D卷积操作，通过可学习的滤波器（或称为卷积核）来提取输入的局部特征
- **核心参数**：
  - in_channels (int):   输入图像的通道数（例如，灰度图为1，RGB彩色图为3）
  - out_channels (int):   输出的通道数，也即卷积核的数量。这个值决定了提取特征的种类数量
  - kernel_size (int or tuple):   卷积核的大小。例如 3 或 (3, 5)
  - stride (int or tuple, optional):   卷积核滑动的步长，默认为1
  - padding (int or tuple, optional):   在输入数据的边缘填充0的数量，默认为0，padding可以帮助控制输出特征图的空间尺寸
  - bias (bool, optional):   是否添加一个可学习的偏置，默认为 True
- **输入/输出形状**：
  - 输入: 
    $$
    (N,\ C_{in}\ ,\ H_{in}\ ,\ W_{in})
    $$

  - 输出: 
  $$
  (N,\ C_{out}\ ,\ H_{out}\ ,\ W_{out})
  $$
    其中 $H_{out}$ 和 $W_{out}$ 的计算公式为：
  $$
  outputSize = floor(\frac{(inputSize + 2 * padding - kernelSize)}{stride} + 1)
  $$
  -   其中 $N$ 是批量大小  batch size


------

## 2. 池化层  Pooling Layers

通常跟在卷积层之后，用于对特征图进行下采样（降维），以减少计算量、减小过拟合风险，并使特征具有一定的旋转和平移不变性

**nn.MaxPool2d**

- **作用**：  对2D输入执行最大池化操作，在一个窗口（由 kernel_size 定义）内，只取最大值作为输出
- **核心参数**：
  - kernel_size (int or tuple):   池化窗口的大小
  - stride (int or tuple, optional):   窗口滑动的步长。默认值为 kernel_size，这意味着窗口之间通常不重叠
  - padding (int or tuple, optional):   边缘填充，默认为0
- **输入/输出形状**：  与卷积层类似，输出尺寸由 kernel_size 和 stride 决定

------

## 3. 线性层  Linear Layers

也称为全连接层  Fully Connected Layers，用于对输入进行线性变换

**nn.Linear**

- **作用**：  
    执行 $y = x*A^T + b$ 的线性变换，在CNN中，通常用于将卷积和池化层提取的特征映射到最终的分类分数上
    
- **核心参数**：
  - in_features (int):   每个输入样本的特征维度
  - out_features (int):   每个输出样本的特征维度
  - bias (bool, optional):   是否添加一个可学习的偏置，默认为 True
- **输入/输出形状**：
  - 输入:   $(N\ ,\ *\ ,\ H_{in})$，其中 * 表示任意数量的附加维度，$H_{in}$ 必须等于 in_features
  - 输出:   $(N\ ,\ *\ ,\ H_{out})$，其中 $H_{out}$ 等于 out_features

------

## 4. 激活函数层  Activation Function Layers

为神经网络引入非线性，使其能够学习和表示比线性模型复杂得多的函数

### nn.ReLU

- **作用**：  修正线性单元  Rectified Linear Unit，逐元素地应用函数 f(x) = max(0, x)
- **参数**：
  - inplace (bool, optional):   如果设置为 True，会原地修改输入，节省内存，但可能会覆盖输入；默认为 False
- **特点**：  计算简单，能有效缓解梯度消失问题，是目前最常用的激活函数

### nn.Sigmoid 和 nn.Tanh

- **nn.Sigmoid**：  将输入压缩到 (0, 1) 区间，常用于二分类问题的输出层或表示概率
- **nn.Tanh**：  将输入压缩到 (-1, 1) 区间，通常在循环神经网络（RNN）中比 Sigmoid 表现更好

### nn.Softmax

- **作用**：  将一个N维实数向量转换为一个表示概率分布的N维实数向量。通常用于多分类问题的输出层
- **核心参数**：
  - dim (int):   指定应用 Softmax 的维度，对于批处理的输出 (N, C)，通常设置为 dim=1，以在类别维度 C 上计算概率

------

## 5. 归一化层  Normalization Layers

用于稳定和加速神经网络的训练

**nn.BatchNorm2d**

- **作用**：  对2D输入的通道维度 channel dimension进行批归一化；它通过重新中心化和缩放，使得每个通道的输出均值接近0，方差接近1
- **核心参数**：
  - num_features (int):   输入张量的通道数 C（来自 (N, C, H, W)）
- **特点**：  能显著加速收敛，允许使用更高的学习率，并具有一定的正则化效果。
- **注意**：  BatchNorm 在训练和评估模式下的行为不同，因此必须使用 model.train() 和 model.eval() 来切换

------

## 6. 循环层  Recurrent Layers

用于处理序列数据，如文本、语音和时间序列

**nn.LSTM 和 nn.GRU**

- **作用**：  实现长短期记忆网络 (LSTM) 或门控循环单元 (GRU)；它们是RNN的变体，通过引入门控机制来解决标准RNN中的梯度消失/爆炸问题，从而能更好地学习序列中的长期依赖关系
- **核心参数**：
  - input_size:   输入序列中每个时间步的特征维度
  - hidden_size:   隐藏状态的特征维度
  - num_layers:   循环层的层数（堆叠的RNN数量）
  - batch_first:   如果为 True，则输入和输出张量的维度格式为 (batch, seq, feature)，否则为 (seq, batch, feature)。强烈推荐设置为 True，更符合习惯

------

## 7. Dropout 层

一种正则化技术，用于防止神经网络过拟合

**nn.Dropout**

- **作用**：  在训练期间，以一定的概率 p 将输入张量中的部分元素随机置为零。这可以防止神经元之间产生过强的协同适应关系。
- **核心参数**：
  - p (float):   元素被置零的概率，默认为0.5
- **注意**：  与 BatchNorm 类似，Dropout 在训练和评估模式下行为不同。在 model.eval() 模式下，Dropout 层会自动失效

------

## 8. 辅助层  Utility Layers

**nn.Flatten**

- **作用**：  将一个连续范围的维度展平为一个张量。在CNN中，常用于将卷积层的输出特征图 (N, C, H, W) 展平为 (N, C*H*W) 的向量，以便送入线性层
- **核心参数**：
  - start_dim (int), end_dim (int):   指定要展平的维度范围

## 综合示例：构建一个简单的CNN模型

这个例子将把上面介绍的多种层组合起来，构建一个用于图像分类的简单卷积神经网络 (LeNet-5 风格)

```python
import torch
import torch.nn as nn

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 特征提取部分 (卷积 + 池化)
        self.features = nn.Sequential(
            # 输入: (B, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # -> (B, 6, 28, 28)
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (B, 6, 14, 14)
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # -> (B, 16, 10, 10)
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 16, 5, 5)
        )
        
        # 辅助层：展平
        self.flatten = nn.Flatten()
        
        # 分类部分 (全连接层)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Dropout(p=0.5), # 在全连接层之间使用Dropout
            
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            
            nn.Linear(in_features=84, out_features=num_classes)
            # 注意：在计算 CrossEntropyLoss 时，不需要在这里加 Softmax
        )

    def forward(self, x):
        # 数据流
        x = self.features(x)
        x = self.flatten(x) # 展平操作
        logits = self.classifier(x)
        return logits

# 实例化模型
model = SimpleCNN()
print("--- Model Architecture ---")
print(model)

# 创建一个假的输入张量来测试模型
batch_size = 4
# 假设输入是 28x28 的灰度图 (如 MNIST)
dummy_input = torch.randn(batch_size, 1, 28, 28)

# --- 演示 train() 和 eval() 的区别 ---
print("\n--- Model Mode Demonstration ---")
# 1. 训练模式
model.train()
print(f"Is model in training mode? {model.training}")
# 在训练模式下，BatchNorm 和 Dropout 会起作用
train_output = model(dummy_input)
print(f"Output shape in train mode: {train_output.shape}")

# 2. 评估模式
model.eval()
print(f"Is model in training mode? {model.training}")
# 在评估模式下，BatchNorm 使用学习到的统计量，Dropout 失效
eval_output = model(dummy_input)
print(f"Output shape in eval mode: {eval_output.shape}")
  
```

这个例子清晰地展示了如何像搭积木一样，将不同的 nn.Layer 组合在一起，定义一个完整的神经网络模型

理解这些基础层的输入、输出和作用，是进行任何深度学习模型设计的基础