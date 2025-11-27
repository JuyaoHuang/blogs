---
title: 'torchvision.transforms模块'
published: 2025-11-27
description: "PyTorch计算机视觉图像预处理与增强工具箱transforms的介绍"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['CV','Pytorch']
draft: false
---

# torchvision.transforms 概述

`torchvision.transforms` 是 PyTorch 官方计算机视觉库 `torchvision` 中的一个核心模块，专门用于图像的预处理和 数据增强

在深度学习训练流程中，原始图片通常不能直接输入模型，需要经过一系列变换：
1.  格式转换：将图片文件（PIL Image 或 NumPy 数组）转换为 PyTorch 的 Tensor
2.  尺寸统一：将不同大小的图片调整为模型固定的输入尺寸（如 224x224）
3.  数据增强：通过随机变换（翻转、裁剪、变色等）扩充数据集，防止模型过拟合，提高泛化能力
4.  标准化：对像素值进行归一化，加速模型收敛

------

## 1. 容器与组合

通常使用 `transforms.Compose` 将多个变换操作串联起来，形成一个处理流水线：

-   作用：将多个 transform 操作组合在一起，按顺序执行
-   参数：
    
    -   `transforms` (list of Transform objects): 包含要执行的变换列表
-   示例：
    ```python
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ```

------

## 2. 格式与数值转换

这些变换主要用于改变数据的数据类型或数值范围，通常放在 `Compose` 的**末尾**

**transforms.ToTensor**

-   作用：将 PIL Image 或 numpy.ndarray (H x W x C) 转换为 tensor (C x H x W)
-   数值变化：会**自动**将像素值从 [0, 255] 归一化到 **[0.0, 1.0]** 之间
-   注意：这是最基础的一步，大多数 CNN 模型都需要输入 Tensor
-   位置：  通常放在几何/颜色变换之后，Normalize 之前

**transforms.Normalize(mean, std)**

- 作用：用均值和标准差对 Tensor 进行标准化

- 公式：
  $$
  output[channel] = \frac{(input[channel] - mean[channel])}{std[channel]}
  $$
  

-   参数：
    -   `mean` (sequence):   各通道的均值
    -   `std` (sequence):   各通道的标准差
    
-   常用值：对于 ImageNet 预训练模型，通常使用：
    
    -   mean = `[0.485, 0.456, 0.406]`
    -   std = `[0.229, 0.224, 0.225]`
    
-   目的：使数据分布呈标准正态分布，有助于梯度下降算法更快收敛

------

## 3. 几何变换 

主要用于改变图像的空间结构，是数据增强的重要手段

**transforms.Resize(size, interpolation=...)**

-   作用：调整图像大小
-   参数：
    -   `size` (int or tuple):   输出尺寸。如果是一个整数 (e.g., 256)，则将较短边缩放到该数值，另一边按比例缩放；如果是 (h, w)，则强制缩放到该尺寸

**transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.33))** (训练时常用)

-   作用：先随机裁剪出图片的一部分，然后将其缩放到指定大小。这是训练 CNN 非常强大的增强手段，迫使模型学习物体的局部特征。
-   参数：
    -   `size`: 最终输出的大小
    -   `scale`: 随机裁剪面积的比例区间，默认 (0.08, 1.0)
    -   `ratio`: 裁剪区域的长宽比范围

**transforms.RandomHorizontalFlip(p=0.5)**

-   作用：以一定概率水平翻转图像（左右镜像）
-   参数：
    -   `p` (float): 翻转概率，默认为 0.5

**transforms.RandomRotation(degress)**

-   作用：随机旋转图像
-   参数：
    -   `degrees` (sequence or float/int):    旋转角度范围。如 `(-10, 10)` 或 `15` (表示 -15 到 15 度)

------

## 4. 颜色变换

用于改变图像的色彩属性，模拟不同的光照条件

**transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)**

-   作用：随机改变图像的亮度、对比度、饱和度和色调
-   参数：
    -   `brightness` (float):   亮度抖动范围
    -   `contrast` (float):   对比度抖动范围
    -   `saturation` (float):   饱和度抖动范围
    -   `hue` (float):   色调抖动范围

**transforms.Grayscale(num_output_channels=1)**

-   作用：将图像转换为灰度图
-   参数：
    -   `num_output_channels` (int): 输出通道数（1或3）

------

## 5. 综合示例：构建训练与验证 Transform

在实际项目中，训练集通常需要较强的数据增强，而验证/ 测试集只需要确定性的调整大小和归一化。

以下示例使用 **Food-11** 数据集：

```python
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义训练集的 Transform (包含数据增强) 
train_transform = transforms.Compose([
    # 几何变换：增强空间不变性
    transforms.RandomResizedCrop(224),       # 随机裁剪并缩放到 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.RandomRotation(15),           # 随机旋转 +/- 15度
    
    # 颜色变换：增强光照不变性
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    
    # 格式转换
    transforms.ToTensor(),                   # 转为Tensor, 归一化至[0,1]
    
    # 标准化 (使用ImageNet统计量)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 2. 定义验证/测试集的 Transform (仅基础处理) 
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 确定性地调整大小
    # 或者先Resize大一点再CenterCrop，效果可能更好：
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 可视化 Tensor 图片
def imshow(tensor_img, title=None):
    """
    将 Normalize 后的 Tensor 还原为可视化的 numpy 图片
    """
    # Clone 以免修改原数据
    img = tensor_img.clone().detach().cpu().numpy()
    
    # 维度变换: (C, H, W) -> (H, W, C)
    img = img.transpose((1, 2, 0))
    
    # 反标准化 (Un-Normalize): img = img * std + mean
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    
    # 截断数值到 [0, 1] 范围，防止因浮点误差导致的显示噪点
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off') # 不显示坐标轴
    plt.show()

# 使用示例
# 假设 img 是 PIL Image 对象
# transformed_img = train_transform(img)
# imshow(transformed_img, title="Augmented Image")
```

## 总结

*   **ToTensor()** 是必须的，它连接了 PIL/NumPy 和 PyTorch Tensor
*   **Normalize()** 是训练高性能模型的标准配置
*   **Compose()** 像胶水一样把各种变换粘合在一起
*   在实验中，**可视化**非常重要，但要注意 Normalize 后的图片不能直接画，需要先反归一化