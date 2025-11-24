---
title: 'nn.lossfunction模块'
author: Alen
published: 2025-10-27
description: "PyTorch神经网络损失函数模块nn.Module.lossfunctions的介绍"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['ML','DL']
draft: false
---

# 损失函数


优化器用于更新参数，使模型的损失值向最低值下降；而损失函数就是用于计算 Loss 的方法

**核心作用**：  损失函数（也称准则或目标函数）接收模型的**预测输出  prediction** 和真实的**目标标签  target**，然后计算出一个**标量值（损失值）**，这个值衡量了模型预测的 "不准确程度" 或 "误差大小"

- **损失值越大**，代表模型预测得越差
- **损失值越小**，代表模型预测得越接近真实目标

神经网络训练的整个过程，就是通过优化器不断调整模型参数，以**最小化损失函数计算出的这个损失值**

loss.backward() 正是基于这个最终的标量损失值来计算所有参数的梯度

------

## 常用的损失函数详解

PyTorch 在 torch.nn 模块中提供了多种预先实现好的损失函数；它们可以大致分为两类：回归损失和分类损失

### 1. 回归损失  Regression Losses：用于预测连续值

当目标是预测一个连续的数值（例如房价、温度、股票价格）时，则使用回归损失

#### MSE Loss

MSE Loss，即均方误差是回归任务中最常用的损失函数，也称为 L2 损失

- **作用**：计算模型预测值与真实目标值之间**差值的平方的平均值**

- **数学公式**：

$$
L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

- **torch方法**：`nn.MSELoss`

- **核心参数**：
  - reduction (str, optional):   指定如何聚合批次中样本的损失
  - 'mean' (默认):   返回所有样本损失的平均值
  - 'sum':   返回所有样本损失的总和
  - 'none':   返回每个样本的损失，不进行聚合
- **输入/输出形状**：
  - 输入 (Prediction):   (N, *)，* 表示任意数量的维度
  - 目标 (Target):   (N, *)，形状必须与输入完全相同
- **特点**：对误差进行平方，因此对**异常值（outliers）**非常敏感；一个大的误差会不成比例地增加总损失

#### 平均绝对误差MAE

平均绝对误差也称为 MAE 或 L1 损失

- **作用**：  计算模型预测值与真实目标值之间**差值的绝对值的平均值**

- **数学公式**：
  $$
  L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
  $$

- **torch方法**：  `nn.L1Loss`

- **核心参数**：

  - reduction (str, optional):   与 MSELoss 相同

- **输入/输出形状**：  与 MSELoss 相同

- **特点**：相比 MSE，L1 损失对异常值的**鲁棒性更好**，因为它不对误差进行平方

### 2。分类损失  Classification Losses：用于预测类别

当目标是从多个类别中预测一个（或多个）时，则使用分类损失

#### 交叉熵损失函数 nn.CrossEntropyLoss
nn.CrossEntropyLoss 是**多分类任务**中最常用、最重要的损失函数

- **作用**： 衡量的是模型输出的概率分布与真实的类别分布之间的  “距离”

- **注意**：

  nn.CrossEntropyLoss 在内部**组合了 nn.LogSoftmax 和 nn.NLLLoss (负对数似然损失)**

  这意味着：

  - **模型最后一层不应该有 nn.Softmax 激活函数**
  - 直接将未经激活的原始输出（称为 **logits**）喂给这个损失函数

- **核心参数**：

  - weight (Tensor, optional):   需要手动设置的权重张量，用于处理**类别不平衡**问题；如果你有C个类别，可以传入一个大小为C的张量，为每个类别赋予不同的权重。
  - reduction (str, optional):   指定如何聚合批次中样本的损失

- **输入/输出形状**

  - 输入 (Logits):   $(N, C)$，其中 $N$ 是批量大小，$C$ 是类别总数
  - 目标 (Target):   $(N)$，一个一维张量，每个值是 [0, C-1] 范围内的**整数类别索引**

#### 二元交叉熵损失函数 BCEWithLogitsLoss

nn.BCEWithLogitsLoss 是**二分类任务**或**多标签分类任务**的标准损失函数

- **作用**：计算二元交叉熵
- **注意**： 与 CrossEntropyLoss 类似，nn.BCEWithLogitsLoss 在内部**组合了 nn.Sigmoid 和 nn.BCELoss**；这样做比手动分开计算在数值上更稳定。
  - **模型最后一层不应该有 nn.Sigmoid 激活函数**
  - 直接将原始的 logit 输出喂给它
- **核心参数**：
  - pos_weight (Tensor, optional):   一个权重张量，用于控制正样本的权重，常用于处理二分类中的类别不平衡
  - reduction (str, optional):   指定如何聚合批次中样本的损失
- **输入/输出形状**：
  - 输入 (Logits):   (N, *)。对于标准的二分类，通常是 (N, 1) 或 (N)
  - 目标 (Target):   (N, *)，形状必须与输入相同，且目标值应为**浮点数**（例如 0.0 或 1.0）

------

## 示例代码

下面的代码演示了如何在回归和分类任务中使用这些损失函数

```python
import torch
import torch.nn as nn

# --- 1. 回归损失示例 (MSELoss 和 L1Loss) ---
print("--- Regression Loss Example ---")
# 假设模型输出和真实目标
predictions_reg = torch.randn(4, 1)  # 4个样本，每个预测1个值
targets_reg = torch.randn(4, 1)

# 实例化损失函数
mse_loss_fn = nn.MSELoss()
l1_loss_fn = nn.L1Loss()

# 计算损失
mse_loss = mse_loss_fn(predictions_reg, targets_reg)
l1_loss = l1_loss_fn(predictions_reg, targets_reg)

print(f"Predictions: {predictions_reg.squeeze().detach().numpy()}")
print(f"Targets:     {targets_reg.squeeze().detach().numpy()}")
print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"L1 Loss (MAE): {l1_loss.item():.4f}")


# --- 2. 多分类损失示例 (CrossEntropyLoss) ---
print("\n--- Multi-class Classification Loss Example ---")
# 假设批量大小为4，共有3个类别
batch_size = 4
num_classes = 3

# 模型输出的原始 logits (未经 Softmax)
logits_multi_class = torch.randn(batch_size, num_classes)
# 真实的类别索引 (必须是 LongTensor)
targets_multi_class = torch.tensor([0, 2, 1, 0]) 

# 实例化损失函数
cross_entropy_loss_fn = nn.CrossEntropyLoss()

# 计算损失
ce_loss = cross_entropy_loss_fn(logits_multi_class, targets_multi_class)

print(f"Model Logits (shape: {logits_multi_class.shape}):\n{logits_multi_class.detach().numpy()}")
print(f"Target Labels (shape: {targets_multi_class.shape}):\n{targets_multi_class.numpy()}")
print(f"CrossEntropy Loss: {ce_loss.item():.4f}")


# --- 3. 二分类损失示例 (BCEWithLogitsLoss) ---
print("\n--- Binary Classification Loss Example ---")
# 假设批量大小为4
# 模型输出的原始 logit (未经 Sigmoid)
logits_binary = torch.randn(4, 1)
# 真实的标签 (必须是 FloatTensor)
targets_binary = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

# 实例化损失函数
bce_loss_fn = nn.BCEWithLogitsLoss()

# 计算损失
bce_loss = bce_loss_fn(logits_binary, targets_binary)

print(f"Model Logits (shape: {logits_binary.shape}):\n{logits_binary.detach().numpy()}")
print(f"Target Labels (shape: {targets_binary.shape}):\n{targets_binary.numpy()}")
print(f"BCEWithLogits Loss: {bce_loss.item():.4f}")
  
```

---

## 总结

|         损失函数         |     问题类型      |       模型最后输出       |               目标标签                |
| :----------------------: | :---------------: | :----------------------: | :-----------------------------------: |
|      **nn.MSELoss**      |       回归        |       预测的连续值       |     真实的连续值，形状与预测相同      |
|      **nn.L1Loss**       |       回归        |       预测的连续值       |     真实的连续值，形状与预测相同      |
| **nn.CrossEntropyLoss**  |      多分类       | **Logits** (未经Softmax) |     **整数类别索引** (LongTensor)     |
| **nn.BCEWithLogitsLoss** | 二分类/多标签分类 | **Logits** (未经Sigmoid) | **浮点数** (0.0或1.0)，形状与预测相同 |
