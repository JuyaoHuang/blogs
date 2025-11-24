---
title: 'torch.optim模块'
author: Alen
published: 2025-10-27
description: "PyTorch优化器模块torch.optim的介绍"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['ML','DL']
draft: false
---

# torch.optim 模块概述

torch.optim 是一个实现了各种优化算法的模块，它的核心作用是**根据计算出的梯度来更新神经网络的参数（权重和偏置）**，从而最小化损失函数

在 PyTorch 的训练流程中，它扮演着承上启下的关键角色

## 核心工作流程

在一个典型的训练步骤中，优化器的使用遵循一个固定的三步模式：

1. **optimizer.zero_grad()**：**清除旧的梯度**

   PyTorch 的 .backward() 方法会累加梯度

   如果不清除，当前 batch 计算出的梯度会和之前所有 batch 的梯度累加在一起，这会导致错误的参数更新方向。因此，在每次计算新梯度之前，必须将旧的梯度清零

2. **loss.backward()**：**计算新的梯度**

   这一步会根据损失函数 loss，通过反向传播算法计算出模型中所有可学习参数（requires_grad=True 的张量）的梯度。计算出的梯度会保存在每个参数的 .grad 属性中

3. **optimizer.step()**：**更新模型参数**

   这是优化器发挥作用的核心步骤

   optimizer.step() 会访问它所管理的所有参数，并根据各自的 .grad 属性以及优化器自身的更新规则（例如，学习率、动量等）来更新参数的值

------

## 封装的主要优化算法 

torch.optim 模块封装了多种优化算法，从最基础的到最先进的都有

下面详细介绍几个最常用、最经典的优化器

### 1. torch.optim.SGD - 随机梯度下降法

这是最基础的优化算法，它沿着梯度的负方向更新参数。

- **作用**：  实现随机梯度下降（或小批量梯度下降）及其动量变体
- **核心参数**：
  - params (iterable):   需要优化的模型参数，通常传入 model.parameters()
  - lr (float):     **学习率 (Learning Rate)**，这是最重要的超参数，控制每次参数更新的步长
  - momentum (float, optional):     动量因子 (默认为0)，引入动量可以帮助加速 SGD 在相关方向上的收敛并抑制振荡
  - weight_decay (float, optional):     权重衰减 (L2 正则化)，用于防止过拟合
  - dampening (float, optional):     动量的阻尼 (默认为0)
  - nesterov (bool, optional):     是否使用 Nesterov 动量 (默认为 False)

### 2. torch.optim.Adagrad - 自适应梯度算法

一个自适应学习率的优化算法，它对不频繁的参数进行较大更新，对频繁的参数进行较小更新。

- **作用**：  为每个参数维护一个独立的学习率
- **核心参数**：
  - params, lr:     同上
  - lr_decay (float, optional):     学习率衰减
  - weight_decay (float, optional):     权重衰减
- **特点**：  适合处理**稀疏数据**，但其学习率会单调递减，可能导致训练后期学习过早停止

### 3. torch.optim.RMSprop - 均方根传播

也是一种自适应学习率算法，是 Adagrad 的一个改进，旨在解决其学习率急剧下降的问题

- **作用**：  通过使用梯度平方的指数加权移动平均来调整学习率
- **核心参数**：
  - params, lr, weight_decay:     同上
  - alpha (float, optional):     平滑常数 (默认为0.99)，即指数加权平均的因子
  - momentum (float, optional):     动量因子 (默认为0)

### 4. torch.optim.Adam - 自适应矩估计

目前最流行、最常用的优化器之一，它结合了 Momentum (一阶矩估计) 和 RMSProp (二阶矩估计) 的思想。

- **作用**：    为每个参数计算自适应的学习率，通常在各种任务中都能取得不错的、鲁棒的性能。

- **核心参数**：

  - params, lr:     同上，推荐 0.01到0.001

  - betas (Tuple[float, float], optional): 

    用于计算梯度及其平方的运行平均值的系数 (默认为 (0.9, 0.999))。

    $ \beta_1$ 是动量项的衰减率，$\beta_2$是梯度平方的衰减率

  - eps (float, optional):     为了增加数值稳定性而加到分母里的小项 (默认为 1e-8)。

  - weight_decay (float, optional):     权重衰减。

### 5. torch.optim.LBFGS - 有限内存的 Broyden-Fletcher-Goldfarb-Shanno 算法

一种拟牛顿法，属于二阶优化方法，它通过近似 Hessian 矩阵来进行更大、更有效的更新。

- **作用**：  适用于可以一次性处理整个数据集（全批量）的场景。
- **核心参数**：
  - params, lr:   同上。lr 通常设为1，因为步长由线搜索决定。
  - max_iter (int):   每次优化的最大迭代次数 (默认为20)
  - history_size (int):   存储历史信息的数量，用于近似Hessian矩阵
- **使用方式非常特殊**：  它的 step 方法需要传入一个**闭包 (closure)** 函数；这个闭包函数需要重新评估模型、计算损失并返回

---

## 学习率快速参考表

Momentum = 0.9

|        优化器         |  常用初始学习率  |                        关键特性与说明                        |
| :-------------------: | :--------------: | :----------------------------------------------------------: |
| **SGD with Momentum** | 0.1, 0.01, 0.001 | **非常敏感**。0.1 适用于简单或浅层网络，0.01 是更安全、更通用的起点。通常**必须**配合学习率衰减 (LR Decay) |
|      **AdaGrad**      |       0.01       | **自适应学习率**。对初始学习率不那么敏感。主要问题是学习率会随训练单调递减，可能过早停止学习 |
|      **RMSProp**      |      0.001       | **自适应学习率**。是 AdaGrad 的改进版，缓解了学习率过早衰减的问题。这是一个非常稳健的默认值 |
|  **Adam (作为对比)**  |      0.001       | **自适应+动量**。结合了 RMSProp 和 Momentum 的优点。0.001 是 PyTorch/TensorFlow 的默认值，也是绝大多数情况下的最佳起点 |

------

------

## 示例代码

下面的示例将演示如何在一个简单的线性回归问题上使用这些不同的优化器。

### 1. 准备工作：模型、数据和损失函数

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 0. 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. 创建数据集
X_numpy = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_numpy = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

X = torch.from_numpy(X_numpy)
y = torch.from_numpy(y_numpy)

# 2. 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)

# 3. 定义损失函数
criterion = nn.MSELoss()

# 4. 训练函数 (通用，适用于一阶优化器)
def train_with_optimizer(optimizer, model, num_epochs=100):
    print(f"\n--- Training with {optimizer.__class__.__name__} ---")
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 反向传播和优化 (三步曲)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 打印最终学习到的参数
    [w, b] = model.parameters()
    print(f"Learned parameters: w={w.item():.3f}, b={b.item():.3f}")
  
```

### 2. 使用不同的一阶优化器

```python
# --- 使用 SGD ---
# 1. 实例化设计的模型/网络结构
# 2. chuan't
model_sgd = LinearRegressionModel()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
train_with_optimizer(optimizer_sgd, model_sgd)

# --- 使用 SGD with Momentum ---
model_momentum = LinearRegressionModel()
optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)
train_with_optimizer(optimizer_momentum, model_momentum)

# --- 使用 RMSprop ---
model_rmsprop = LinearRegressionModel()
optimizer_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=0.01)
train_with_optimizer(optimizer_rmsprop, model_rmsprop)

# --- 使用 Adam ---
model_adam = LinearRegressionModel()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001) # Adam通常可以用稍小的学习率
train_with_optimizer(optimizer_adam, model_adam)
  
```

### 3. 使用 L-BFGS (特殊用法)

```python
# --- 使用 L-BFGS ---
model_lbfgs = LinearRegressionModel()
# L-BFGS 的学习率通常是1，步长由线搜索决定
optimizer_lbfgs = optim.LBFGS(model_lbfgs.parameters(), lr=1, max_iter=20)

print(f"\n--- Training with {optimizer_lbfgs.__class__.__name__} ---")

# L-BFGS 的 step 需要一个闭包函数
def closure():
    # 清除梯度
    optimizer_lbfgs.zero_grad()
    # 前向传播
    outputs = model_lbfgs(X)
    # 计算损失
    loss = criterion(outputs, y)
    # 反向传播
    loss.backward()
    return loss

# 训练步骤
# 对于L-BFGS，通常迭代次数不多，因为每一步都很强
for i in range(5): 
    print(f'Step [{i+1}/5]')
    # 调用 step，并传入闭包
    # closure 函数会被多次调用
    optimizer_lbfgs.step(closure)
    
    # 打印损失 (需要重新计算，因为step后参数已更新)
    with torch.no_grad():
        final_loss = criterion(model_lbfgs(X), y)
        print(f'Loss: {final_loss.item():.4f}')

[w, b] = model_lbfgs.parameters()
print(f"Learned parameters: w={w.item():.3f}, b={b.item():.3f}")
  
```

Adam 通常能非常快地收敛到一个好的解

L-BFGS 步数虽少，但每一步的计算成本更高，在这个简单问题上也能迅速找到最优解
