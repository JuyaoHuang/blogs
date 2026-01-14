---
title: 'torch.optim.lr_scheduler模块'
publishDate: 2025-12-01
description: "PyTorch自适应学习率模块torch.optim.lr_scheduler的介绍"
tags: ['ML','DL']
language: 'Chinese'
first_level_category: "人工智能"
second_level_category: "深度学习框架"
draft: false
---

在 PyTorch 中，`torch.optim.lr_scheduler` 模块提供了多种动态调整学习率的方法


## 1. 目的

使用 Scheduler 的**核心目的**是解决固定学习率带来的训练瓶颈：

- 加速收敛：在训练初期（Loss 较大时），使用较大的学习率可以快速接近最优解
- 精细搜索：在训练后期（接近最优解时），减小学习率可以避免震荡，帮助模型下降到 Loss 的最低点
- **逃离局部最优**：某些特殊的 Scheduler（如 CosineAnnealing）通过周期性调整学习率，甚至有机会帮助模型跳出局部最优陷阱

简而言之：**让模型在该快的时候快，该慢的时候慢**

## 2. 参数 

不同的 Scheduler 类有不同的参数，但它们都必须接收一个核心参数：Optimizer，优化器

以下是几种最常用 Scheduler 的参数说明：

### 2.1. 通用参数
- `optimizer`：必须是包装了模型参数的优化器实例（如 `torch.optim.Adam(...)`）
- `last_epoch` （默认 -1）：用于断点续训，指示从哪个 epoch 开始恢复
- `verbose` （默认 False）：如果设为 `True`，每次更新学习率时会在控制台打印一条消息

### 2.2. 特定 Scheduler 的关键参数

- `StepLR`：等间距衰减

  - `step_size` (int)： 衰减的周期。例如设为 10，则每 10 个 epoch 调整一次
  - `gamma` (float)： 衰减系数。例如设为 0.1，则学习率变为原来的 10%

- `MultiStepLR`：指定间隔衰减

  - `milestones` (list)： 一个列表，指定在哪些 epoch 调整。例如 `[30, 80]` 表示在第 30 和第 80 个 epoch 调整
  - `gamma` (float)： 衰减系数

- **`ReduceLROnPlateau`**根据指标自适应衰减 - 非常实用

  - `mode`： `'min'` ：监测 Loss 时用，越小越好；`'max'`：监测 Acc 时用，越大越好
  - `factor`： 衰减系数（如 0.1）
  - `patience` (int)： 耐心值。如果指标在 `patience` 个 epoch 内没有改善，才触发衰减
  - `threshold`： 衡量指标是否改善的阈值

- `CosineAnnealingLR`：余弦退火

  - `T_max` (int)： 一个周期的迭代次数，通常设为总 Epoch 数）
  - `eta_min` (float)： 学习率的最小值，不会降到 0，而是降到这个值

## 3. 调用语法

使用 Scheduler 的标准流程分为三步：

1.  **定义优化器** `optimizer`
2.  **定义调度器** `scheduler`，并将优化器传入
3.  **在训练循环中调用** `scheduler.step()`

**标准代码结构**：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义模型和优化器
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 2. 定义调度器 (必须在优化器之后)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 3. 训练循环
for epoch in range(total_epochs):
    # 训练步骤
    train(...) 
    
    # 验证步骤
    validate(...)
    
    # 更新学习率
    # 注意：通常在每个 Epoch 结束时调用一次
    scheduler.step()
```

**特殊情况注意**：

如果是 `ReduceLROnPlateau`，调用语法略有不同，因为它需要传入监控指标：
```python
# 假设 val_loss 是验证集计算出的损失
scheduler.step(val_loss)
```

## 4. 示例

### 示例 1：`StepLR`，最基础

每 10 轮将学习率减半。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 每 10 个 epoch，lr = lr * 0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(100):
    train(...)
    scheduler.step()
    # Epoch 1-10: lr=0.1
    # Epoch 11-20: lr=0.05
    # ...
```

### 示例 2：`ReduceLROnPlateau`，进阶

当验证集 Loss 不再下降时，自动减小学习率。这是打比赛或做实验时非常常用的策略，因为它看得懂数据。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 监控指标不再优化时，lr = lr * 0.1，耐心为 5 个 epoch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

for epoch in range(100):
    train(...)
    val_loss = validate(...) # 必须计算验证集损失
    
    # 将验证损失传给 scheduler
    scheduler.step(val_loss)
    
    # 逻辑：
    # 如果 val_loss 连续 5 个 epoch 都没有明显下降（说明在波谷附近徘徊，出现过拟合）
    # 在第 6 个 epoch 开始时，学习率会自动从 0.001 变成 0.0001
```

### 示例 3：`CosineAnnealingLR` (平滑下降)

学习率像余弦函数曲线一样平滑下降，通常能获得比阶梯下降更好的最终效果。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# T_max 通常设置为总 epoch 数
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

for epoch in range(100):
    train(...)
    scheduler.step()
    # 学习率会从 0.1 开始，沿着余弦曲线慢慢降到 0
```
