---
title: '自动求导autograd'
author: Alen
published: 2025-10-26
description: "Autograd 自动求导介绍和核心语法"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['python']
draft: false
---


# Autograd 核心语法总结

[参考文档](https://gitee.com/flycity/ai_tutorial_book/blob/master/chap5-Pytorch%E5%9F%BA%E7%A1%80/5.2%20%E5%BC%A0%E9%87%8F%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.ipynb)

Autograd 是一个自动计算梯度的系统

在神经网络中，我们通过**前向传播**计算模型的输出和损失，然后通过**反向传播**（Backward Pass）计算损失函数相对于模型每个参数的梯度。最后，我们使用这些梯度来更新模型的参数（例如，通过梯度下降）

Autograd 的作用就是自动完成**反向传播**计算梯度的过程

你只需要定义好前向传播的计算，PyTorch 就会自动构建一个**动态计算图**，然后通过这个图来高效地计算梯度

Autograd 的工作主要围绕 torch.Tensor 对象的三个核心属性和一个核心方法展开：

### 1. requires_grad 属性

这是一个布尔类型的属性

- 如果要让 PyTorch 追踪对某个张量的操作以进行自动求导，你需要将其 `requires_grad` 属性设置为 `True`
- 模型中需要学习的参数，例如线性回归的$W,b$，必须将其设置为 `True`
- 默认情况下，新建立的tensor 的此属性为 `False`

-   **创建时指定**

    ```python
    import torch
    
    # 创建一个需要梯度的张量
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    # tensor([[1., 1.],
    #         [1., 1.]], requires_grad=True)
    ```

-   **后续修改**

    ```python
    a = torch.randn(2, 2) # 默认 requires_grad=False
    a.requires_grad_(True) # 就地修改
    print(a.requires_grad) # True
    ```

-   线性回归：

    ```python
    import torch
    
    # x 是输入数据，不需要对它求梯度
    x = torch.tensor([2.0, 3.0])
    print(f"x.requires_grad: {x.requires_grad}") # 输出: False
    
    # w 是模型权重，需要对它求梯度以进行优化
    w = torch.tensor([0.5, 0.1], requires_grad=True)
    print(f"w.requires_grad: {w.requires_grad}") # 输出: True
    
    b = torch.tensor(0.1, requires_grad=True)
    print(f"b.requires_grad: {b.requires_grad}") # 输出: True
    ```

-   任何从带有 `requires_grad=True` 的张量计算得到的新张量，都会自动地 `requires_grad=True`，并且其 `grad_fn` 属性会指向创建它的函数（例如，加法是 `AddBackward0`）

### 2. backward() 方法

这是启动梯度计算的启动键

一旦有了一个标量 $Scaler$ 输出，通常是损失值$Loss$，就可以调用它的 `.backward()` 方法来自动计算所有 `requires_grad=True` 的输入张量相对于这个标量输出的梯度

在 loss 上调用` loss.backward()` 时，`Autograd` 会：

1. 从 loss 这个节点开始，沿着 .grad_fn 构成的计算图向后追溯。
2. 使用**链式法则**计算 loss 相对于图中每个叶子节点（即设置了 requires_grad=True 的张量）的梯度

- **基本用法 (标量输出)**

  ```python
  x = torch.ones(2, 2, requires_grad=True)
  y = x + 2
  z = y * y * 3
  out = z.mean() # out 是一个标量
  
  print(out) # tensor(27., grad_fn=<MeanBackward0>)
  
  # 计算梯度
  out.backward()
  ```

- **非标量输出的 `backward()`**
  如果 `backward()` 被调用在一个非标量张量上，需要提供一个与该张量形状相同的 `gradient` 参数，它代表了上游传入的梯度（通常在向量-雅可比积中用到）

  ```python
  x = torch.randn(3, requires_grad=True)
  y = x * 2
  # y 不是标量，直接 y.backward() 会报错
  # 假设我们想计算 y 相对于 x 的雅可比矩阵与向量 v 的乘积
  v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
  y.backward(gradient=v) # 或者 y.backward(v)
  print(x.grad) # tensor([0.2000, 2.0000, 0.0020])
  ```

  在典型的神经网络训练中，损失函数总是一个标量，所以通常不需要传递 `gradient` 参数。

### 3. .grad 属性

在调用 `.backward()` 之后，梯度会累积到参与计算的、`requires_grad=True` 的**叶子节点**张量的 `.grad` 属性中

**注意**：  梯度是**累加** 的，而不是覆盖。这意味着在每次反向传播之前，通常需要手动将梯度清零。

- **访问梯度：**

  ```python
  # out.backward() 已经执行
  print(x.grad)
  # tensor([[4.5000, 4.5000],
  #         [4.5000, 4.5000]])
  ```

- **注意：**

  *   只有叶子节点（用户创建的、`requires_grad=True`的张量，或者通过 `nn.Parameter` 定义的模型参数）才会填充 `.grad`。中间张量的梯度在计算后通常会被释放以节省内存
  *   **梯度是累积的：** 多次调用 `backward()` 会将新的梯度值加到已有的 `.grad` 上。因此，在每次优化迭代开始时，通常需要使用 `optimizer.zero_grad()` 或手动将梯度清零（例如 `x.grad.zero_()`）

### 4. 停止梯度追踪 (torch.no_grad() 和 .detach())

有时我们不希望 PyTorch 追踪某些操作的梯度

- **`with torch.no_grad():` 上下文管理器：**
  在这个代码块内的所有计算都不会被追踪，即使输入张量设置了 `requires_grad=True`。这在模型评估/推断阶段非常有用，可以加速计算并节省内存。

  ```python
  x = torch.ones(1, requires_grad=True)
  print(x.requires_grad) # True
  with torch.no_grad():
      y = x * 2
  print(y.requires_grad) # False
  # y.backward() # 这会报错，因为 y 没有 grad_fn
  ```

  也可以用作装饰器 `@torch.no_grad`。

- **.detach() 方法：**
  创建一个与原张量共享数据但不参与梯度计算的新张量。它会从当前的计算图中分离出来。

  ```python
  x = torch.ones(1, requires_grad=True)
  y = x * 2
  z = y.detach() # z 不再与 x 的计算图连接
  
  print(z.requires_grad) # False
  # 如果你想让 z 之后可以求导（开始新的历史），可以：
  # z.requires_grad_(True)
  
  # 即使 y 的梯度被计算，它也不会回传到 x 通过 z 的路径
  out = y * 3 # 假设这是损失
  out.backward()
  print(x.grad) # tensor([6.])
  
  # 如果我们基于 z 进行操作
  # x.grad.zero_() # 清零梯度
  # out_detached = z * 3
  # out_detached.requires_grad_(True) # 假设我们想从这里开始新的计算
  # print(out_detached.grad_fn) # None，因为它是一个新的叶子节点
  ```

  `.detach()` 常用于以下情况：

  1.  当你想将一个张量的值用于某些不应影响梯度的计算时（例如，将其转换为 NumPy 数组，或作为某个固定值使用）。
  2.  在某些复杂的网络结构中，需要显式地切断梯度流。

### 5. grad_fn 属性

当一个张量是通过对设置了 `requires_grad=True` 的张量进行操作而创建出来时，它会自动获得一个 `.grad_fn` 属性

这个属性引用了一个创建该张量的函数对象（例如，加法操作的结果张量，其 `grad_fn` 是 `<AddBackward0 object>`），该对象记录了创建这个张量的操作

用户手动创建的张量，称为**叶子节点** 的 `grad_fn` 是 `None`

`autograd` 通过这些 `grad_fn` 回溯计算图

```python
# w, b 是叶子节点，它们是手动创建的
print(f"w.grad_fn: {w.grad_fn}") # 输出: None
print(f"b.grad_fn: {b.grad_fn}") # 输出: None

# y 是通过操作 w, x, b 创建的
y = torch.dot(w, x) + b
print(f"y.requires_grad: {y.requires_grad}") # 输出: True (因为 w 和 b 需要梯度)
print(f"y.grad_fn: {y.grad_fn}")           # 输出: <AddBackward0 object at ...>
```

​	y 的 grad_fn 是 AddBackward0，因为它是由一个加法操作最后生成的。你可以通过 .next_functions 继续回溯，看到完整的计算图

### 6. 关于 backward() 的更多细节

- **retain_graph=True：**
  默认情况下，在执行 .backward() 后，为了节省内存，计算图会被释放。

  如果你需要对同一个计算图（或其一部分）多次调用 backward()

  例如，计算多个不同损失对相同参数的梯度，或者需要梯度的梯度，需要在非最后一次的 backward() 调用中设置 retain_graph=True

  ```py
  x = torch.tensor([2.0], requires_grad=True)
  y1 = x * 2
  y2 = x * 3
  
  # 假设 y1 和 y2 都是某种损失的一部分
  y1.backward(retain_graph=True) # 保留图
  print(x.grad) # tensor([2.])
  
  y2.backward() # 这是最后一次，可以不设置或 retain_graph=False (默认)
  print(x.grad) # tensor([5.]) (2.0 + 3.0，因为梯度累积)
  ```

  过度使用 retain_graph=True 会导致内存消耗增加

- **create_graph=True：**
  如果设置为 True，会构建导数计算的计算图。这允许计算高阶导数（例如，梯度的梯度）。

  ```py
  x = torch.tensor([2.0], requires_grad=True)
  y = x ** 3
  grad_x = torch.autograd.grad(y, x, create_graph=True) # 计算一阶导数，并创建图
  print(grad_x) # (tensor([12.], grad_fn=<MulBackward0>),)
  
  # grad_x[0] 是 dy/dx
  # 现在可以计算二阶导数 d(dy/dx)/dx
  grad_grad_x = torch.autograd.grad(grad_x[0], x)
  print(grad_grad_x) # (tensor([12.]),) (对于 y=x^3, y''=6x, x=2 时为12)
  ```

  注意：torch.autograd.grad() 是一个更底层的函数，可以直接计算梯度，而不仅仅是调用 .backward()。 

### 7. 完整示例：线性回归

让我们通过一个完整的例子把所有概念串起来。
假设我们有一个简单的模型 $y_pred = w * x + b$，目标是计算损失 loss 相对于 w 和 b 的梯度

```python
import torch

def pr(val):
    print(val)

# 1. 定义输入数据和需要学习的参数 (叶子节点)
x = torch.tensor(3.0)
w = torch.tensor(4.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
y_true = torch.tensor(18.0) # 真实的目标值
# 2. 前向传播
y_pred = w*x+b
pr(f'y_pred:{y_pred}')
pr(f'y_pred.grad_fn:{y_pred.grad_fn}')
pr(f'y_pred.requires_grad:{y_pred.requires_grad}')
# 3. 计算loss：标量
loss = (y_pred-y_true)**2
pr(f'loss:{loss}')
pr(f'loss.grad_fn:{loss.grad_fn}')
# 反向传播：自动计算梯度
pr(f'w.grad:{w.grad}')
pr(f'b.grad:{b.grad}')
loss.backward()
# 5.查看梯度
pr(f'w.grad:{w.grad}')
pr(f'b.grad:{b.grad}')
```

```
y_pred:17.0
y_pred.grad_fn:<AddBackward0 object at 0x0000017A339E7E80>
y_pred.requires_grad:True
loss:1.0
loss.grad_fn:<PowBackward0 object at 0x0000017A339E7E80>
w.grad:None
b.grad:None
w.grad:-6.0
b.grad:-2.0
```

### 8. 注意事项

#### 8.1 梯度累加和清零

在典型的训练循环中，梯度会累加。如果你不清除它们，梯度就会出错

```python
optimizer = torch.optim.SGD([w, b], lr=0.01)

# 在训练循环中
for i in range(2):
    # 1. 将梯度清零
    optimizer.zero_grad()
    # 或者手动清零:
    # if w.grad is not None:
    #     w.grad.zero_()
    # if b.grad is not None:
    #     b.grad.zero_()

    # 2. 前向传播
    y_pred = w * x + b
    loss = (y_pred - y_true)**2

    # 3. 反向传播
    loss.backward()

    print(f"Epoch {i+1}: w.grad={w.grad}, b.grad={b.grad}")

    # 4. 更新参数 (这里我们只演示，不实际更新以观察梯度)
    # optimizer.step()
```

输出会显示，两次循环的梯度值是相同的，因为每次都清零了

如果注释掉 optimizer.zero_grad()，第二次的梯度将会是第一次的两倍

### **总结**

Autograd 的工作流程可以概括为：

1. **设置**：   为需要优化的参数张量设置 requires_grad=True
2. **前向传播**：  执行计算，PyTorch 会自动构建一个记录了所有操作的动态计算图
3. **计算损失**：得到一个最终的标量损失值
4. **反向传播**：在损失上调用 `.backward()`，PyTorch 会遍历计算图，自动计算梯度
5. **读取梯度**：从参数的 .grad 属性中获取计算出的梯度
6. **梯度清零**：在下一次迭代前，记得使用 `optimizer.zero_grad()`清除旧的梯度
