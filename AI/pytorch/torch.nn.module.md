---
title: 'nn.module模块'
author: Alen
published: 2025-10-27
description: "PyTorch神经网络基底模型nn.module的介绍"
first_level_category: "AI"
second_level_category: "PyTorch"
tags: ['ML','DL']
draft: false
---

# torch.nn.Module 

nn.Module 是 PyTorch 中所有神经网络模块的**基类**，可以把它想象成一个智能容器或一个蓝图

- **作为容器**：

  它可以包含其他 nn.Module（例如神经网络的层，如 nn.Linear, nn.Conv2d），也可以包含可学习的参数（nn.Parameter，本质上是 torch.Tensor）

- **作为蓝图**：

  通过继承 nn.Module 并实现其方法（尤其是 forward 方法）来定义自己的网络架构

**核心思想**：nn.Module 设计哲学是面向对象方法，将神经网络的通用的层封装好，后续只需要调用工具箱内部的API即可构建模型

## nn.Module 的自动参数注册功能

创建一个类并继承 nn.Module后，在 __init__ 方法中，将一个 nn.Module 的子类（如 nn.Linear）或一个 nn.Parameter 赋值给类的属性时，nn.Module 会**自动**将它们注册为模块的子模块或参数

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 这行代码做了两件事:
        # 1. 创建一个 nn.Linear 实例
        # 2. 将其作为 MyModel 的一个子模块自动注册，名字是 "layer1"
        self.layer1 = nn.Linear(in_features=10, out_features=5) 
        
        # 这也会被自动注册为一个可学习的参数
        self.my_bias = nn.Parameter(torch.randn(5)) 
```

这个自动注册机制是 `nn.Module` 便捷性的核心，它使得后续的参数访问、模型移动等操作变得异常简单

---

## 封装的核心方法和属性

以下是 `nn.Module` 中最重要、最常用的一些方法和属性：

### 1. `__init__(self)`：构造函数

- **作用**：  定义神经网络的结构。在这里，你需要实例化所有需要的层（如卷积层、线性层、激活函数等）并将它们作为类的属性
- **注意**：  在子类的 `__init__` 方法中，**第一行必须调用 `super().__init__()`**，以确保父类的初始化逻辑被正确执行

### 2. forward(self, *input)：前向传播

- **作用**：  
        
    定义数据在网络中的**计算流程**
    这个方法接收输入张量，让数据依次流过在 `__init__` 中定义的各个层，并最终返回输出张量
- **注意**：  此方法**必须手动实现**
- **调用方式**：

    通常不直接调用 `model.forward(x)`，而是使用 `model(x)`，这种调用方式除了执行 `forward` 外，还会执行 PyTorch 内部注册的一些  "钩子 (hooks)"，这是一种更规范的做法

### 3. .parameters() 和 .named_parameters()

- **作用**：获取模型中所有**可学习的参数**（即权重和偏置）
    - `.parameters()`:   返回一个包含所有参数 (tensors) 的**迭代器**，这是传递给优化器 `torch.optim.Optimizer` 的对象
    - `.named_parameters()`:   返回一个迭代器，每个元素是一个 `(name, parameter)` 的元组，这在调试或需要对特定参数进行操作时非常有用

### 4. .children() 和 .modules()

- **作用**：  遍历模型的各个子模块
    - `.children()`:   返回一个迭代器，只包含模型的**直接子模块**
    - `.modules()`:   返回一个迭代器，会**递归地**遍历模型的所有模块（包括模型自身和所有深层嵌套的子模块）

### 5. .train() 和 .eval()

- **作用**：  切换模型的模式

    这是两个非常重要的方法，尤其是在使用**Dropout**和**Batch Normalization**等层时
    - `.train()`: 将模型设置为**训练模式**。在这种模式下，Dropout 层会随机丢弃神经元，Batch Normalization 层会使用当前批次数据的均值和方差进行归一化。
    - `.eval()`: 将模型设置为**评估模式**。在这种模式下，Dropout 层会失效（所有神经元都参与计算），Batch Normalization 层会使用在整个训练集上学习到的均值和方差进行归一化。
*   **注意**：在训练开始前调用 `model.train()`，在验证或测试时调用 `model.eval()` 是一个必须养成的良好习惯。

### 6. .to(device)

- **作用**：  将模型的所有参数和缓冲区移动到指定的设备上，例如 CPU (`'cpu'`) 或 GPU (`'cuda'`)
- **注意**：  这是一个**原地 (in-place)** 操作。`model.to(device)` 会修改模型自身。同时，输入数据也必须和模型在同一个设备上才能进行计算

### 7. .state_dict() 和 .load_state_dict(state_dict)

- **作用**：  用于模型的**保存和加载**
- `.state_dict()`:   返回一个 Python 字典，将每个参数和持久化缓冲区（persistent buffer）的名字映射到其对应的张量，**只包含可学习的参数和缓冲区，不包含模型结构**。
- `.load_state_dict(state_dict)`:   将从 `state_dict` 字典中保存的参数和缓冲区加载到当前模型中。

---

## 示例代码：构建一个简单的多层感知机 (MLP)

这个例子将把上面介绍的所有概念串联起来。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 0. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 定义模型 - 继承 nn.Module
class SimpleMLP(nn.Module):
    # (1) __init__: 定义网络结构
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__() # 必须先调用父类构造函数
        
        # 定义网络层
        self.flatten = nn.Flatten() # 将输入的图像展平
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    # (2) forward: 定义数据流
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. 实例化模型
input_size = 28 * 28  # 假设是 MNIST 图像
hidden_size = 512
num_classes = 10
model = SimpleMLP(input_size, hidden_size, num_classes)

# 3. 将模型移动到设备
model.to(device)

# 4. 探索模型的方法和属性
print("--- Model Architecture ---")
print(model)

print("\n--- Named Parameters ---")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Name: {name}, Shape: {param.shape}")

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 将模型的所有可学习参数传递给优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 模拟训练过程
print("\n--- Simulating Training ---")
# 创建一些假数据
batch_size = 64
dummy_input = torch.randn(batch_size, 1, 28, 28).to(device) # (B, C, H, W)
dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

# (5.1) 设置为训练模式
model.train() 

# 训练三步曲
optimizer.zero_grad()               # 清空梯度
outputs = model(dummy_input)        # 前向传播 (注意是 model(x), 不是 model.forward(x))
loss = criterion(outputs, dummy_labels) # 计算损失
loss.backward()                     # 反向传播，计算梯度
optimizer.step()                    # 更新参数

print(f"Calculated Loss: {loss.item():.4f}")

# 7. 模拟评估过程
print("\n--- Simulating Evaluation ---")
# (5.2) 设置为评估模式
model.eval() 
with torch.no_grad(): # 在评估时，关闭梯度计算以节省内存和计算
    test_output = model(dummy_input)
    _, predicted = torch.max(test_output.data, 1)
    print(f"Predicted labels (first 10): {predicted[:10].cpu().numpy()}")

# 8. 保存和加载模型
print("\n--- Saving and Loading Model State ---")
# (7.1) 保存模型的状态字典
torch.save(model.state_dict(), "mlp_model.pth")
print("Model state_dict saved to mlp_model.pth")

# (7.2) 加载模型
# 首先需要创建一个相同结构的模型实例
new_model = SimpleMLP(input_size, hidden_size, num_classes)
# 然后加载保存的参数
new_model.load_state_dict(torch.load("mlp_model.pth"))
new_model.to(device)
new_model.eval()
print("New model instance created and state_dict loaded.")

# 验证加载是否成功
with torch.no_grad():
    new_output = new_model(dummy_input)
    # 检查新旧模型的输出是否一致
    assert torch.allclose(test_output, new_output)
    print("Verification successful: New model produces the same output.")
  
```

### **总结**

nn.Module 是 PyTorch 的核心抽象。通过继承它，你可以：

1. 在 __init__ 中**定义**你的网络层
2. 在 forward 中**定义**数据的流动方式
3. 自动获得对所有参数的管理能力，将它们传递给优化器 (`.parameters()`)
4. 切换训练/评估模式 (`.train(), .eval()`)
5. 将整个模型及其所有参数移至不同设备 (`.to(device)`)
6. 实现模型的保存和加载 (`.state_dict(), .load_state_dict()`)
