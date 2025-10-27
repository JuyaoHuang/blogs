---
title: 'nn.activate_funcs模块'
author: Alen
published: 2025-10-27
description: "PyTorch神经网络激活函数模块nn.Module.activate_funcs的介绍"
first_level_category: "AI"
second_level_category: "PyTorch"
tags: ['ML','DL']
draft: false
---

# 激活函数

在神经网络中，每一层都包含线性运算（如卷积或矩阵乘法）和非线性激活。

如果没有激活函数，无论神经网络有多少层，它本质上都只是一个巨大的线性模型（*详情查看《PyTorch框架简要分析》中的激活函数部分*）。这样的模型能力非常有限，无法学习数据中复杂的模式

激活函数的作用就是为神经网络引入**非线性**，使其能够拟合各种复杂的函数，从而极大地提升模型的表达能力。它们作用于线性层的输出之上，决定了哪些神经元应该被激活

**在 PyTorch 中使用激活函数的两种方式**

在PyTorch种，以下两种方式等效：

1. **模块化方式 (nn.Module 子类)**
   - 例如 nn.ReLU(), nn.Sigmoid()
   - 这种方式将激活函数作为网络的一层。在 __init__ 中定义它，然后在 forward 方法中调用
   - **优点**：  当使用 nn.Sequential 构建模型时，结构非常清晰
2. **函数式方式 (torch.nn.functional)**
   - 例如 F.relu(x), F.sigmoid(x) (需要 `import torch.nn.functional as F`)
   - 这种方式直接将激活函数当作一个普通函数来调用
   - **优点**：  更灵活，无需在 __init__ 中预先定义，在 forward 方法中直接使用即可，代码可能更简洁

这两种方式在功能上是完全等价的，选择哪一种主要取决于个人编码风格和模型构建的复杂度

------

## 常用激活函数详解

下面详细介绍 torch.nn 中最常用的激活函数。

### 1. nn.ReLU - 修正线性单元 (Rectified Linear Unit)

这是目前深度学习中最常用、也是默认推荐的激活函数

- **数学公式**：
  $$
  f(x)=\max⁡(0,x)
  $$
  
  <img src="./imgs/ReLU.png" alt="relu" style="zoom:50%;" />
  
- **作用**：它是一个非常简单的  “斜坡 ”函数。如果输入大于0，则原样输出；如果输入小于等于0，则输出0

- **核心参数**：

  - inplace (bool, optional):   如果设置为 True，会原地修改输入张量，以节省内存。默认为 False

- **优点**：

  - 计算效率极高。
  - 在正数区间（x > 0）梯度恒为1，有效缓解了**梯度消失**问题
  - 能使网络具有稀疏性（一些神经元输出为0）
  
- **缺点**：

  - **Dying ReLU Problem**：如果一个神经元的输入恒为负数，那么它的输出将永远是0，梯度也永远是0，导致该神经元在后续的训练中无法被更新（“死亡”）
  
- **使用场景**：几乎所有类型神经网络的隐藏层

#### 2. nn.LeakyReLU - 带泄露的ReLU (Leaky Rectified Linear Unit)

为了解决 "Dying ReLU" 问题而提出

- **数学公式**：
  $$
  f(x) = 
  \begin{cases}
     x, & \text{if }x \geq 0 \\
     \text{negative\_slope} * x, & \text{otherwise}
  \end{cases}
  $$
   

  <img src="./imgs/LeakyReLU.png" style="zoom:50%;" />

- **作用**：  与ReLU类似，但在输入为负数时，不再输出0，而是输出一个非常小的、非零的、乘以一个固定斜率的负值

- **核心参数**：

  - negative_slope (float): 控制负斜率的角度。默认为 1e-2 (0.01)
  - inplace (bool, optional): 同上

- **优点**：

  - 继承了 ReLU 的大部分优点
  - 通过允许负值区域存在一个小的梯度，解决了 "Dying ReLU" 问题

- **使用场景**：当怀疑网络中存在大量死亡神经元时，可以尝试用 LeakyReLU 替代 ReLU

#### 3. nn.Sigmoid

一个历史悠久且经典的激活函数

- **数学公式**：

  $$
  f(x) = \frac{1}{1 + e^{-x}}
  $$
  ​    

- **作用**：将任意实数输入压缩到 (0, 1) 的范围内

- **优点**：

  - 输出范围有限，非常适合用作**二分类问题的输出层**，表示一个类别的概率
  - 在循环神经网络（RNN）的门控机制中（如 LSTM 的遗忘门、输入门）有重要应用

- **缺点**：

  - **梯度消失**：  当输入非常大或非常小时，函数的梯度（导数）接近于0，这会导致在反向传播时梯度信号变得非常微弱，使得网络难以训练
  - 输出不是零中心的，可能导致收敛速度变慢

- **使用场景**：主要用于二分类任务的**输出层**，或者需要将输出表示为概率的场景。**不推荐在隐藏层中使用**

#### 4. nn.Tanh - 双曲正切函数

- **数学公式**：

  $$
  f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  $$
  ​      

- **作用**：将任意实数输入压缩到 (-1, 1) 的范围内

- **优点**：

  - 输出是**零中心**的，通常比 Sigmoid 在隐藏层中表现更好，收敛速度更快

- **缺点**：

  - 与 Sigmoid 类似，在输入饱和区域（绝对值很大时）仍然存在**梯度消失**问题

- **使用场景**：  在 RNN 的隐藏层中非常常用。在某些MLP或CNN的隐藏层中也可以作为 ReLU 的替代品，但 ReLU 及其变体通常是首选

#### 5. nn.Softmax

它通常不被看作是隐藏层的激活函数，而是专门用于多分类问题的输出层。

- **数学公式**：

  $$
  f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
  $$
  ​     

  **作用**：将一个包含任意实数值的向量（通常称为 logits）转换为一个**概率分布**，其中每个元素都在 (0, 1) 之间，且所有元素之和为1

- **核心参数**：

  - dim (int):   指定沿着哪个维度进行 Softmax 计算。对于一个形状为 (batch_size, num_classes) 的输出，应设置为 dim=1

- **重要提示**：  在 PyTorch 中，如果你使用的损失函数是 `nn.CrossEntropyLoss`，你**不需要**在网络的最后一层手动添加 nn.Softmax；因为 nn.CrossEntropyLoss 内部已经隐式地包含了 LogSoftmax 操作

- **使用场景**：**专门用于多分类问题的输出层**

------

## 示例代码

下面的代码演示了如何在模型中同时使用模块化和函数式两种方式来应用激活函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 准备数据
# 假设有一批数据，每条数据有 100 个特征
batch_size = 4
input_features = 100
dummy_input = torch.randn(batch_size, input_features)

# 2. 定义一个包含多种激活函数的网络
class ActivationSamplerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ActivationSamplerNet, self).__init__()
        
        # 使用模块化方式定义层 
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # 将 ReLU 作为网络的一层
        self.relu_layer = nn.ReLU() 
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # LeakyReLU 也可以作为一层
        self.leaky_relu_layer = nn.LeakyReLU(negative_slope=0.01)
        
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 模块化调用 
        x = self.fc1(x)
        print(f"After fc1: {x[0, :5].detach().numpy()}")
        x = self.relu_layer(x) # 调用预定义的 ReLU 层
        print(f"After ReLU: {x[0, :5].detach().numpy()}")
        
        # 函数式调用 
        x = self.fc2(x)
        print(f"\nAfter fc2: {x[0, :5].detach().numpy()}")
        # 直接使用 functional API 中的 tanh 函数
        x = F.tanh(x) 
        print(f"After Tanh: {x[0, :5].detach().numpy()}")
        
        # 最终输出 logits (用于分类)
        logits = self.fc3(x)
        print(f"\nFinal Logits (before Softmax): {logits.detach().numpy()}")
        
        return logits

# 3. 实例化并测试网络
hidden1, hidden2, num_classes = 64, 32, 5
model = ActivationSamplerNet(input_features, hidden1, hidden2, num_classes)

# 将数据输入模型
output_logits = model(dummy_input)

# 4. 演示 Softmax 的使用 (通常在模型外部或损失函数内部)
print("\n--- Softmax Demonstration ---")
# 使用 functional API
probabilities = F.softmax(output_logits, dim=1) # 在类别维度上计算概率

print(f"Output Logits shape: {output_logits.shape}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Probabilities for first sample:\n{probabilities[0].detach().numpy()}")
print(f"Sum of probabilities for first sample: {torch.sum(probabilities[0]).item()}")
  
```

---

## 总结

|     激活函数     | 输出范围 |          优点           |           缺点           |          主要使用场景          |
| :--------------: | :------: | :---------------------: | :----------------------: | :----------------------------: |
|   **nn.ReLU**    | [0, +∞)  |  计算快，缓解梯度消失   |    可能导致神经元死亡    |      **隐藏层的默认选择**      |
| **nn.LeakyReLU** | (-∞, +∞) |   解决了死亡ReLU问题    |    性能提升不总能保证    |          ReLU的替代品          |
|  **nn.Sigmoid**  |  (0, 1)  |  输出为概率，用于门控   | 易导致梯度消失，非零中心 | **二分类输出层**，LSTM/GRU门控 |
|   **nn.Tanh**    | (-1, 1)  |       输出零中心        |      易导致梯度消失      |  RNN隐藏层，某些MLP/CNN隐藏层  |
|  **nn.Softmax**  |  (0, 1)  | 输出为概率分布，总和为1 |       计算相对复杂       |        **多分类输出层**        |