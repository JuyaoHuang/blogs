---
title: Tensor基础
author: Alen
published: 2025-10-26
description: "Tensor的基础语法和数据操作"
first_level_category: "AI"
second_level_category: "PyTorch"
tags: ['python']
draft: false
---

# Tensor

在 PyTorch 中，**张量 (Tensor)** 是进行所有计算的核心。

你可以把它看作是 NumPy ndarray 的一个功能更强大的替代品。它不仅提供了类似 NumPy 的多维数组操作功能，还带来了两个关键特性：

1. **GPU 加速**：张量可以轻松地在 CPU 和 GPU 之间转移，从而利用 GPU 强大的并行计算能力来加速运算
2. **自动求导**：张量能够自动追踪其计算历史，用于高效地计算梯度，这是深度学习模型训练的核心

Tensor与Numpy的具体差别查看<a href="https://juayohuang.top/posts/ai/pytorch/tensor%E4%B8%8Enumpy%E7%9A%84%E5%8C%BA%E5%88%AB/" target="_blank" ref="noopener noreferrer">此篇文章</a>

<a href="https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/" target="_blank" ref="noopener noreferrer">有关tensor的全部语法查看此篇文章</a>

本篇参考: <a href="https://zh.d2l.ai/chapter_preliminaries/ndarray.html" target="_blank" ref="noopener noreferrer"> 《动手学深度学习》 </a>

就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构。张量是描述具有**任意数量轴**的 $n$维数组的通用名字。

例如：向量是一阶张量，矩阵是二阶张量




## 1. 创建tensor

可以从 Python 的 list 或 tuple 创建张量，也可以创建具有特定形状和值的张量

```python
import torch

# 1. 从已有的 Python 列表创建张量
x_from_list = torch.tensor([1, 2, 4, 8])
print("从列表创建:\n", x_from_list)
# tensor([1, 2, 4, 8])

# 2. 创建指定形状的全零张量
x_zeros = torch.zeros((2, 3, 4)) # 创建一个形状为 2x3x4 的三维 全0张量
print("\n全零张量:\n", x_zeros)
# tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#
#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

# 3. 创建指定形状的全一张量
x_ones = torch.ones((2, 3, 4))
print("\n全一张量:\n", x_ones)
# 与上面类似，只是元素为1

# 4. 创建指定形状的随机张量
# 每个元素从均值为0、方差为1的标准正态分布中随机采样
x_randn = torch.randn(3, 4)
print("\n随机张量:\n", x_randn)
# 一个3x4的矩阵，元素是随机数

# 5. 直接通过提供列表来创建具有特定值的张量
x_direct = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("\n直接创建:\n", x_direct)
# tensor([[2, 1, 4, 3],
#         [1, 2, 3, 4],
#         [4, 3, 2, 1]])

# 查看张量的形状、大小和数据类型
print("\nx_direct的形状 (shape):", x_direct.shape)
# torch.Size([3, 4])
print("x_direct的元素总数 (numel):", x_direct.numel())
# 12
print("x_direct的数据类型 (dtype):", x_direct.dtype)
# torch.int64
```

## 2. 张量的运算

PyTorch 提供了丰富的运算，包括标准的算术运算和更复杂的操作

### 元素级运算

以下操作会对 tensor里的每个元素进行操作

1. 矩阵的按元素乘法

   两个矩阵的按元素乘法称为*Hadamard积*，数学符号 $\odot$

   ```
   A * B
   ```

   A，B都是矩阵

```python
import torch

x = torch.tensor([1.0, 2.0, 4.0, 8.0])
y = torch.ones_like(x) * 2 # 创建一个和x形状相同，元素全为2的张量
print('x =', x)
print('y =', y)
print('x + y =', x + y)  # 加法
print('x - y =', x - y)  # 减法
print('x * y =', x * y)  # 乘法 (哈达玛积:Hadamard积)
print('x / y =', x / y)  # 除法
print('x ** y =', x ** y) # 幂运算
print('exp(x) =', torch.exp(x)) # 指数运算
```
```bash
x = tensor([1., 2., 4., 8.])
y = tensor([2., 2., 2., 2.])
x + y = tensor([ 3.,  4.,  6., 10.])
x - y = tensor([-1.,  0.,  2.,  6.])
x * y = tensor([ 2.,  4.,  8., 16.])
x / y = tensor([0.5000, 1.0000, 2.0000, 4.0000])
x ** y = tensor([ 1.,  4., 16., 64.])
exp(x) = tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

### 张量连接

可以将多个张量按照 行/列 连接在一起

```python
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 按维度0 (行) 连接
cat_dim0 = torch.cat((X, Y), dim=0)
print("按维度0连接 (上下堆叠):\n", cat_dim0)
print("形状:", cat_dim0.shape)

# 按维度1 (列) 连接
cat_dim1 = torch.cat((X, Y), dim=1)
print("\n按维度1连接 (左右拼接):\n", cat_dim1)
print("形状:", cat_dim1.shape)
```
```bash
按维度0连接 (上下堆叠):
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [ 2.,  1.,  4.,  3.],
        [ 1.,  2.,  3.,  4.],
        [ 4.,  3.,  2.,  1.]])
形状: torch.Size([6, 4])

按维度1连接 (左右拼接):
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
        [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])
形状: torch.Size([3, 8])
```

### 求和

```python
import torch
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))

# 求和所有元素
sum_all = X.sum()
print("所有元素的和:", sum_all)

# 按行求和 (dim=0)
sum_dim0 = X.sum(dim=0)
print("按行求和:", sum_dim0)

# 按列求和 (dim=1)
sum_dim1 = X.sum(dim=1)
print("按列求和:", sum_dim1)
```

#### 降维求和

默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量

我们还可以指定张量沿哪一个轴来通过求和降低维度

以矩阵为例，为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定`axis=0`。 由于输入矩阵沿 0 轴降维以生成输出向量，因此输入轴 0 的维数在输出形状中消失，即轴 0被压缩为 1维

```python
def pr(val):
    print(val)

x = torch.arange(12).reshape(3, 4)
pr(x)
pr(x.sum(axis=0))
pr(x.sum(axis=1))
```

```bash
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
tensor([12, 15, 18, 21])
tensor([ 6, 22, 38])
```

一个与求和相关的量是*平均值*（mean或average）。 我们通过将总和除以元素总数来计算平均值。 在代码中，我们可以调用函数来计算任意形状张量的平均值

```python
pr(x.sum()/x.numel())
```

```bash
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
tensor(5.5000)
```

同样，计算平均值的函数也可以沿指定轴降低张量的维度。

```
pr(x.mean(axis=0))
```

```bash
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]], dtype=torch.float64)
tensor([4., 5., 6., 7.], dtype=torch.float64)
```



#### 非降维求和

但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用

例如，由于`sum_A`在对每行进行求和后仍保持两个轴，我们可以通过广播将`A`除以`sum_A`

```python
def pr(val):
    print(val)

x = torch.arange(12,dtype=torch.float32).reshape(3, 4)
pr(x)
sum_x = x.sum(axis=1,keepdim=True)
pr(sum_x)
pr(x/sum_x)
```

```bash
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
tensor([[ 6.],
        [22.],
        [38.]])
tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895]])
```



## 3. 广播机制 Broadcasting

当对两个形状不同的张量进行运算时，PyTorch 会尝试**广播** 它们，使其形状匹配，从而执行元素级运算。广播的规则是：

1. 如果张量的维度不同，先在维度较小的张量前面补 1，使其维度相同
2. 在任何一个维度上，如果一个张量的大小是 1，另一个张量的大小大于 1，那么大小为1的张量会沿着该维度被复制扩展，以匹配另一个张量的大小
3. 如果在任何一个维度上，两个张量的大小都大于 1 但不相等，则无法广播，会报错

```python
import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

print("a (shape 3x1):\n", a)
print("b (shape 1x2):\n", b)

# a被广播成3x2, b被广播成3x2
c = a + b
print("a + b (shape 3x2):\n", c)
```

```bash
a (shape 3x1):
 tensor([[0],
        [1],
        [2]])
b (shape 1x2):
 tensor([[0, 1]])
a + b (shape 3x2):
 tensor([[0, 1],
        [1, 2],
        [2, 3]])
```



## 4. 索引和切片

可以像在 NumPy 中一样，通过索引访问张量的部分元素

```python
import torch
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))

print("原始张量 X:\n", X)
print("\n最后一个元素:", X[-1])
print("第1行, 第2列的元素:", X[1, 2])

# 切片
print("\n第1行到第2行:\n", X[1:3, :])
print("第1列:\n", X[:, 1])

# 写入值
X[1, 2] = 99
print("\n修改后的 X:\n", X)

# 通过切片写入多个值
X[0:2, :] = 12
print("\n再次修改后的 X:\n", X)
```

```bash
原始张量 X:
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])

最后一个元素: tensor([ 8.,  9., 10., 11.])
第1行, 第2列的元素: tensor(6.)

第1行到第2行:
 tensor([[ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
第1列:
 tensor([1., 5., 9.])

修改后的 X:
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5., 99.,  7.],
        [ 8.,  9., 10., 11.]])

再次修改后的 X:
 tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
```



## 5. 节省内存

深度学习中，我们经常处理大数据和大型模型，内存开销是一个重要问题。Y = X + Y 这样的操作会为结果 Y 分配新的内存。为了节省内存，我们可以使用**原地操作**

```python
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.ones_like(X)

# 记录Y在操作前的内存地址
before = id(Y)
# 这种操作会分配新内存
Y = Y + X
after = id(Y)
print("Y = Y + X, 内存地址是否相同:", before == after) # False

# --- 使用原地操作 ---
Z = torch.zeros_like(X)
print('Z的id:', id(Z))
# 方式一：使用切片赋值
Z[:] = X + Y
print('Z[:] = ...后, Z的id:', id(Z))

# 方式二：使用带下划线的函数 (例如 .add_())
before_add_ = id(Z)
Z.add_(X) # Z = Z + X
after_add_ = id(Z)
print("Z.add_(X)后, Z的id:", id(Z))
print("内存地址是否相同:", before_add_ == after_add_) # True
```



## 6. 与np互换

将 PyTorch 张量转换为 NumPy 数组（ndarray），反之亦然，非常简单。**但需要注意：torch张量和 numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量**

```python
import torch
import numpy as np

# Tensor to NumPy
A = torch.randn(3, 4)
B = A.numpy()
print("PyTorch Tensor:\n", A)
print("NumPy ndarray:\n", B)

# 修改一个会影响另一个
A.add_(1)
print("\n修改Tensor后, Tensor A:\n", A)
print("修改Tensor后, NumPy B:\n", B)

# NumPy to Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print("\nNumPy ndarray:\n", a)
print("PyTorch Tensor:\n", b)

# 修改一个会影响另一个
np.add(a, 1, out=a) # a = a + 1
print("\n修改NumPy后, NumPy a:", a)
print("修改NumPy后, Tensor b:", b)
```

## 7. 线性代数运算

1. **点积**

   给定两个向量 $\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，它们的点积 $\mathbf{x}^\top\mathbf{y}$是相同位置的按元素乘积的和 
   $$
   \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i
   $$

   ```python
   x = torch.arange(4,dtype=torch.float32)
   y = torch.ones(4,dtype=torch.float32)
   pr(x)
   pr(y)
   pr(torch.dot(x,y)
   ```

   ```bash
   tensor([0., 1., 2., 3.])
   tensor([1., 1., 1., 1.])
   tensor(6.)
   ```

   当然，我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积：

   ```python
   print(torch.sum(x*y))
   ```

2. **矩阵-向量积**

   即矩阵和向量的乘积

   设一矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$和向量 $\mathbf{x} \in \mathbb{R}^n$，矩阵以行向量表示：
   $$
   \begin{split}\mathbf{A}=
   \begin{bmatrix}
   \mathbf{a}^\top_{1} \\
   \mathbf{a}^\top_{2} \\
   \vdots \\
   \mathbf{a}^\top_m \\
   \end{bmatrix},\end{split}
   $$
   其中每个 $a_i^T \in R$都是行向量，表示矩阵的第 i 行。

   向量积是$\mathbf{Ax}$一个是一个长度为 m 的列向量

   我们可以把一个矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$乘法看作一个从 $\mathbb{R}^{n}$ 到 $\mathbb{R}^{m}$ 向量的转换。

   说白了就是：

   对于一个 $\mathbf{A} \in \mathbb{R}^{m \times n}$的矩阵，将它按**行**切分，变为一个 $m*1$的$m$维列向量$VerctorA$，列向量的每一个元素都是一个$1*n$的行向量，然后这个$VerctorA$去和目标向量 $\mathbf{x}$进行元素相乘，得到一个 m维的列向量
   $$
   \begin{split}\mathbf{A}\mathbf{x}
   = \begin{bmatrix}
   \mathbf{a}^\top_{1} \\
   \mathbf{a}^\top_{2} \\
   \vdots \\
   \mathbf{a}^\top_m \\
   \end{bmatrix}\mathbf{x}
   = \begin{bmatrix}
    \mathbf{a}^\top_{1} \mathbf{x}  \\
    \mathbf{a}^\top_{2} \mathbf{x} \\
   \vdots\\
    \mathbf{a}^\top_{m} \mathbf{x}\\
   \end{bmatrix}.\end{split}
   $$
   

   在代码中使用张量表示矩阵-向量积，我们使用`mv`函数。 当我们为矩阵`A`和向量`x`调用`torch.mv(A, x)`时，会执行矩阵-向量积。 注意，`A`的列维数（沿轴1的长度）必须与`x`的维数（其长度）相同

   ```python
   def pr(val):
       print(val)
   
   x = torch.arange(12,dtype=torch.float32).reshape(3,4)
   y = torch.ones(4,dtype=torch.float32)
   pr(x)
   pr(y)
   print(torch.mv(x,y))
   ```

   ```bash
   tensor([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])
   tensor([1., 1., 1., 1.])
   tensor([ 6., 22., 38.])
   ```

3. **矩阵乘法**

   ```python
   def pr(val):
       print(val)
   
   x = torch.arange(12,dtype=torch.float32).reshape(3,4)
   y = torch.ones(4,dtype=torch.float32).reshape(4,1)
   pr(x.shape)
   pr(y.shape)
   print(torch.mm(x,y))
   ```

   ```bash
   torch.Size([3, 4])
   torch.Size([4, 1])
   tensor([[ 6.],
           [22.],
           [38.]])
   ```

4. **范数**

   $L_2$范数也称欧几里得距离：假设 $n$维向量$\mathbf{x}$中的元素是$x_1,x_2,...,x_n$，其$L_2$范数是向量元素平方和的平方根
   $$
   ||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^n x_i^2},
   $$

   ```python
   u = torch.tensor([3.0, -4.0])
   pr(torch.norm(u))
   ```

   ```bash
   tensor(5.)
   ```

   深度学习中常用到 $L_2$范数和 $L_1$范数，$L_1$范数是向量各元素的绝对值之和：
   $$
   \|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.
   $$

   ```python
   torch.abs(u).sum()
   ```

   更一般的 $L_p$范数：
   $$
   \|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.
   $$
   在深度学习中，我们经常试图解决优化问题： *最大化*分配给观测数据的概率; *最小化*预测和真实观测之间的距离。 用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。 

   目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。

