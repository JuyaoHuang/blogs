---
title: Tensor与AutoGrad语法
author: Alen
published: 2025-10-10
description: "Tensor与AutoGrad的语法和操作"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['python']
draft: false
---

# Tensor 的语法和操作

本文档总结了 PyTorch 中 Tensor 的核心概念、创建方法、常用属性以及各种基本运算操作。
内容主要参考自 Gitee 上的 [PyTorch基础教程 - 张量与自动求导](https://gitee.com/flycity/ai_tutorial_book/blob/master/chap5-Pytorch%E5%9F%BA%E7%A1%80/5.2%20%E5%BC%A0%E9%87%8F%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.ipynb)。

## 1. Tensor (张量) 简介



Tensor 是 PyTorch 中最基本的数据结构，类似于 NumPy 的 `ndarray`，但增加了在 GPU 上进行计算以及自动求导等功能。它可以是标量（0维）、向量（1维）、矩阵（2维）或更高维度的数组。

## 2. 创建 Tensor

有多种方法可以创建 Tensor：

### 2.1. 从数据直接创建

*   `torch.tensor(data, dtype=None, device=None, requires_grad=False)`:     最常用的创建方式，会复制数据。
    
    ```python
    import torch
    import numpy as np
    
    # 从 Python list 创建
    data_list = [[1, 2], [3, 4]]
    t1 = torch.tensor(data_list, dtype=torch.float32)
    print(t1)
    
    # 从 NumPy array 创建
    arr = np.array([[5, 6], [7, 8]])
    t2 = torch.tensor(arr) # dtype 会从 numpy array 推断
    print(t2)
    
    # 指定设备
    if torch.cuda.is_available():
        t_gpu = torch.tensor([1,2,3], device="cuda")
        print(t_gpu)
    ```
    
*   `torch.Tensor(data)` 或 `torch.FloatTensor(data)` 等: 
    
    ​	这种方式可能会与 Python list 或 NumPy array 共享内存（取决于数据类型和内部机制，不推荐作为首选，`torch.tensor()` 更安全和可预测）。它还会使用全局默认的 `dtype`。
    
    ```python
    t3 = torch.Tensor([1, 2, 3]) # 使用全局默认 dtype (通常是 float32)
    print(t3.dtype)
    ```
    
*   `torch.as_tensor(data, ...)`: 
    
    ​	尽可能避免数据复制，如果 `data` 是 NumPy 数组且数据类型匹配，则会共享内存。
    
    ```python
    arr_share = np.array([10, 11], dtype=np.float32)
    t_share = torch.as_tensor(arr_share)
    arr_share[0] = 99
    print(t_share) # t_share 的值也会改变
    ```

### 2.2. 创建特定形状和数值的 Tensor

* `torch.empty(size)`: 

  ​	创建未初始化的 Tensor。

* `torch.zeros(size, dtype=None, ...)`: 

  ​	创建全0 Tensor。

* `torch.ones(size, dtype=None, ...)`: 

  ​	创建全1 Tensor。

*   `torch.full(size, fill_value, dtype=None, ...)`: 
    
    ​	创建指定填充值的 Tensor。
    
    ```python
    size = (2, 3)
    t_zeros = torch.zeros(size)
    t_ones = torch.ones(size, dtype=torch.int)
    t_full = torch.full(size, 7.7)
    print(t_zeros)
    print(t_ones)
    print(t_full)
    ```
    
*   `torch.zeros_like(input_tensor)` / `torch.ones_like(input_tensor)` / `torch.full_like(input_tensor, fill_value)`: 
    
    创建与 `input_tensor` 形状和设备相同，但数值不同的 Tensor。
    
    ```python
    t_like = torch.ones_like(t1) # t1 是前面创建的 [[1,2],[3,4]]
    print(t_like)
    ```

### 2.3. 创建序列 Tensor

* `torch.arange(start=0, end, step=1, ...)`: 

  类似 Python `range()`。

* `torch.linspace(start, end, steps=100, ...)`: 

  在 `start` 和 `end` 之间生成等间隔的 `steps` 个点。

*   `torch.logspace(start, end, steps=100, base=10.0, ...)`: 
    
    在对数空间生成。
    
    ```python
    t_arange = torch.arange(0, 5, step=1)
    t_linspace = torch.linspace(0, 1, steps=5)
    print(t_arange)
    print(t_linspace)
    ```

### 2.4. 创建随机 Tensor

* `torch.rand(size)`: 

  [0, 1) 均匀分布。

* `torch.randn(size)`: 

  标准正态分布（均值为0，方差为1）。

*   `torch.randint(low=0, high, size, ...)`: 
    
    在 `[low, high)` 区间内生成随机整数。
    
    ```python
    t_rand = torch.rand(2, 2)
    t_randn = torch.randn(2, 2)
    t_randint = torch.randint(0, 10, (2, 2))
    print(t_rand)
    print(t_randn)
    print(t_randint)
    ```

## 3. Tensor 属性

* `tensor.shape` 或 `tensor.size()`: 

  返回 Tensor 的形状 (一个 `torch.Size` 对象，类似元组)。

* `tensor.dtype`: 

  Tensor 中元素的数据类型 (如 `torch.float32`, `torch.int64`)。

* `tensor.device`: 

  Tensor 所在的设备 (如 `cpu`, `cuda:0`)。

* `tensor.ndim`: 

  Tensor 的维度数量。

* `tensor.numel()`: 

  Tensor 中元素的总数量。

*   `tensor.requires_grad`: 
    
    布尔值，指示是否需要为该 Tensor 计算梯度 (用于自动求导)。
    
    ```python
    a = torch.randn(3, 4, 5)
    print(f"Shape: {a.shape}")
    print(f"Dtype: {a.dtype}")
    print(f"Device: {a.device}")
    print(f"Ndim: {a.ndim}")
    print(f"Numel: {a.numel()}")
    print(f"Requires_grad: {a.requires_grad}")
    ```

## 4. Tensor 索引与切片

与 NumPy 类似，支持标准的索引和切片操作。

```python
t = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(t[0])          # 第0行: tensor([0, 1, 2, 3])
print(t[:, 1])       # 第1列: tensor([1, 5, 9])
print(t[1, 1:3])     # 第1行，第1到2列: tensor([5, 6])
print(t[0:2, 0:2])   # 子矩阵

# 布尔索引
mask = t > 5
print(t[mask])       # tensor([ 6,  7,  8,  9, 10, 11])

# 高级索引 (使用整数列表/Tensor 进行索引)
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 3])
print(t[rows, cols]) # tensor([1, 11]) (t[0,1] 和 t[2,3])

# ... (省略号)
print(t[..., 1])     # 取所有行的第1列

```

特定索引函数：

- torch.index_select(input, dim, index): 

  沿指定维度 dim 选择 index 指定的行/列。

- torch.masked_select(input, mask): 

  使用布尔 mask 选择元素，返回一个1D Tensor。

- torch.where(condition, x, y): 

  根据 condition 选择 x 或 y 中的元素。

## 5. Tensor 形状修改

- tensor.view(*shape): 

  返回具有新形状的 Tensor，**共享原始数据** (如果可能)。新旧形状的总元素数必须相同。要求 Tensor 是连续的 (contiguous)。

- tensor.reshape(*shape): 

  功能类似 view，但更灵活，如果 view 无法操作（如 Tensor 不连续），reshape 可能会创建副本。通常推荐使用 reshape。

- tensor.resize_(*shape):

  **In-place** 修改 Tensor 形状。如果新形状元素更多，则未初始化；如果更少，则截断。**慎用**。

- tensor.squeeze(dim=None): 

  移除所有大小为1的维度。若指定 dim，则只移除该维度（如果其大小为1）。

- tensor.unsqueeze(dim): 

  在指定位置 dim 增加一个大小为1的维度。

- tensor.permute(*dims): 

  重新排列 Tensor 的维度。

- tensor.transpose(dim0, dim1): 

  交换指定的两个维度。

- tensor.flatten(start_dim=0, end_dim=-1): 

  将指定范围内的维度展平成一维。
  
  ```py
  t = torch.arange(12)
  t_reshaped = t.reshape(3, 4)
  print(t_reshaped.shape)
  
  t_view = t_reshaped.view(2, 6) # 共享数据
  print(t_view.shape)
  
  a = torch.rand(1, 2, 1, 3)
  print(a.squeeze().shape)      # torch.Size([2, 3])
  print(a.squeeze(dim=0).shape) # torch.Size([2, 1, 3])
  
  b = torch.rand(2, 3)
  print(b.unsqueeze(0).shape)   # torch.Size([1, 2, 3])
  print(b.unsqueeze(1).shape)   # torch.Size([2, 1, 3])
  
  c = torch.rand(2,3,4)
  print(c.permute(2,0,1).shape) # torch.Size([4,2,3])
  print(c.transpose(0,1).shape) # torch.Size([3,2,4])
      
  ```

## 6. Tensor 运算

### 6.1. 逐元素运算 (Element-wise Operations)

这些操作对 Tensor 中的每个元素独立进行。

#### 6.1.1. 基本算术运算

*   **加法**:     `a + b` 或 `torch.add(a, b, out=None)`
*   **减法**:     `a - b` 或 `torch.sub(a, b, out=None)` 或 `torch.subtract(a, b, out=None)`
*   **乘法**:     `a * b` 或 `torch.mul(a, b, out=None)` 或 `torch.multiply(a, b, out=None)`
*   **除法**:     `a / b` 或 `torch.div(a, b, out=None)` 或 `torch.divide(a, b, out=None)`
    *   `torch.true_divide(a, b)`:     总是执行浮点除法。
*   **地板除 (整除)**:     `a // b` (注意：PyTorch 中 `//` 对于浮点数可能行为与 Python 不同，推荐使用 `torch.floor_divide`)
    *   `torch.floor_divide(a, b)`:     计算 `floor(a/b)`。
*   **取模 (求余)**:     `a % b` 或 `torch.fmod(a, b)` (浮点数余数) 或 `torch.remainder(a, b)` (与 Python `%` 行为一致)。
*   **幂运算**:      `a ** b` 或 `torch.pow(a, b)`

```python
import torch
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 0.5, 2.])
scalar = 2

print(f"a + b: {a + b}")              # tensor([5.0000, 2.5000, 5.0000])
print(f"a - scalar: {a - scalar}")      # tensor([-1.,  0.,  1.])
print(f"a * b: {a * b}")              # tensor([4., 1., 6.])
print(f"a / b: {a / b}")              # tensor([0.2500, 4.0000, 1.5000])
print(f"torch.floor_divide(a,b): {torch.floor_divide(a,b)}") # tensor([0., 4., 1.])
print(f"a % 2: {a % 2}")              # tensor([1., 0., 1.])
print(f"a ** 2: {a ** 2}")            # tensor([1., 4., 9.])
print(f"torch.pow(a, b): {torch.pow(a,b)}") # tensor([1.0000, 1.4142, 9.0000])
```

#### 6.1.2. 指数与对数运算

- **指数**:
  - torch.exp(input): 
  
    计算 e 的 input 次幂 (e^x)。
  
  - torch.expm1(input): 
  
    计算 e^x - 1，对于接近0的 x 更精确。
  
- **对数**:
  - torch.log(input): 
  
    自然对数 (ln(x))。
  
  - torch.log10(input): 
  
    以10为底的对数。
  
  - torch.log2(input): 
  
    以2为底的对数。
  
  - torch.log1p(input): 
  
    计算 ln(1+x)，对于接近0的 x 更精确。
  
- **其他**:
  - torch.sqrt(input): 
  
    平方根。
  
  - torch.rsqrt(input): 
  
    平方根的倒数 (1/sqrt(x))。

```py
x = torch.tensor([1., 2., 0.01])
print(f"torch.exp(x): {torch.exp(x)}")     # tensor([2.7183, 7.3891, 1.0101])
print(f"torch.log(x): {torch.log(x)}")     # tensor([0.0000, 0.6931, -4.6052])
print(f"torch.sqrt(x): {torch.sqrt(x)}")    # tensor([1.0000, 1.4142, 0.1000])
```

#### 6.1.3. 三角函数

- torch.sin(input),     torch.cos(input),     torch.tan(input)

- torch.asin(input),     torch.acos(input),      torch.atan(input)

- torch.atan2(y, x): 

  计算 atan(y/x)，并根据 y 和 x 的符号确定象限。

- torch.sinh(input),     torch.cosh(input),     torch.tanh(input) (双曲函数)

```py
angles = torch.tensor([0, torch.pi/2, torch.pi])
print(f"torch.sin(angles): {torch.sin(angles)}") # tensor([0.0000e+00, 1.0000e+00, -8.7423e-08]) (接近0)
print(f"torch.tanh(x): {torch.tanh(x)}")       # tensor([0.7616, 0.9640, 0.0100])
```

#### 6.1.4. 取整与符号函数

- torch.round(input):     四舍五入到最近的整数。
- torch.floor(input):     向下取整。
- torch.ceil(input):     向上取整。
- torch.trunc(input):     截断，去除小数部分 (向零取整)。
- torch.frac(input):     返回小数部分。
- torch.abs(input) 或 torch.absolute(input):      绝对值。
- torch.sign(input):     符号函数 (-1 if x < 0, 0 if x == 0, 1 if x > 0)。
- torch.sgn(input):     复数符号函数。

```py
vals = torch.tensor([-1.7, -0.2, 0.0, 1.3, 2.8])
print(f"torch.round(vals): {torch.round(vals)}") # tensor([-2., -0.,  0.,  1.,  3.])
print(f"torch.floor(vals): {torch.floor(vals)}") # tensor([-2., -1.,  0.,  1.,  2.])
print(f"torch.ceil(vals): {torch.ceil(vals)}")   # tensor([-1., -0.,  0.,  2.,  3.])
print(f"torch.trunc(vals): {torch.trunc(vals)}") # tensor([-1., -0.,  0.,  1.,  2.])
print(f"torch.abs(vals): {torch.abs(vals)}")   # tensor([1.7000, 0.2000, 0.0000, 1.3000, 2.8000])
print(f"torch.sign(vals): {torch.sign(vals)}") # tensor([-1., -1.,  0.,  1.,  1.])
```

#### 6.1.5. 其他常用逐元素函数

- torch.clamp(input, min=None, max=None):      将 input 中的元素限制在 [min, max] 区间内。
- torch.neg(input):     取负值 (-input)。
- torch.reciprocal(input):      取倒数 (1/input)。
- torch.sigmoid(input):     Sigmoid 函数。
- torch.erf(input), torch.erfc(input):     误差函数及其互补误差函数。

#### 6.1.6. In-place 操作

许多逐元素操作都有一个带下划线的 in-place 版本，会直接修改原 Tensor，例如：

- a.add_(b)
- a.mul_(scalar)
- a.exp_()
- a.round_()

```py
t_inplace = torch.tensor([1.1, 2.9])
t_inplace.add_(1)      # t_inplace is now tensor([2.1000, 3.9000])
t_inplace.round_()     # t_inplace is now tensor([2., 4.])
print(t_inplace)
```

**注意**:   In-place 操作会影响所有共享该 Tensor 数据的视图 (views)，并且在需要计算梯度时可能会引发问题 (因为会修改用于反向传播的原始值)，通常不建议在计算图中广泛使用。

### 6.2. 归并/聚合运算 (Reduction)

这些操作会沿着一个或多个维度减少 Tensor 的维度。

- torch.sum(input, dim=None, keepdim=False)
- torch.mean(input, dim=None, keepdim=False)
- torch.prod(input, dim=None, keepdim=False) (连乘)
- torch.max(input, dim=None, keepdim=False) (返回 (values, indices) 对，如果指定 dim)
- torch.min(input, dim=None, keepdim=False) (同上)
- torch.argmax(input, dim=None, keepdim=False) (返回最大值索引)
- torch.argmin(input, dim=None, keepdim=False) (返回最小值索引)
- torch.std(input, dim=None, unbiased=True, keepdim=False) (标准差)
- torch.var(input, dim=None, unbiased=True, keepdim=False) (方差)
- torch.median(input, dim=None, keepdim=False) (中位数)
- **torch.norm(input, p='fro', dim=None, keepdim=False) (范数)**

```py
t = torch.arange(1., 7.).reshape(2, 3)
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
print(torch.sum(t))             # tensor(21.) (所有元素求和)
print(torch.sum(t, dim=0))      # tensor([5., 7., 9.]) (按列求和)
print(torch.sum(t, dim=1))      # tensor([ 6., 15.]) (按行求和)
print(torch.mean(t, dim=0, keepdim=True)) # tensor([[2.5000, 3.5000, 4.5000]])
values, indices = torch.max(t, dim=1)
print(f"Max values: {values}, Max indices: {indices}")
```

### 6.3. 比较运算

返回布尔类型的 Tensor。

- \>,     <,     >=,     <=,      ==,     !=
- torch.eq(),     torch.gt(),     torch.lt(),     torch.ge(),     torch.le(),     torch.ne()
- torch.equal(tensor1, tensor2):     判断两个 Tensor 是否所有元素都相等。
- torch.all(input):     判断 Tensor 中所有元素是否都为 True。
- torch.any(input):     判断 Tensor 中是否有元素为 True。
- torch.isfinite(),     torch.isinf(),     torch.isnan()

```py
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 0], [5, 4]])
print(a == b)
# tensor([[ True, False],
#         [False,  True]])
print(torch.equal(a, a)) # True
  
```

### 6.4. 矩阵运算与线性代数

#### 6.4.1. 转置 (Transpose)

- tensor.t():     对于 2D Tensor，计算其转置。

- tensor.transpose(dim0, dim1):     交换指定的两个维度。对于任意维度 Tensor 都适用。

- tensor.permute(*dims):     重新排列 Tensor 的维度，提供更灵活的维度交换。
  - tensor.T 属性:  
  
    对于 >= 2D 的 Tensor，是 tensor.permute(*range(tensor.ndim - 1, -1, -1)) 的快捷方式，即反转所有维度。对于 1D Tensor，返回其本身；对于 2D Tensor，等同于 .t()。

```py
mat_2d = torch.arange(6).reshape(2, 3)
# tensor([[0, 1, 2],
#         [3, 4, 5]])
print(f"mat_2d.t():\n{mat_2d.t()}")
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])

tensor_3d = torch.arange(24).reshape(2, 3, 4)
print(f"tensor_3d.transpose(0, 1).shape: {tensor_3d.transpose(0, 1).shape}") # torch.Size([3, 2, 4])
print(f"tensor_3d.permute(2, 0, 1).shape: {tensor_3d.permute(2, 0, 1).shape}") # torch.Size([4, 2, 3])
print(f"tensor_3d.T.shape: {tensor_3d.T.shape}") # torch.Size([4, 3, 2]) (反转了所有维度)
```

#### 6.4.2. 乘法 (Multiplication)

- **逐元素乘法**:     a * b 或 torch.mul(a, b) (前面已述)。

- **点积 (Dot Product / Inner Product) for 1D Tensors**:
  
  - torch.dot(tensor1, tensor2):     要求 tensor1 和 tensor2 都是 1D 且元素数量相同。返回一个标量。
  
- **矩阵-向量乘法 (Matrix-Vector Multiplication)**:
  - torch.mv(matrix, vector):     matrix (2D) 乘以 vector (1D)。
  
- **矩阵-矩阵乘法 (Matrix-Matrix Multiplication)**:
  - torch.mm(mat1, mat2):     严格的 2D 矩阵乘法 (mat1 (m*n), mat2 (n*p) -> (m*p))。不支持广播。
  
  - torch.matmul(input, other) 或 @ 运算符:     更通用的矩阵乘法，支持广播，可以处理更高维度的 Tensor。其行为根据输入 Tensor 的维度有所不同：
    - 如果都是 1D: 计算点积。
    - 如果都是 2D: 计算矩阵乘法 (同 torch.mm)。
    - 如果 input 是 1D，other 是 2D: input 前面补一个维度 (1*n)，进行矩阵乘法，然后移除添加的维度。
    - 如果 input 是 2D，other 是 1D: other 后面补一个维度 (p*1)，进行矩阵乘法，然后移除添加的维度。
    - 如果维度 > 2D (批处理): 将前 N-2 维视为批次维度，对最后两个维度进行矩阵乘法。批次维度需要兼容广播。
    
  - torch.bmm(input, other): 
  
    批处理矩阵乘法。
  
    要求 input 和 other 都是 3D，且第一个维度（批次大小）相同。对每个批次内的 2D 矩阵进行乘法。

```py
v1 = torch.tensor([1., 2., 3.])
v2 = torch.tensor([4., 5., 6.])
print(f"torch.dot(v1, v2): {torch.dot(v1, v2)}") # tensor(32.)

mat_A = torch.randn(2, 3)
vec_x = torch.randn(3)
print(f"torch.mv(mat_A, vec_x).shape: {torch.mv(mat_A, vec_x).shape}") # torch.Size([2])

mat_B = torch.randn(3, 4)
print(f"torch.mm(mat_A, mat_B).shape: {torch.mm(mat_A, mat_B).shape}") # torch.Size([2, 4])
print(f"(mat_A @ mat_B).shape: { (mat_A @ mat_B).shape }")      # torch.Size([2, 4])

batch_mat1 = torch.randn(10, 2, 3) # 10个 2x3 矩阵
batch_mat2 = torch.randn(10, 3, 4) # 10个 3x4 矩阵
print(f"torch.bmm(batch_mat1, batch_mat2).shape: {torch.bmm(batch_mat1, batch_mat2).shape}") # torch.Size([10, 2, 4])
```

#### 6.4.3. 其他线性代数运算

- torch.inverse(input):     计算方阵的逆。
- torch.det(input):     计算方阵的行列式。
- torch.slogdet(input):     计算方阵行列式的符号和对数值 (sign, logabsdet)。
- torch.trace(input):     计算 2D 方阵的迹 (主对角线元素之和)。
- torch.diag(input, diagonal=0):
  - 如果 input 是 1D Tensor，返回一个以 input 为主对角线（或指定对角线）的 2D 方阵。
  - 如果 input 是 2D Tensor，返回其主对角线（或指定对角线）元素，作为 1D Tensor。
- **torch.linalg 模块**: 提供了更全面的线性代数函数，如：
  - **torch.linalg.svd(A)**:     奇异值分解。
  - **torch.linalg.qr(A)**:     QR 分解。
  - **torch.linalg.eig(A) / torch.linalg.eigh(A):     特征值/特征向量 (后者用于对称/厄米矩阵)。**
  - torch.linalg.solve(A, B):     解线性方程组 Ax = B。
  - **torch.linalg.matrix_norm(A, ord=...):**     矩阵范数。
  - **torch.linalg.cond(A, p=...):     条件数。**

```py
square_mat = torch.tensor([[1., 2.], [3., 4.]])
print(f"torch.inverse(square_mat):\n{torch.inverse(square_mat)}")
print(f"torch.det(square_mat): {torch.det(square_mat)}")       # tensor(-2.)
print(f"torch.trace(square_mat): {torch.trace(square_mat)}")     # tensor(5.)

diag_elements = torch.tensor([1, 2, 3])
print(f"torch.diag(diag_elements):\n{torch.diag(diag_elements)}")
# tensor([[1, 0, 0],
#         [0, 2, 0],
#         [0, 0, 3]])
```

## 7. 广播机制

与 NumPy 类似，PyTorch 也支持广播机制，使得不同形状的 Tensor 在进行运算时能够自动扩展，以匹配彼此的形状。

规则：

1. 如果两个 Tensor 的维度数不同，在维度较少的 Tensor 的前面补1，直到维度数相同。
2. 对于每个维度，如果两个 Tensor 在该维度上的大小相同，或者其中一个 Tensor 在该维度上的大小为1，则它们在该维度上是兼容的。
3. 如果所有维度都兼容，则可以进行广播。
4. 运算后结果的形状是每个维度上取两者中较大的那个尺寸。
5. 大小为1的维度会沿着该维度扩展以匹配另一个 Tensor 的大小。

```py
a = torch.arange(3).reshape(3, 1) # shape (3, 1)
# tensor([[0],
#         [1],
#         [2]])
b = torch.arange(2).reshape(1, 2) # shape (1, 2)
# tensor([[0, 1]])

c = a + b # a 扩展为 (3,2), b 扩展为 (3,2)
print(c)
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
print(c.shape) # torch.Size([3, 2])
    
```

## 8. Tensor 与 NumPy 转换

- tensor.numpy():     将 CPU 上的 Tensor 转换为 NumPy ndarray。两者共享内存，修改一方会影响另一方。
- torch.from_numpy(ndarray):     将 NumPy ndarray 转换为 Tensor。两者共享内存。

```py
# Tensor to NumPy
cpu_tensor = torch.ones(5)
numpy_array = cpu_tensor.numpy()
cpu_tensor.add_(1) # In-place modification
print(cpu_tensor)  # tensor([2., 2., 2., 2., 2.])
print(numpy_array) # [2. 2. 2. 2. 2.]

# NumPy to Tensor
arr = np.ones(5)
tensor_from_numpy = torch.from_numpy(arr)
np.add(arr, 1, out=arr) # In-place modification of numpy array
print(arr)                # [2. 2. 2. 2. 2.]
print(tensor_from_numpy)  # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

# 如果 Tensor 在 GPU 上，需要先移到 CPU
if torch.cuda.is_available():
    gpu_tensor = torch.rand(2, device="cuda")
    numpy_from_gpu = gpu_tensor.cpu().numpy() # .cpu() 返回一个 CPU 上的副本
    
```

## 9. Tensor 设备间的移动

- tensor.to(device_str_or_torch_device_obj):     最通用的移动方法。
- tensor.cpu():     移动/复制到 CPU。
- tensor.cuda(device_id=None):     移动/复制到指定的 GPU (或默认 GPU)。

```python
x = torch.randn(2,2)
print(x.device) # cpu

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device) # 直接在 GPU 上创建
    x = x.to(device)                      # 将 x 移动到 GPU
    z = x + y
    print(z)
    print(z.device)
    # 将结果移回 CPU
    z_cpu = z.cpu()
    print(z_cpu.device)
```



------


