---
title: NumPy
author: Alen
published: 2025-10-12
description: "核心数据分析工具：NumPy介绍"
first_level_category: "AI"
second_level_category: "机器学习"
tags: ['机器学习','numpy']
draft: false
---

NumPy 是 Python 科学计算生态系统的基石，几乎所有高级数据分析和机器学习库（如 Pandas, Scikit-learn, TensorFlow, PyTorch）都构建于其上或与之深度集成。

### 什么是 NumPy？

NumPy 的全称是 **Numerical Python**，即“数值 Python”。它是一个开源的 Python 库，主要功能如下：

1. **提供强大的 N 维数组对象 (ndarray)**：这是 NumPy 的核心。它是一个多维的、由**相同类型**元素组成的数组。
2. **提供复杂的广播函数**：允许对不同形状的数组进行数学运算。
3. **提供丰富的数学函数库**：包含了大量用于操作数组的数学、逻辑、**线性代数、傅里叶变换**、随机数生成等函数。
4. **高性能**：NumPy 的底层代码由 C 和 Fortran 编写，这使得其数组操作和数学运算的速度远超原生的 Python 列表。

**为什么不用 Python 列表？**

| 特性         | Python 列表 (list)                               | NumPy 数组 (ndarray)                                        |
| ------------ | ------------------------------------------------ | ----------------------------------------------------------- |
| **元素类型** | 可以存储不同类型的元素（如整数、字符串、对象）   | 只能存储相同类型的元素（如全是 int32 或全是 float64）       |
| **性能**     | 运算速度慢，因为需要类型检查且元素在内存中不连续 | 运算速度极快，无类型检查，元素在内存中连续存储，利于CPU缓存 |
| **内存**     | 存储元素本身和其类型信息，内存开销大             | 只存储元素，内存开销小                                      |
| **数学运算** | 不支持对整个列表进行向量化数学运算（需要写循环） | 支持对整个数组进行高效的向量化运算                          |

------

### NumPy 的核心

#### 1. 创建数组

首先，你需要导入 NumPy 库，通常我们将其简写为 np。

```python
import numpy as np
```

**a. 从 Python 列表创建**

这是最基本的方式。

```Python
# 一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # 输出: [1 2 3 4 5]

# 二维数组（矩阵）
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# 输出:
# [[1 2 3]
#  [4 5 6]]
  
```

**b. 使用内置函数创建**

NumPy 提供了很多便捷的函数来创建特定类型的数组。

```Python
# 创建一个从0到9的数组
# arange 函数用于创建元素值连续整数的数组
arr_arange = np.arange(10)
print(arr_arange)  # 输出: [0 1 2 3 4 5 6 7 8 9]

# 创建一个全为0的3x4数组
arr_zeros = np.zeros((3, 4))
print(arr_zeros)

# 创建一个全为1的2x3数组
arr_ones = np.ones((2, 3))
print(arr_ones)

# 创建一个3x3的单位矩阵
arr_eye = np.eye(3)
print(arr_eye)

# 创建一个在[0, 10)区间内，有5个等间距点的数组
# linspace 函数用于创建在指定区间内均匀分布的数据，默认 dtype 为 float64
arr_linspace = np.linspace(0, 10, 5)
print(arr_linspace) # 输出: [ 0.   2.5  5.   7.5 10. ]

# 创建指定形状的随机数数组（0到1之间）
arr_rand = np.random.rand(2, 3)
print(arr_rand)

# 创建指定形状的标准正态分布随机数数组
arr_randn = np.random.randn(2, 3)
print(arr_randn)
  
```

**c.随机数据生成**

numpy 中的 random 模块提供多种函数，可以生成各种随机数据

1. ```py
   from numpy import random
   random.rand() #生成[0,1)区间的 float 类型随机数
   ```

2. ```py
   random.rand(2,3) #生成 2 行 3 列的随机浮点数二维数组，注意参数的写法
   ```

3. ```py
   random.random((2,3)) #生成 2 行 3 列的随机浮点数二维数组，注意参数的写法（0到1之间）
   ```

4. ```python
   random.randn(2,3) #生成 2 行 3 列的随机浮点数二维数组，注意元素值的分布为标准高斯分布
   ```

5. ```python
   random.randint(13) #生成一个值在[0,13)区间的随机整数
   random.randint(10, size=(2,3)) #生成 2 行 3 列的随机整数二维数组
   ```

6. ```py
   random.normal(10,3.5, size=(2,3)) #生成 2 行 3 列的二维数组，注意元素值的分布
   ```

7. Numpy 中使用 np.random.seed()函数设置随机数的种子，对于同一代码， 若设置随机数种子相同，则无论在什么环境下何时执行，得到的随机序列都是相同的。

   ```py
   np.random.seed(0) #设置随机数种子为 0
   ```

8. 可以使用 python 提供的 dir 和 help 函数， 查看 python 包、模块或类中所有的成员属性和函数，以及相应的官方提供的使用帮助

   ```py
   print(dir(np.random)) #查看 random 提供的所有属性和函数
   help(np.random) #查看 random 包的帮助说明
   help(random.random_integers) #查看 random.random_integets 函数的帮助说明
   ```

   ​	

#### 2. 数组的属性

了解数组的属性对于操作数组至关重要。

```Python
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print("数组维度:", arr.ndim)     # 输出: 2
print("数组形状:", arr.shape)    # 输出: (2, 3) -> 2行3列
print("数组总元素数:", arr.size) # 输出: 6
print("元素数据类型:", arr.dtype) # 输出: float64
  
```

- ndim：数组的维数（也叫轴 axis 的数量）。
- shape：一个元组，表示数组在每个维度上的大小。这是**最重要**的属性。
- size：数组中元素的总数。
- dtype：数组元素的数据类型，如 int32, float64 等。

------



### 数组的索引与切片

这是 NumPy 最强大、最灵活的功能之一。

#### 1. 基本索引与切片

和 Python 列表类似，但功能更强。

```Python
arr = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]

# 获取单个元素
print(arr[5])  # 输出: 5

# 切片获取一个子数组
print(arr[2:6]) # 输出: [2 3 4 5]

# 切片并赋值
arr[2:6] = 99
print(arr) # 输出: [ 0  1 99 99 99 99  6  7  8  9]
  
```

**重要提示：数组切片是原始数组的“视图” (View)**

这意味着修改切片会直接影响到原始数组！这与 Python 列表的行为不同，这样做是为了性能和节省内存。

```Python
arr_slice = arr[2:6:2] # 步长为 2
arr_slice[0] = 1000
print(arr) # 输出: [   0    1 1000   99   99   99    6    7    8    9] -> 原始数组被修改了！

# 如果需要副本(Copy)，而不是视图，需要显式复制
arr_copy = arr[2:6].copy()
arr_copy[0] = 0
print(arr) # 原始数组不会被修改
  
```

**多维数组的索引**

```Python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 获取单个元素 (推荐使用逗号)
print(arr2d[1, 2]) # 第1行，第2列（从0开始计数），输出: 6

# 获取一行
print(arr2d[1]) # 输出: [4 5 6]

# 多维数组切片
# 获取前2行，和1-2列
sub_arr = arr2d[:2, 1:3]
print(sub_arr)
# 输出:
# [[2 3]
#  [5 6]]
  
```



#### 2. 布尔索引 (Boolean Indexing)

这是进行数据筛选和过滤的利器。

```Python
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
data = np.random.randn(4, 3) # 4x3 的随机数据

print(data)
# 假设输出:
# [[-0.41 -0.57  1.22]
#  [ 0.38 -0.39 -0.61]
#  [-0.3  -1.12  0.4 ]
#  [ 0.81 -0.19  1.29]]

# 找出所有 'Bob' 对应的数据行
print(names == 'Bob') # 输出: [ True False False  True]
print(data[names == 'Bob'])
# 输出:
# [[-0.41 -0.57  1.22]
#  [ 0.81 -0.19  1.29]]

# 也可以组合条件
# 找出 'Bob' 对应的数据行，并且只看第1列
print(data[names == 'Bob', 0]) # 输出: [-0.41  0.81]

# 使用 & (与) | (或) 组合多个布尔条件
mask = (names == 'Bob') | (names == 'Will')
print(data[mask])

# 将data中小于0的数都设为0
data[data < 0] = 0
print(data)
  
```

### 向量化计算与广播

#### 1. 向量化计算

这是 NumPy 高性能的核心。简单来说，就是对**整个数组执行操作**，而不需要编写显式的 for 循环。

1. **基础算术运算**：

   - np.add(x1, x2):                  元素级加法 (相当于 x1 + x2)
   - np.subtract(x1, x2):          元素级减法 (相当于 x1 - x2)
   - np.multiply(x1, x2):          元素级乘法 (相当于 x1 * x2)--**数组乘法，与严格的矩阵乘法 @不同，有广播机制**
   - np.divide(x1, x2):              元素级浮点除法 (相当于 x1 / x2)
   - np.floor_divide(x1, x2):    元素级整除 (相当于 x1 // x2)
   - np.power(x1, x2):              元素级指数运算 (相当于 x1 ** x2)
   - np.sqrt(x):                           计算每个元素的平方根
   - np.abs(x) 或 np.absolute(x):             计算每个元素的绝对值
   - np.negative(x):                     取每个元素的负数 (相当于 -x)

2. **三角函数**：

      输入参数默认为**弧度**

   - np.sin(x), np.cos(x), np.tan(x):                     标准三角函数
   - np.arcsin(x), np.arccos(x), np.arctan(x):    反三角函数
   - np.degrees(x):                                               将弧度转换为角度
   - np.radians(x):                                                将角度转换为弧度
   - np.hypot(x1, x2):                                          计算斜边 $\sqrt{x_1^2+x_2^2}$

3. **指数和对数**

   - np.exp(x):       计算每个元素的自然指数 $e^x$
   - np.log(x):        计算每个元素的自然对数 $ln(x)$
   - np.log2(x):      计算每个元素的以2为底的对数 $log_2(x)$
   - np.log10(x):    计算每个元素的以10为底的对数

4. **舍入与取余**

   - np.round(x, decimals=0):        四舍五入到指定小数位数
   - np.floor(x):                                 向下取整，返回不大于输入参数的最大整数
   - np.ceil(x):                                    向上取整，返回不小于输入参数的最小整数
   - np.trunc(x):                                截断小数部分，只保留整数部分
   - np.mod(x1, x2) 或 np.remainder(x1, x2):               元素级求余数 (相当于 x1 % x2)

5. **统计函数**

   - np.sum(a, axis=None):                          计算数组元素的和

   - np.prod(a, axis=None):                         计算数组元素的积

   - np.mean(a, axis=None):                        计算算术平均值

   - np.median(a, axis=None):                     计算中位数

   - np.std(a, axis=None):                             计算标准差

   - np.var(a, axis=None):                             计算方差

   - np.min(a, axis=None):                            找出最小值

   - np.max(a, axis=None):                           找出最大值

   - np.argmin(a, axis=None):                      找出最小值的索引

   - np.argmax(a, axis=None):                     找出最大值的索引

   - np.percentile(a, q, axis=None):             计算q-百分位数

   - np.cumsum(a, axis=None):                    计算元素的累积和

   - np.cumprod(a, axis=None):                    计算元素的累积积

   - 参数 axis：

     - axis=None (默认): 对整个数组进行计算。

     - axis=0: 沿**列**的方向进行计算。

     - axis=1: 沿**行**的方向进行计算。

```Python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 逐元素运算
print(arr * 2)      # 数组每个元素都乘以2
print(arr + arr)    # 两个数组对应元素相加	
print(1 / arr)      # 数组每个元素求倒数
print(arr ** 0.5)   # 数组每个元素开平方
  
```

#### 2. 通用函数

NumPy 提供了大量对 ndarray 进行逐元素操作的函数。

```Python
arr = np.arange(5)

print(np.sqrt(arr))   # 计算平方根
print(np.exp(arr))    # 计算指数 e^x
print(np.sin(arr))    # 计算正弦
```

#### 3. 广播

这是 NumPy 一个非常强大的机制，它允许在形状不同的数组之间执行数学运算。

**广播规则：**
当操作两个数组时，NumPy会逐个比较它们的维度（**从后往前**）。

1. 如果两个数组的维度数不同，那么在维度较小的数组的形状前面补1。
2. 在任何一个维度上，**如果两个数组的该维度大小相同，或者其中一个为1**，那么它们在该维度上是兼容的。
3. 如果两个数组在所有维度上都是兼容的，它们就可以被广播。
4. 广播后，每个数组的行为都好像它在该维度上的形状等于两个输入数组在该维度上形状的最大值。
5. **广播的比较规则**：
   1. 规则 0：从右向左对齐
      NumPy 会将两个形状元组的**末尾对齐**，然后从右到左逐个比较维度的大小。
   2. 规则 1：维度数量的比较 (处理维度不等长)
      如果在对齐后，一个数组的维度比另一个少（例如，比较 (3, 4) 和 (4,)），NumPy 会在**维度较少的那个数组的形状元组的左边补 1**，直到它们的维度数量相同。
      - A.shape = (3, 4) (2个维度)
      - B.shape = (4,) (1个维度)
      - 补齐后，B 的形状被当作 (1, 4) 来参与比较。
   3. 规则 2：逐个维度的兼容性检查
      在维度数量对齐后，NumPy 从右到左逐个比较每个维度的大小。对于每一对被比较的维度，必须满足以下**两个条件之一**才算兼容：
      1. 两个维度的大小完全相等。
      2. 其中一个维度的大小是 1。
   4. **从右向左，逐个比较，要么相等，要么其中一个是1**
6. **"最后一个维度"** ：
   1. 在 NumPy 中，一个数组的形状 (shape) 是一个元组，比如 (3, 4) 或 (5, 3, 4)
   2. **"最后一个维度" 就是这个元组中最右边的那个数字**
   3. 对于形状为 (3, 4) 的二维数组（矩阵），它有两个维度：
      - 第一个维度（行）：大小为 3
      - **最后一个维度（列）**：大小为 4
7. **如果维度不同，可以使用reshape函数显示转换为相同的维度**

**示例：**
一个二维数组（形状为 (3, 4)）减去一个一维数组（形状为 (4,) ）。

```Python
arr = np.arange(12).reshape((3, 4))
# arr is:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

mean_row = arr.mean(axis=0) # 计算每列的均值，形状为 (4,)
# mean_row is: [4. 5. 6. 7.]

# 广播机制生效
# arr(3, 4) - mean_row(4,)
# NumPy 自动将 mean_row "拉伸" 成 (3, 4) 的形状
# [[4. 5. 6. 7.]
#  [4. 5. 6. 7.]
#  [4. 5. 6. 7.]]
# 然后再执行逐元素相减
demeaned_arr = arr - mean_row
print(demeaned_arr)
  
```

广播使得我们无需创建 mean_row 的显式副本，非常高效。

### 高级计算

#### 1. 线性代数运算

处理向量和矩阵的数学运算，这些运算在 `np.linalg` 模块中。

- np.dot(a, b):                             矩阵/向量点积

- np.matmul(a, b):                     矩阵乘法 (在Python 3.5+中，更推荐使用 @ 运算符, a @ b)

  注意：当使用 @进行矩阵乘法时，如果其中一个操作数是 2D 矩阵 (M x N)，另一个是 1D 数组 (长度为 N)，NumPy 会自动将这个 1D 数组**当作一个列向量 (N x 1)** 来进行计算。

  1. numpy发现第二个矩阵是一个一维数组$1*N$，会自动转为$N*1$的列矩阵，但是由于输入 arr1 是一个 1D 数组，所以最终输出是一个一维数组$[val1,val2]$，而不是

     ```bash
     [[val1]
      [val2]]
     ```

- np.transpose(a) 或 a.T:          矩阵转置

- np.linalg.inv(a):                       计算矩阵的逆

- np.linalg.det(a):                       计算矩阵的行列式

- np.linalg.eig(a):                        计算矩阵的特征值和特征向量

- np.linalg.solve(a, b):                求解线性方程组 ax = b

**示例**

假设我们有方程组：
$$
2x + y = 8\\
x + 3y = 11
$$

这可以表示为矩阵形式 Ax = b：
$$
A = [[2, 1], [1, 3]], \\ b = [8, 11]
$$

```python
import numpy as np

A = np.array([[2, 1], [1, 3]])
b = np.array([8, 11])

# 使用 NumPy 求解 x
x = np.linalg.solve(A, b)

print(f"解为 x = {x[0]}, y = {x[1]}") # 输出: 解为 x = 2.6, y = 2.8
# 验证: 2*2.6 + 2.8 = 8.0,  2.6 + 3*2.8 = 11.0
```

#### 2. 傅里叶变换与信号处理

信号从时域（或空间域）转换到频域，看到信号由哪些频率的波组成，这些运算在 `np.fft` 模块中。

**对于更高级的信号处理库，搜索 SciPy**，`scipy.signal` 模块专门用于处理信号，并且内置了常见的函数和功能：

- 内置信号生成器：如冲激、阶跃、方波、锯齿波等。
- 窗口函数：如汉明窗、汉宁窗等。
- 滤波器设计：可以轻松设计和应用各种数字滤波器。
- 卷积、相关性计算等高级功能。

**示例**：找到某一信号的主频率

```python
# 创建一个频率为 3 Hz 的正弦波信号
sampling_rate = 100 # 采样率
t = np.arange(0, 1, 1/sampling_rate) # 1秒的时间
signal = np.sin(3 * 2 * np.pi * t)

# 进行傅里叶变换
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)

# 找到信号最强的频率
strongest_freq_index = np.argmax(np.abs(fft_result))
strongest_freq = frequencies[strongest_freq_index]

print(f"信号的主频率大约是: {abs(strongest_freq):.1f} Hz") # 输出: 信号的主频率大约是: 3.0 Hz

```

#### 3. 复杂的统计与随机数生成

生成满足特定分布（如多元正态分布）的随机数，进行蒙特卡洛模拟等。

- np.random.multivariate_normal()：生成多元正态分布（高斯分布）的随机数，可以指定变量之间的相关性。
- np.corrcoef()：计算相关系数矩阵。
- np.histogram()：高效地计算直方图。

**应用场景**：

- 金融建模：模拟股票价格的随机游走。
- 物理学：模拟粒子运动。
- 机器学习：创建带有特定统计属性的合成数据集来测试算法。

```python
# 定义均值和协方差矩阵（表示两个变量正相关）
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]] # 协方差矩阵

# 生成 1000 个二维数据点
data = np.random.multivariate_normal(mean, cov, 1000)

# 验证相关性
correlation_matrix = np.corrcoef(data, rowvar=False) # rowvar=False 表示每列是一个变量
print("生成数据的相关系数矩阵:\n", correlation_matrix)
# 输出会非常接近原始的协方差矩阵中的相关性（0.8）
# [[1.         0.796... ]
#  [0.796...   1.         ]]
```



------



### 数学与统计方法

NumPy 提供了很多用于聚合计算的函数。

```Python
arr = np.random.randn(5, 4) # 5x4 的随机数组

print(arr.mean())   # 计算所有元素的平均值
print(arr.sum())    # 计算所有元素的和
print(arr.min())    # 找出最小值
print(arr.max())    # 找出最大值
print(arr.std())    # 计算标准差
print(arr.argmax()) # 找出最大值的索引（拉平成一维后）
  
```

**沿着轴 axis 进行计算**

这是非常重要和常用的功能。在一个二维数组中：

- axis=0 代表沿着**行**操作（对每一**列**进行计算）。
- axis=1 代表沿着**列**操作（对每一**行**进行计算）。

```Python
# 计算每一列的和
print(arr.sum(axis=0)) # 返回一个长度为4的数组

# 计算每一行的平均值
print(arr.mean(axis=1)) # 返回一个长度为5的数组
  
```

### 总结

NumPy 是数据科学和机器学习的“水电煤”。你可能不总会直接使用它（比如在使用 Pandas 时），但它无处不在。

- **核心是 ndarray**：一个快速、高效的多维数组。
- **向量化是关键**：避免在 Python 中写循环，利用 NumPy 底层的 C 代码获得巨大性能提升。
- **索引和切片是利器**：特别是布尔索引，提供了无与伦比的数据筛选能力。
- **广播是魔法**：让你能灵活地处理不同形状数组间的运算。