---
title: Matplotlib
author: Alen
published: 2025-10-12
description: "数据可视化工具：Matplotlib介绍"
first_level_category: "人工智能"
second_level_category: "数据分析工具"
tags: ['机器学习','matplotlib']
draft: false
---

Matplotlib 是 Python 数据可视化领域的“奠基石”，几乎所有更高级的可视化库（如 Seaborn）都是在它的基础上构建的。它功能强大、灵活，可以让你创建几乎任何你能想到的静态、动态和交互式图表。

### 什么是 Matplotlib？

Matplotlib 是一个用于创建高质量图表的 Python 库。你可以用它来生成折线图、散点图、柱状图、直方图、饼图等等。它的设计哲学是尽可能地模仿 MATLAB 的绘图功能，因此对于有 MATLAB 使用经验的用户来说非常友好。

**核心理念：一切皆对象**

理解 Matplotlib 的关键在于它的两个核心对象：

1. **Figure (画布)**：    整个图表窗口。你可以把它想象成一张画纸，你所有的绘图内容都在这张纸上。一个 Figure 对象可以包含一个或多个 Axes 对象。
2. **Axes (坐标系/子图)**：    实际进行绘图的区域。你可以把它想象成画纸上的一个坐标系，大部分的绘图操作（如画线、画点）都是在 Axes 对象上进行的。

**两种绘图接口**

Matplotlib 提供了两种不同的编程接口：

1. **基于 pyplot 的状态机接口**：    这是一系列类似 MATLAB 的函数式命令。例如 plt.plot()、plt.title()。它会自动管理 Figure 和 Axes 对象。这种方式适合快速、简单的绘图。
2. **面向对象的接口**：    显式地创建 Figure 和 Axes 对象，然后调用这些对象的方法来进行绘图。例如 ax.plot()、ax.set_title()。**这是官方推荐的方式**，因为它对复杂的图表有更好的控制力，代码也更清晰。

**本指南将主要使用面向对象的接口，因为它更强大、更规范。**

首先，导入 pyplot 模块，通常简写为 plt。

```python
import matplotlib.pyplot as plt
import numpy as np 
```

### 创建一个简单的图表 (面向对象方式)

推荐的起点是使用 plt.subplots()，它会同时创建一个 Figure 和一个 Axes 对象。

```python
# 创建一个 Figure 和一个 Axes
fig, ax = plt.subplots()

# 准备数据
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

# 在 Axes 上绘图
ax.plot(x, y)

# 显示图表
plt.show()
```

------



### 常用图表类型

#### 1. 折线图 (.plot)

用于显示数据随某个连续变量变化的趋势。非常适合用于你的实验二中绘制 loss 和 accuracy 曲线。

```python
# 准备数据
x = np.linspace(0, 10, 100) # 0到10之间生成100个点
y1 = np.sin(x)
y2 = np.cos(x)

# 创建 Figure 和 Axes
fig, ax = plt.subplots()

# 绘制多条折线
ax.plot(x, y1, label='sin(x)') # label 用于图例
ax.plot(x, y2, label='cos(x)')

# 添加图例
ax.legend()

# 添加标题和坐标轴标签
ax.set_title("Sine and Cosine Waves")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

plt.show()
  
```

#### 2. 散点图 (.scatter)

用于展示两个变量之间的关系。非常适合用于你的实验一中绘制逻辑回归的散点图。

```python
# 准备数据
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50) # 颜色
sizes = 1000 * np.random.rand(50) # 点的大小

fig, ax = plt.subplots()

# 绘制散点图
# c: 颜色, s: 大小, alpha: 透明度
ax.scatter(x, y, c=colors, s=sizes, alpha=0.6)

ax.set_title("Scatter Plot Example")
plt.show()
  
```

#### 3. 柱状图/条形图 (.bar / .barh)

用于比较不同类别的数据。非常适合用于你的实验一中可视化特征的重要性。

```python
# 准备数据
categories = ['Feature A', 'Feature B', 'Feature C']
values = [10, 25, 15]

fig, ax = plt.subplots()

# 绘制垂直柱状图
ax.bar(categories, values)

ax.set_title("Feature Importance")
ax.set_ylabel("Importance Score")

plt.show()
  
```

#### 4. 直方图 (.hist)

用于显示单个数值变量的分布情况。

```python
# 准备数据 (从正态分布中随机采样)
data = np.random.randn(1000)

fig, ax = plt.subplots()

# 绘制直方图
# bins: 分成多少个柱子
ax.hist(data, bins=30)

ax.set_title("Histogram of a Normal Distribution")
plt.show()
  
```

#### 5. 热力图 (.imshow)

用于将一个矩阵的数据值以颜色的形式展现出来。非常适合用于你的实验二中可视化混淆矩阵。

```python
# 准备一个随机矩阵
matrix = np.random.rand(5, 5)

fig, ax = plt.subplots()

# 绘制热力图
im = ax.imshow(matrix, cmap='viridis') # cmap 是颜色映射方案

# 添加颜色条
fig.colorbar(im)

ax.set_title("Heatmap Example")
plt.show()
  
```

> **提示**：对于混淆矩阵，使用 seaborn.heatmap() 会更方便，因为它能自动添加数值标签，但其底层仍然是 Matplotlib。

------



### 图表定制

你可以控制图表的几乎所有元素。

```python
x = np.linspace(0, 10, 50)
y = x**2

fig, ax = plt.subplots(figsize=(8, 6)) # figsize 控制画布大小

# 控制线条样式、颜色、标记
ax.plot(x, y,
        color='red',           # 颜色
        linestyle='--',        # 虚线
        linewidth=2,           # 线宽
        marker='o',            # 标记点样式
        markersize=5,          # 标记点大小
        label='y = x^2')

# 设置标题和标签，并控制字体大小
ax.set_title("Customized Plot", fontsize=16)
ax.set_xlabel("X Axis", fontsize=12)
ax.set_ylabel("Y Axis", fontsize=12)

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)

# 添加网格
ax.grid(True, linestyle=':', alpha=0.7)

# 添加图例
ax.legend()

plt.show()
  
```

### 创建多个子图 (Subplots)

当需要在一个画布中展示多个图表时，plt.subplots() 非常有用。

```python
# 创建一个 1 行 2 列的子图布局
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# --- 在第一个子图 (axes[0]) 上绘图 ---
x1 = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x1)
axes[0].plot(x1, y1)
axes[0].set_title("Sine Curve")

# --- 在第二个子图 (axes[1]) 上绘图 ---
x2 = ['A', 'B', 'C', 'D']
y2 = [5, 8, 3, 6]
axes[1].bar(x2, y2, color='green')
axes[1].set_title("Bar Chart")

# 自动调整子图布局，防止重叠
plt.tight_layout()

plt.show()
  
```

### 保存图表

在脚本的最后，用 plt.savefig() 代替 plt.show() 或在其之前调用，可以将图表保存为文件。

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# 保存图表
# dpi: 分辨率, bbox_inches='tight': 自动裁剪掉多余的白边
plt.savefig("my_plot.png", dpi=300, bbox_inches='tight')

# 你仍然可以显示它
# plt.show()
  
```

### 结合 Pandas 使用

Pandas 的 DataFrame 对象内置了 .plot() 方法，它是 Matplotlib 的一个便捷封装。

核心的集成点是 DataFrame 和 Series 对象的 **.plot()** 方法。这个方法的强大之处在于它的 **kind** 参数，通过改变 kind 的值，你可以轻松地创建不同类型的图表。

#### **基础示例**

```python
# 创建一个包含不同类型数据的 DataFrame
data = {
    'A': np.random.randn(10).cumsum(), # 随机游走数据
    'B': np.random.rand(10) * 100,      # 0-100的随机数
    'C': np.random.randint(1, 5, 10),   # 1-4的随机整数
    'D': np.abs(np.random.randn(10))    # 随机正数
}
df = pd.DataFrame(data)
```

1. 折线图 (kind='line') - 默认

   这是最基础的图表，用于展示数据随索引变化的趋势。

   ```python
   # kind='line' 是默认值，可以省略
   df.plot(y=['A', 'B'], title='Line Plot Example')
   plt.show()
   ```

2. 柱状图/条形图 (kind='bar' / kind='barh')

   ```python
   # 截取前5行数据进行比较
   sub_df = df.head()
   
   # 垂直柱状图
   sub_df.plot(kind='bar', y='B', title='Bar Plot')
   plt.ylabel('Value')
   plt.show()
   
   # 堆叠柱状图
   sub_df.plot(kind='bar', y=['B', 'D'], stacked=True, title='Stacked Bar Plot')
   plt.show()
   
   # 水平条形图
   sub_df.plot(kind='barh', y='B', title='Horizontal Bar Plot')
   plt.xlabel('Value')
   plt.show()
   ```

3. 直方图 (kind='hist')

   用于展示单个数值变量的分布情况

   ```python
   # 绘制'A'列和'B'列的直方图
   df.plot(kind='hist', y=['A', 'B'], bins=10, alpha=0.7, title='Histogram')
   plt.show()
   ```

4. 箱形图 (kind='box')

   用于展示一组数据的分布情况，包括最小值、第一四分位数(Q1)、中位数、第三四分位数(Q3)和最大值，还能显示异常值。

   ```python
   df.plot(kind='box', y=['A', 'B', 'D'], title='Box Plot')
   plt.show()
   ```

5. 面积图 (kind='area')

   类似于折线图，但会填充线下方的区域，常用于展示累积总量随时间的变化。

   ```python
   df.plot(kind='area', y=['A', 'D'], stacked=False, alpha=0.5, title='Area Plot')
   plt.show()
   ```

6. 散点图 (kind='scatter')

   用于展示两个数值变量之间的关系。

   ```python
   # 必须指定 x 和 y
   df.plot(kind='scatter', x='A', y='B', title='Scatter Plot between A and B')
   plt.show()
   ```

   你还可以加入第三个变量来控制颜色 (`c`) 或大小 (`s`)：

   ```python
   df.plot(kind='scatter', x='A', y='B', c='C', cmap='viridis', s=df['D']*50, title='Advanced Scatter Plot')
   # c='C': 用'C'列的值来决定颜色
   # cmap='viridis': 使用'viridis'颜色映射
   # s=df['D']*50: 用'D'列的值来决定点的大小
   plt.show()
   ```

7. 饼图 (kind='pie')

   用于展示各部分占总体的比例。

   ```python
   # 饼图通常用于单列数据
   df['C'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Pie Chart for Category C')
   # .value_counts() 先统计C列中各类别的数量
   # autopct 用于显示百分比
   plt.ylabel('') # 隐藏y轴标签
   plt.show()
   ```

#### 高级绘图：pandas.plotting 模块

除了 .plot() 方法，Pandas 还有一个专门的 plotting 模块，提供了一些更复杂、更具分析性的可视化工具。

1. **散点图矩阵 (scatter_matrix)**

   它可以一次性展示 DataFrame 中所有数值变量两两之间的关系（散点图）以及每个变量自身的分布（直方图或核密度图）。

   ```python
   from pandas.plotting import scatter_matrix
   
   # 对整个DataFrame创建散点图矩阵
   # diagonal='kde' 表示在对角线上绘制核密度估计图
   scatter_matrix(df, alpha=0.8, figsize=(10, 10), diagonal='kde')
   plt.suptitle('Scatter Matrix') # 添加总标题
   plt.show()
   ```

2. **安德鲁斯曲线 (andrews_curves)**

   一种将多维数据点可视化为曲线的方法，用于发现数据中的聚类情况。属于同一类别的样本点，其曲线通常会聚集在一起。

   ```python
   from pandas.plotting import andrews_curves
   
   # 为了演示，我们先添加一个分类标签
   df['category'] = ['G1' if c > 2 else 'G2' for c in df['C']]
   
   andrews_curves(df, 'category', colormap='viridis')
   plt.title('Andrews Curves')
   plt.show()
   ```

​	

Pandas 绘图最强大的地方在于，**它的所有 .plot() 方法都会返回一个 Matplotlib 的 Axes 对象**。这意味着可以先用 Pandas 快速生成一个基本图表，然后用 Matplotlib 的全部功能对这个图表进行精细的定制。

```python
# 1. 使用 Pandas 快速生成基础图表，并捕获 Axes 对象
ax = df.plot(kind='line', y='A', figsize=(10, 6), title='Pandas Plot with Matplotlib Customization')

# 2. 使用 Matplotlib 的方法对 ax 对象进行精细调整
ax.set_xlabel('Index Position', fontsize=12)
ax.set_ylabel('Cumulative Sum', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6) # 添加网格
ax.axhline(0, color='red', linewidth=2) # 在 y=0 处画一条红色的水平线
ax.legend(['My Custom Label']) # 自定义图例

plt.show()
```

### 总结

Matplotlib 是一个极其强大的库，虽然初看起来有些复杂，但掌握了其核心的**面向对象**接口（Figure 和 Axes）后，你就能自如地创建和定制各种图表。

**对于你的实验项目：**

- **实验一**：
  - **任务 (2)**：使用 ax.scatter() 绘制散点图。
  - **任务 (4)**：使用 ax.bar() 或 ax.barh() 绘制特征重要性的条形图。
- **实验二**：
  - **任务 (2)**：使用 ax.plot() 绘制训练和验证过程中的 loss 和 accuracy 曲线。
  - **任务 (4)**：使用 ax.imshow() (或 seaborn.heatmap) 绘制混淆矩阵。

从简单的图表开始，逐步添加定制元素，是学习 Matplotlib 的最佳路径。
