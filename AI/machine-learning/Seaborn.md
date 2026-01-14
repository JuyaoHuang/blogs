---
title: Seaborn
publishDate: 2025-10-25
description: "数据可视化工具：Seaborn介绍"
tags: ['ML']
language: 'Chinese'
first_level_category: "人工智能"
second_level_category: "数据分析工具"
draft: false
---

# Seaborn 

## Seaborn 绘图的核心思想-数据驱动

Seaborn 是建立在 Matplotlib之上的，它提供了一个更高级的接口，能够轻松地绘制出各种美观且信息丰富的统计图形。
Seaborn 的核心理念：通常会先有一个 Pandas DataFrame，然后告诉 Seaborn 函数使用 DataFrame 的哪一列作为 x 轴，哪一列作为 y 轴，哪一列用来区分颜色 (hue)

---

## 通用的重要参数

- `data`

  - **作用**：指定用于绘图的 Pandas DataFrame。
  - **示例**：`data=feature_importance_sorted`
  - **解释**：这是你告诉 Seaborn   “嘿，从这张表里拿数据”的方式。一旦指定了 data，下面的 x, y, hue 参数就只需要传递**列名字符串**即可

- `x, y`

  - **作用**：将 DataFrame 中的 **列** 映射到图表的 x 轴和 y 轴。

  - **示例**：`x='Absolute Coefficient', y='Feature'`

  - **解释**：

    这是绘图的核心。对于 barplot，通常一个是类别变量（如 'Feature'），另一个是数值变量（如 'Absolute Coefficient'）

    Seaborn 通常能根据你传递的数据类型自动判断是画水平条形图还是垂直条形图。

- `hue`

  - **作用**：引入**第三个维度**进行分组。它会根据指定列中的不同类别，将 x 或 y 上的条形图进一步拆分并用**不同颜色**加以区分

  - **示例**：假设 DataFrame 中有一列叫 'Feature Type'（例如 '生理指标', '血液指标'），你可以写 hue='Feature Type'

  - **解释**：

    hue 是 Seaborn 最强大的参数之一

    它能让你在同一张图上进行非常直观的比较。例如，在分析泰坦尼克数据时，可以用 x='Pclass' (舱位)，y='Fare' (票价)，hue='Sex' (性别) 来同时展示不同舱位下男女乘客的平均票价

- `palette`

  - **作用**：

    控制颜色盘。使用 hue 参数时，palette 决定了用来区分不同类别的颜色集合。它也可以为没有 hue 的图指定整体色系

  - **示例**：`palette='viridis', palette='rocket', palette='coolwarm', palette='Set2'`

  - **解释**：这是让图表变漂亮的关键。Seaborn 提供了大量预设的、经过精心设计的调色板。可以去 Seaborn 官方文档查看所有可用的调色板

- `color`

  - **作用**：在**不使用 hue** 的情况下，为图中的所有元素设置一个单一的、统一的颜色
  - **示例**：`color='steelblue'`
  - **解释**：如果只是想画一个简单的单色图，用 color 就很方便。如果同时指定了 hue 和 color，hue 的优先级更高

- `orient`

  - **作用**：显式指定图的方向，是垂直 ('v') 还是水平 ('h')
  - **示例**：`orient='h'`
  - **解释**：大部分情况下，Seaborn 能根据 x 和 y 的数据类型（数值 vs. 类别）自动推断方向，所以这个参数不常用。但在某些模糊的情况下，可以用它来强制指定方向。


## 折线图

### Seaborn 函数：sns.lineplot()

**核心参数**：

- data:  DataFrame
- x:   指定作为 x 轴的列名（例如时间、epoch）。
- y:   指定作为 y 轴的列名（例如销售额、模型损失值）
- hue:   指定一个分类列，用于在同一张图上绘制多条不同颜色的折线进行对比

**示例代码与说明**：
假设想可视化模型在训练过程中的损失（loss）变化

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设这是你记录的训练历史
history = {
    'epoch': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'loss': [0.6, 0.45, 0.3, 0.2, 0.15, 0.7, 0.6, 0.55, 0.52, 0.5],
    'type': ['Training', 'Training', 'Training', 'Training', 'Training',
             'Validation', 'Validation', 'Validation', 'Validation', 'Validation']
}
history_df = pd.DataFrame(history)

plt.figure(figsize=(10, 6))
sns.lineplot(data=history_df, x='epoch', y='loss', hue='type', marker='o')
plt.title('模型训练与验证损失曲线', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True)
plt.show()
```

---

##  散点图

**主要用途**：探究两个数值变量之间的关系和分布模式

### Seaborn 函数：sns.scatterplot()

**核心参数**：

- `data`:     DataFrame
- `x, y`:   分别指定作为 x 轴和 y 轴的数值列名
- `hue`:   指定一个分类列，让不同类别的点显示不同的颜色，这对于分类问题极其有用：该列应该是一个标签列，例如布尔值 0和 1，进而让函数可据此进行分类填色
- `size`:   指定一个数值列，用点的大小来表示第四个维度的信息
- `style`:   指定一个分类列，用点的形状来表示第五个维度的信息

**示例代码与说明**：
处理皮马印第安人糖尿病数据时，你想看看血糖（Glucose）和身体质量指数（BMI）与是否患病（Outcome）的关系

```python
# 假设 df 是加载的糖尿病数据集 DataFrame
# df = pd.read_csv('diabetes.csv')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', alpha=0.7)
plt.title('血糖、BMI与糖尿病结果的关系', fontsize=16)
plt.xlabel('血糖 (Glucose)', fontsize=12)
plt.ylabel('身体质量指数 (BMI)', fontsize=12)
plt.grid(True)
plt.show()
```

---

## 柱状图/条形图

**主要用途**：比较不同类别下的某个数值的统计量（通常是均值）

**Seaborn 函数**：

- `sns.barplot()`:   计算并展示每个类别的中心趋势（默认是均值）。
- `sns.countplot()`:   只统计每个类别出现的次数，相当于一种特殊的直方图

**核心参数 (barplot)**：

- data:  DataFrame
- x, y: 一个是类别列，一个是数值列
- hue: 进一步对每个条进行分组
- orient: 'v' (垂直) 或 'h' (水平)。通常 Seaborn 会自动判断

**示例代码与说明**

```python
# 构建新的 DataFrame
crucial_feat = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefficients})
# 绘制条形图
plt.figure(1, figsize=(18, 8))
# y轴是特征名称，x轴是其重要性（系数绝对值）
ax = sns.barplot(x='Abs Coefficient', y='Feature', data=crucial_feat, palette='viridis')
```

---

## 直方图 

**主要用途**：理解单个数值变量的分布情况

### Seaborn 函数：sns.histplot()

**核心参数**

- data:  DataFrame。
- x: 想要查看分布的数值列
- bins: 直方图的箱子（柱子）数量，bins 越多，图形越精细
- kde=True: 同时绘制核密度估计曲线（一条平滑的线），更好地展示分布形状
- hue: 在同一张图上绘制并比较不同类别的分布

**示例代码与说明**：
想看看患病和不患病的人在年龄（Age）分布上有没有差异。

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Outcome', kde=True, bins=30, palette='coolwarm')
plt.title('不同结果下的年龄分布', fontsize=16)
plt.xlabel('年龄 (Age)', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.show()
```

---

## 热力图

**主要用途**：将一个矩阵（二维数组）中的数值用颜色深浅来进行可视化。最常见的用途是展示特征之间的**相关性矩阵**

### Seaborn 函数：sns.heatmap()

**核心参数**

- **第一个参数**: 一个二维的数据矩阵，例如用 `df.corr() `计算出的相关性矩阵
- `annot=True`: 在每个格子里显示数值
- `cmap`: 指定颜色映射方案，例如 `'coolwarm', 'viridis', 'YlGnBu'`
- `fmt='.2f'`: 控制格子里数值的显示格式（例如保留两位小数）

**示例代码与说明**：
这是实验报告中分析特征关系时的必备图形

```python
# 1. 计算特征相关性矩阵
correlation_matrix = df.corr()
# 2. 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('皮马印第安人糖尿病数据集特征相关性热力图', fontsize=16)
plt.show()
```

![pima_heatmap](./imgs/seaborn/pima_heatmap.png)

这张图用颜色直观地展示了所有特征两两之间的相关性。暖色（如红色）表示强正相关，冷色（如蓝色）表示强负相关，颜色浅表示相关性弱。这有助于理解哪些特征是高度相关的（可能存在冗余），哪些特征和目标变量 Outcome 关系最密切

---

## 总结

| 图表类型   | Seaborn 函数      | 主要用途                               |
| ---------- | ----------------- | -------------------------------------- |
| **折线图** | sns.lineplot()    | 查看数值随连续变量（如时间）的变化趋势 |
| **散点图** | sns.scatterplot() | 探究两个数值变量之间的关系             |
| **柱状图** | sns.barplot()     | 比较不同类别下某数值的统计量（如均值） |
| **计数图** | sns.countplot()   | 统计不同类别的样本数量                 |
| **直方图** | sns.histplot()    | 查看单个数值变量的分布                 |
| **热力图** | sns.heatmap()     | 可视化矩阵数据，常用于展示相关性矩阵   |