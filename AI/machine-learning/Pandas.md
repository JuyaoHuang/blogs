---
title: Pandas
author: Alen
published: 2025-10-12
description: "表格-数据分析工具：Pandas介绍"
first_level_category: "人工智能"
second_level_category: "数据分析工具"
tags: ['机器学习','pandas']
draft: false
---

Pandas 是建立在 NumPy 之上的，是 Python 中进行数据处理和分析**最核心、最流行**的库。如果你需要处理表格数据（比如 Excel 表格、CSV 文件或数据库中的表），Pandas 就是你的首选工具。

## 什么是 Pandas？

Pandas 的名字来源于 "Panel Data"（面板数据），是计量经济学中的一个术语。它提供了两种主要的数据结构，使得处理带标签的、异构类型的数据变得非常简单和直观。

1. **Series**：一维带标签的数组。
2. **DataFrame**：二维带标签的表格型数据结构，也是 Pandas 中最常用的。

**为什么需要 Pandas？**

NumPy 擅长处理同类型的数值数组，但现实世界的数据往往更复杂：

- **数据类型混合**：一个表格中通常既有数字，也有文本、日期等。
- **需要标签**：我们需要通过行名（索引）和列名来引用数据，而不是仅仅通过数字位置。
- **缺失数据**：真实数据中经常有缺失值，需要方便地处理。
- **需要高级操作**：如分组、聚合、合并、重塑等。

Pandas 完美地解决了这些问题，可以被看作是 Python 版的 Excel 或 SQL。

------

## Pandas 的核心数据结构

首先，导入 Pandas 库，通常简写为 pd。

```py
import numpy as np 
import pandas as pd # 通常与 NumPy 一起使用
```
#### 1. Series

`Series` 就像一个带索引的**一维**数组或字典。它由两部分组成：

*   `values`：一组数据（一个 NumPy 数组）。
*   `index`：一个与 `values` 相关联的标签数组。

```python
# 从列表创建 Series，索引默认为 0, 1, 2...
s1 = pd.Series([10, 20, 30, 40])
print(s1)
# 输出:
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# 自定义索引
s2 = pd.Series([1.75, 1.80, 1.65], index=['Alice', 'Bob', 'Charlie'])
print(s2)
# 输出:
# Alice      1.75
# Bob        1.80
# Charlie    1.65
# dtype: float64

# 像字典一样使用
print(s2['Bob'])      # 输出: 1.80
print('Alice' in s2) # 输出: True
  
```

#### 2. DataFrame

DataFrame 是 Pandas 的核心。它是一个二维表格，每列可以是不同的数据类型。你可以把它想象成一个 Excel 电子表格或一个 SQL 表。它也有行索引和列索引。

**创建 DataFrame**

最常见的方式是从一个**字典**创建，其中**字典的键成为列名，值（列表或数组）成为列数据**。

```python
    data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)
print(df)
# 输出:	
#       name  age         city
# 0    Alice   25     New York
# 1      Bob   30  Los Angeles
# 2  Charlie   35      Chicago
# 3    David   28      Houston
  
```

**Dataframe的属性**

```python
import pandas as pd
import numpy as np

# 创建一个示例 DataFrame，内容是学生信息和成绩
data = {
    '姓名': ['张三', '李四', '王五', '赵六', '孙七'],
    '专业': ['计算机', '物理', '计算机', '数学', '物理'],
    '年龄': [20, 21, 22, 21, 20],
    '期中成绩': [85, 92, 78, 88, 95],
    '期末成绩': [90, 88, 82, 94, 91]
}
# 使用自定义索引
df = pd.DataFrame(data, index=['s001', 's002', 's003', 's004', 's005'])

print("示例 DataFrame:")
print(df)
```

**属性 (Attributes)**

属性是 DataFrame 固有的特性，就像一个人的身高、体重一样。它们描述了 DataFrame 的结构和元数据，访问它们时后面**不加括号 ()**。

1. **.index (行索引)**

   描述：获取 DataFrame 的行索引（或称行标签）。行索引用于标识和访问每一行。

   就像：Excel 表格中最左侧的行号 (1, 2, 3...) 或你为每一行指定的唯一名称。

   ```python
   print(df.index)
   ```

   **输出**：

   ```bash
   Index(['s001', 's002', 's003', 's004', 's005'], dtype='object')
   ```

   **一句话总结**：查看或操作行的标签。

2. .**columns (列索引)**

   描述：获取 DataFrame 的列索引（或称列标签、列名）。
   就像：Excel 表格中最顶部的列名 (A, B, C...)。

   ```python
   print(df.columns)
   ```

   输出：

   ```bash
   Index(['姓名', '专业', '年龄', '期中成绩', '期末成绩'], dtype='object')
   ```

   **一句话总结**：查看或操作列的标签。

3. **.shape (形状)**

   描述：返回一个元组 (tuple)，表示 DataFrame 的维度，格式为 (行数, 列数)。这是**极其常用**的属性。
   就像：告诉你这个表格有多大，有多少行、多少列。

   ```python
   print(df.shape)
   ```

   **一句话总结**：快速了解数据有多少行、多少列。

4. **.size (元素总数)**

   描述：返回 DataFrame 中元素的总数量，即 行数 * 列数。

   ```bash
   print(df.size)
   ```

   **一句话总结**：告诉你 DataFrame 里总共有多少个数据点。

5. `.ndim` (维度)

   描述：返回数据的维度。对于 DataFrame，这个值永远是 `2`（因为它是一个二维表格）。对于 Series，它是 `1`。

   ```python
   print(df.ndim)
   ```

   **一句话总结**：确认这是个二维的表格。

6. **dtypes (数据类型)**
    描述：返回一个 Series，其中包含了每一列的数据类型。这对于检查数据是否被正确加载非常重要。
    常见类型：int64 (整数), float64 (浮点数), object (通常是字符串), bool (布尔值), datetime64 (日期时间)。
    
    ```python
    print(df.dtypes)
    ```
    **一句话总结**：检查每一列都存的是什么类型的数据。
    
7. **.values (数据值)**
   描述：  将 DataFrame 中的数据以 NumPy N 维数组 (ndarray) 的形式返回。这个过程会**丢弃**行和列的索引信息，只保留纯数据。
   用途：  当你需要将数据传递给 Scikit-learn 或其他只接受 NumPy 数组的库时，这个属性非常有用。
   
    ```python
    print(df.values)
    ```
    **一句话总结** ：只取数据，不要行列标签，得到一个 NumPy 数组。

需要注意的是，若列表中的每个元素为一个字典，则每个元素代表一行，字典中的 key 为列索引

------


## 常用操作

### 1. 数据的读取与写入

Pandas 可以轻松读取多种格式的数据。最常用的是 CSV 文件。

```python
# 假设有一个名为 'data.csv' 的文件
df = pd.read_csv('data.csv')

# 将 DataFrame 写入 CSV 文件
df.to_csv('output.csv', index=False) # index=False 表示不将行索引写入文件
  	
```

### 2. 查看与检查数据

当你拿到一个 DataFrame 后，首先需要了解它的基本情况。

```Python
# 查看前5行
print(df.head())

# 查看后5行
print(df.tail())

# 查看 DataFrame 的简要信息（索引、列、非空值数量、内存使用等）
print(df.info())

# 获取描述性统计信息（计数、均值、标准差、最小值、四分位数、最大值）
print(df.describe())

# 查看形状（行数, 列数）
print(df.shape)

# 查看列名
print(df.columns)
```

**方法 (Methods)**

​    方法是 DataFrame 可以执行的动作或计算。它们后面**需要加括号 ()**，并且可以接受参数。

#### head(n=5) (查看头部数据)

描述：  返回 DataFrame 的前 n 行。默认情况下 n=5。这是加载数据后，**第一个要使用的命令**，用于快速预览数据内容和格式。

```
# 查看默认的前5行
print(df.head())

# 查看指定的前3行
print(df.head(3))
```

#### tail(n=5) (查看尾部数据)

描述：与 head() 相对，返回 DataFrame 的后 n 行。默认情况下 n=5。
用途：可以用来检查数据是否完整加载，或者数据是否有序。

```
# 查看指定的后2行
print(df.tail(2))
```

#### info() (查看简要信息)

描述：打印 DataFrame 的一个简明摘要。这是**极其重要**的方法，提供了大量关键信息。
信息包括：

  - DataFrame 的类型。
  - 行索引的范围和类型。
  - 每列的名称。
  - 每列的**非空值 (Non-Null) 数量** (这是发现**缺失值**的最快方法)。
  - 每列的数据类型 (Dtype)。
  - 内存使用情况。

```
df.info()
```

**总结**：对 DataFrame 进行一次全面的“体检”，快速发现缺失值和类型问题。

#### describe() (获取描述性统计)

描述：针对**数值类型**的列，生成描述性统计信息。
信息包括：

- count: 非空值的数量
- mean: 平均值
- std: 标准差
- min: 最小值
- 25%: 第1四分位数
- 50%: 中位数（第2四分位数）
- 75%: 第3四分位数
- max: 最大值

```
print(df.describe())
```

**总结**：快速了解数值列的统计特征（均值、分布、离散程度等）。

#### .shape --获取行数和列数

#### .columns 获取列名

#### len() 函数可直接获取行数

python 的len() 函数可直接作用 DataFrame，返回行数

#### 总结

在任何数据分析项目中，当你加载完数据得到一个 DataFrame 后，标准的检查流程就是：

1. **df.head()** - 看一眼数据长什么样。
2. **df.shape** - 了解数据有多大。
3. **df.info()** - 进行“体检”，检查缺失值和数据类型。
4. **df.describe()** - 分析数值型数据的统计特征。

### 3. 数据选择与索引

这是 Pandas 中最重要、最灵活的部分。

#### a. 选择列

```Python
# 选择单列，返回一个 Series
ages = df['age']
print(ages)

# 选择多列，返回一个新的 DataFrame
subset = df[['name', 'city']]
print(subset)
```

#### b. 使用 .loc 和 .iloc 选择行和列

这是最规范、最不会引起混淆的选择方式。

- **.loc (Label-based selection)**：   基于**标签**（行索引名、列名）进行选择。
- **.iloc (Integer-based selection)**：    基于**整数位置**（从0开始）进行选择。

```Python
# 设置 'name' 列为新的行索引，方便演示 .loc
df_indexed = df.set_index('name')
print(df_indexed)
# 输出:
#          age         city
# name
# Alice     25     New York
# Bob       30  Los Angeles
# Charlie   35      Chicago
# David     28      Houston

# --- 使用 .loc ---
# 选择单行
print(df_indexed.loc['Bob'])

# 选择多行，这里是省略了列，
# 因为第二个参数是列，第一个参数传入的是列表
# 如果只选择列，行的表达式不能省略
# df.loc[:,'col']
print(df_indexed.loc[['Alice', 'David']])

# 选择行和列
print(df_indexed.loc['Charlie', 'city']) # 输出: Chicago
print(df_indexed.loc[['Alice', 'Bob'], ['age', 'city']])

# --- 使用 .iloc ---
# 选择第一行（位置0）
print(df.iloc[0])

# 选择前两行
# 注意：切片语法不能使用 [] 包住
print(df.iloc[0:2]) # 注意：不包含位置2，和Python切片一样

# 选择特定位置的元素（第2行，第1列）

print(df.iloc[2, 1]) # 输出: 35

# 选择特定的行和列
print(df.iloc[[0, 3], [0, 2]]) # 选择第0,3行和第0,2列
  
```
**注意**
1. 在 Pandas 的 loc 和 iloc 中，冒号 ":" 是一个特殊的符号，它代表 **所有**
   1. 当它用在行选择的位置时，代表“所有行”;当它用在列选择的位置时，代表“所有列”
   2. loc 的写法
   ```python
   df.loc[:, 'Glucose']
   ```
   3. iloc 写法
   ```python
   df.iloc[:, 1]
   # 或者使用切片
   df.iloc[0:784,1]
   ```
   
#### c. 布尔索引 (Boolean Indexing)

根据条件进行数据筛选，这在数据分析中极其常用。

```Python
# 筛选出年龄大于30的行
print(df[df['age'] > 30])

# 筛选出城市为 'New York' 的行
print(df[df['city'] == 'New York'])

# 组合多个条件（使用 & 表示'与'，| 表示'或'，条件需用括号括起来）
print(df[(df['age'] < 30) & (df['city'] == 'New York')])
  
```

### 4. 数据清洗

#### a. 处理缺失值

1. isnull()：  检查缺失值

   在 Pandas 中，缺失值通常表示为 NaN (Not a Number)。

   ```python
   data_missing = {
       'A': [1, 2, np.nan, 4],
       'B': [5, np.nan, 7, 8],
       'C': ['x', 'y', 'z', 'w']
   }
   df_miss = pd.DataFrame(data_missing)
   
   # 检查哪些是缺失值
   print(df_miss.isnull())
   
   # 删除任何包含缺失值的行
   df_dropped = df_miss.dropna()
   print(df_dropped)
   
   # 填充缺失值
   # 用一个常数填充
   df_filled = df_miss.fillna(0)
   print(df_filled)
   
   # 用每列的平均值填充（只对数值列有效）
   df_filled_mean = df_miss.fillna(df_miss.mean(numeric_only=True))
   print(df_filled_mean)
   ```

   > **提示**：在实验一的皮马印第安人糖尿病数据集中，一些0值实际上是缺失值。可以先将这些0替换为np.nan，然后再使用.fillna()方法用该列的均值或中位数来填充。

2. **dropna()**  主要作用是移除缺失值

   在 Pandas 中，缺失值通常用 NaN 来表示。对一个 Series 调用 .dropna() 时，它会遍历这个 Series 中的每一个元素，检查它是否是 NaN。如果某个元素是 NaN，它就会被丢弃。

   最终，这个方法会返回一个新的 Series，这个新的 Series 只包含原始 Series 中所有非缺失的值。

3. **.replace(to_replace, value, inplace=False, limit=None, regex=False)**：用于替换值
   - to_replace
      **要查找的目标值**
      它可以是单个值，也可以是一个列表，或者一个字典
   - value
      **用来替换的新值**
      它可以是单个值，也可以是一个列表或字典，具体取决于 `to_replace` 的形式。

   - inplace
      一个布尔值，默认为 `False`
      - False:   
         该操作会**返回一个新的、修改后的 DataFrame**，原始的 DataFrame **不变**
      - True: 
         该操作会**直接在原始的 DataFrame 上进行修改**，不返回任何东西 (返回 `None`)
   
4. **fillna()** 是 Pandas DataFrame 和 Series 对象的一个核心方法，作用是**填充缺失值**

   在Pandas中，缺失值通常用 NaN (Not a Number) 来表示。fillna() 方法会找到这些 NaN 值，并用指定的值或方法来替换它们。

   ```python
   dataframe_or_series.fillna(value, inplace=False)
   ```

   - value: 填充 NaN 的值。这个值可以是：

     - 一个具体的数值（如 0, 100）。
     - 一个字符串（如 'Unknown'）。
     - 一个字典（可以为不同的列指定不同的填充值）。
     - 一个计算出来的结果，比如均值、中位数或者众数

   - inplace=False (默认):    这是一个非常重要的参数。

     - inplace=False (默认值): fillna() 会创建一个新的、填充好值的DataFrame或Series并返回它，而原始的DataFrame保持不变。
     - inplace=True: fillna() 会直接修改原始的DataFrame，并且不返回任何东西 (返回 None)

     

   通常推荐使用 inplace=False 的方式，然后将结果重新赋值给原来的变量，这样做更安全：

   ```python
   df['age'] = df['age'].fillna(some_value)
   ```

5. **np.nan**： 
      python中的缺失值 NaN
6. **数据填充的选择**

   **选择的填充方法，必须最符合该特征的数据类型和其分布特点，以最大程度地减少对原始数据信息的扭曲**

   1. **均值**

      例如 age 是数值型数据，并且其分布可能受到异常值（极端值）的影响，中位数对异常值不敏感，是比均值更稳健、更安全的“中心”衡量标准。

      - 优点:
        - 实现简单，计算速度快。
        - 保持了数据集的整体均值不变。
      - 缺点:
        - 降低了数据的方差：因为用一个相同的值替换了许多不同的未知值，这使得数据的波动性（方差）减小了。
        - 对异常值敏感：如果数据中有极端值，均值会被拉高或拉低，用这个被污染的均值去填充可能会引入偏差。
        - 忽略了特征之间的相关性。

   2. **中位数**

      - 方法: 

        ```python
        df['age'] = fillna(df['age'].median())
        ```

      - 适用场景:     当数据分布是**偏斜的**或存在异常值时，中位数比均值更具代表性，是更稳健的选择。因为它**不受或很少受**数据集两端的极端值影响

   3. **众数**

      - 方法: (注意：.mode()返回一个Series，所以要取第一个元素 [0])

        ```python
        df['embarked'] = fillna(df['embarked'].mode()[0]) 
        ```

      - 适用场景: 

        ​	主要用于填充**分类型特征**，例如 embarked（登船港口）。用出现次数最多的港口来填充缺失的两个值是合理的。

#### b. 删除重复行 .drop

```Python
data_dup = {'A': [1, 2, 2, 3], 'B': ['a', 'b', 'b', 'c']}
df_dup = pd.DataFrame(data_dup)

# 删除重复行
df_no_dup = df_dup.drop_duplicates()
print(df_no_dup)
原矩阵:
   A  B
0  1  a
1  2  b
2  2  b
3  3  c
删除操作后的矩阵:
   A  B
0  1  a
1  2  b
3  3  c
```

#### c.删除行 df.drop('deck', axis=1)

- 'deck':    要删除的列的名称。
- axis=1:    axis 参数告诉 Pandas  要操作的是**列**。如果 axis=0 (默认值)，它会尝试删除名为 'deck' 的**行**，这通常会报错，因为行没有这样的标签。

### 5. 数据操作与转换

#### 1. 添加/修改列

```Python
# 添加一个新列
df['country'] = 'USA'

# 基于现有列计算新列
df['age_in_10_years'] = df['age'] + 10
print(df)
```

#### 2. 应用函数 (.apply)

可以对行或列应用一个自定义函数。

```Python
def get_age_group(age):
    if age < 30:
        return 'Young'
    else:
        return 'Senior'

# 对 'age' 列的每个元素应用函数
df['age_group'] = df['age'].apply(get_age_group)
print(df)
  
```

#### 3. 布尔列转换为数值(.astype())

大多数机器学习模型（如逻辑回归、神经网络）都需要**数值型**输入，它们无法直接处理 True 或 False 这样的布尔值。

因此，在将数据喂给模型之前，需要将布尔列转换为 **1** 和 **0**。

1. 使用 **.astype(int)**

   它会直接将 True 转换为 1，False 转换为 0

   ```python
   df['alone'] = df['alone'].astype(int)
   print(df['alone'].value_counts())
   ```

   ```python
   alone
   1    537
   0    354
   Name: count, dtype: int64
   ```

2. 自定义函数实现

   当不记得这函数时，手动实现布尔类型的转换

   ```python
   def turn_to_int(alone):
       if alone == 1:
           return 1
       elif alone == 0:
           return 0
   df['alone'] = df['alone'].apply(turn_to_int)
   print(df['alone'].value_counts())
   ```

   ```bash
   alone
   1    537
   0    354
   Name: count, dtype: int64
   ```


#### 4. 数据排序：.sort_values()

.sort_values()用于使 DF按照某一列/行的数值进行排序，参数介绍：

```python
def sort_values(
    self,
    by: IndexLabel,
    *,
    axis: Axis = 0,
    ascending: bool | list[bool] | tuple[bool, ...] = True,
    inplace: bool = False,
    kind: SortKind = "quicksort",
    na_position: str = "last",
    ignore_index: bool = False,
    key: ValueKeyFunc | None = None,
) -> DataFrame | None:
"""
Sort by the values along either axis.
by : str or list of str
            Name or list of names to sort by.
axis : "{0 or 'index', 1 or 'columns'}", default 0
     Axis to be sorted.
ascending : bool or list of bool, default True
     Sort ascending vs. descending. Specify list for multiple sort
     orders.  If this is a list of bools, must match the length of
     the by.
inplace : bool, default False
     If True, perform operation in-place.
     
Returns
-------
DataFrame or None
"""
```

- `by` 用于指定根据什么排序：列名或者几个列名
- `ascending` 指定升降序
- 示例：

```python
print(crucial_feature.sort_values(by='Coefficient', ascending=False))
```

### 6. 分组与聚合

这是 Pandas 最强大的功能之一，遵循 "split-apply-combine"（拆分-应用-合并）的模式。

即能够按照某个键-值对的值，进行特定 列的筛查和聚合

```Python
data_sales = {
    'store': ['A', 'B', 'A', 'B', 'A', 'C'],
    'product': ['apple', 'orange', 'apple', 'grape', 'orange', 'apple'],
    'sales': [100, 150, 120, 80, 200, 90]
}
df_sales = pd.DataFrame(data_sales)

# 按 'store' 分组，并计算每个商店的总销售额
store_sales = df_sales.groupby('store')['sales'].sum()
print(store_sales)
# 输出:
# store
# A    410
# B    230
# C     90
# Name: sales, dtype: int64

# 按 'store' 和 'product' 分组，并计算多种聚合统计(agg)
multi_group = df_sales.groupby(['store', 'product'])['sales'].agg(['mean', 'sum', 'count'])
print(multi_group)
# 输出
#                 mean  sum  count	
# store product                   
# A     apple    110.0  220      2
#       orange   200.0  200      1
# B     grape     80.0   80      1
#       orange   150.0  150      1
# C     apple     90.0   90      1
```

#### 数据分箱 .cut

**作用**：

将连续的数值型数据，根据指定的边界（bins），切割成一系列离散的、不重叠的区间（类别）

```python
pandas.cut(x, bins, right=True, labels=None, include_lowest=False)
```

1. x

   要进行分箱操作的一维数组或Series

2. bins

   核心参数，定义了如何进行切割。它有多种提供方式：

   1. 一个整数：表示将 x 的整个范围（从最小值到最大值）等宽地切分成多少个箱。

   2. 一个序列：可以精确地定义每个箱的边界

      ```python
      bin_edges = [0, 18, 35, 60, 100]
      pd.cut(ages, bins=bin_edges)
      # 结果会按照定义的边界来切分:
      # (0, 18]
      # (18, 35]
      # (35, 60]
      # (60, 100]
      ```

3. right

   决定了区间的闭合方向

   - right=True (默认): 区间是左开右闭的，形式为 (a, b]

     ```python
     # 边界 [0, 18, 60]
     pd.cut(pd.Series([18]), bins=[0, 18, 60], right=True)
     # 输出: (0.0, 18.0] -> 18被包含在第一个区间
     ```

   - right=False: 区间是左闭右开的，形式为 [a, b)

4. labels (列表或 False, 默认为 None)

   为生成的新箱指定自定义的名称，让结果更具可读性

   - labels=False:    只返回代表每个箱的整数索引，而不是区间对象。
   - 提供一个列表: 列  表的长度必须**比箱的边界数量少1**（因为 N 个边界会产生 N-1 个箱）

5. include_lowest (布尔值, 默认为 False)

   当 bins 是一个序列时，这个参数决定了第一个区间的左边界是否为闭合的

   - include_lowest=False (默认): 

     ​	第一个区间是 (a, b] (如果 right=True)。如果你的数据里有一个值正好等于第一个边界 a，它会被标记为缺失值 NaN，因为它不包含在任何区间内。

   - include_lowest=True: 

     ​	第一个区间会变成**全闭合**的 [a, b]。这确保了等于最低边界的值能被包含进去。

| 特性           | pd.cut                                                       | pd.qcut                                                      |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **切割依据**   | **基于值**                                                   | **基于样本数量**                                             |
| **核心思想**   | 创建**等宽**或**自定义宽度**的箱                             | 创建**样本数量大致相等**的箱                                 |
| **箱的宽度**   | 固定或由你定义                                               | 动态变化，通常不相等                                         |
| **箱内样本数** | 通常不相等，差异可能很大                                     | 大致相等                                                     |
| **主要参数**   | bins 可以是整数或边界列表                                    | q 是一个整数，代表箱的数量                                   |
| **适用场景**   | 有明确的业务逻辑或固定的区间标准时（如年龄段、价格范围、分数等级） | 想观察不同收入水平（如最低10%的人，次低10%的人...）或处理高度偏斜的数据时 |

#### 数据合并 ---concat()

concat()函数：

**参数**

```bash
concatenate(...)
    concatenate(
        (a1, a2, ...),
        axis=0,
        out=None,
        dtype=None,
        casting="same_kind"
    )
```

- a1, a2, ... : sequence of array_like

  数组必须具有相同的形状，除了与 `axis` 对应的维度（默认为第一个）。

- axis : int, optional
  数组连接时所沿的轴。如果 axis 为 None，则**数组在使用前会被展平**。默认值为 0

- out : ndarray, optional

  如果提供，则为放置结果的目标。形状必须正确，与 concatenate 返回的结果（如果未指定）相匹配

- dtype : str or dtype
  如果提供，目标数组将具有此数据类型。不能与 `out` 同时提供。

**返回值**

res : ndarray 数组

​	The concatenated array

**示例**

```python
import numpy as np
a = np.array([[1, 2], [3, 4]]) 
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0) # 2*2 1*2 按行合并，那么 [5,6]位于[3,4]下面
输出：
array([[1, 2],
       [3, 4],
       [5, 6]])
np.concatenate((a, b.T), axis=1)
输出：
array([[1, 2, 5],
       [3, 4, 6]])
p.concatenate((a, b), axis=None)
输出：
array([1, 2, 3, 4, 5, 6])
```





## axis 参数详解

axis 参数决定了 Pandas 函数沿着哪个方向进行计算和操作

axis=0 或 axis='index'：**“沿着行向下”**。

- 操作会垂直地穿过所有的行，对**每一列**进行计算。
- 结果的索引通常是原来的列名。
- 假设原本是一个 $ M * N$ 的矩阵，当选择 axis = 0时，得到的矩阵是一个 $1 * N$的矩阵

axis=1 或 axis='columns'：**“沿着列向右”**。

- 操作会水平地穿过所有的列，对**每一行**进行计算。
- 结果的索引通常是原来的行索引。
- 假设原本是一个 $ M * N$ 的矩阵，当选择 axis = 0时，得到的矩阵是一个 $M * 1$的矩阵

## DataFrame 常见的数据指标

#### 1. 衡量集中趋势

- .mean() ：计算均值
- .median()：中位数
- .mode()：众数

#### 2. 衡量离散程度

- .std()：标准差
- .var()：方差
- .min()：最小值
- .max()：最大值
- .quantile()：计算分位数：
  - df.quantile(0.25) 会计算 25% 分位数（第一四分位数，Q1）。
  - df.quantile(0.5) 等同于中位数。
  - df.quantile(0.75) 会计算 75% 分位数（第三四分位数，Q3）。

#### 3. 描述性与信息性方法

- .sum()：计算总和。

- .count()： 计算非缺失值 (non-NA) 的数量。当你处理有缺失值的数据时，这个函数非常重要。

- .describe()： 一键生成描述性统计摘要。

  ​	这是进行探索性数据分析（EDA）时最先使用的函数之一。它会一次性返回计数、平均值、标准差、最小值、四分位数和最大值。

#### 4. 关系分析

- .corr()：计算相关系数矩阵。
  - 含义: 相关系数（通常指皮尔逊相关系数）衡量了两个数值变量之间的线性关系强度和方向。
  - 取值范围在 -1 到 +1 之间。
  - +1: 完全正相关。
  - -1: 完全负相关。
  - 0: 没有线性关系。

## 文本类型转换数值类型

### pd.get_dumies() 函数

在 pandas中， **One Hot Encoding**已经封装为一个函数：`.get_dumies()`

#### 主要作用

实现 one hot encoding，将文本类型转化为数值类型，并消除虚构的顺序关系，**使类别特征适合线性模型**

#### 参数解释

```python
pandas.get_dummies(data, prefix=None, columns=None, drop_first=False, dummy_na=False, ...)
```

1. data

   需要进行转换的数据，可以是一个 Series（单列）或一个 DataFrame（整个数据表）

2. columns

   一个列表，用来指定**想对哪些列**进行独热编码。

   如果省略这个参数，get_dummies() 会自动尝试转换所有数据类型为 object 或 category 的列。

   例如：columns=['sex', 'embarked']

3. prefix

   一个字符串或字典，用于为新生成的虚拟变量列添加**前缀**。这能让新列名更具可读性，并避免命名冲突。

   例如：

   ```python
   prefix='sex'
   ```

    会生成 sex_male, sex_female 这样的列。

   如果columns参数被使用，prefix可以是一个包含每个列对应前缀的列表或字典，如

   ```python
   prefix={'sex': 'Sex', 'embarked': 'Port'}
   ```

   即，可以对每一列生成的虚拟变量添加一个前缀

4. **drop_first**

   - 布尔值，默认为 False。如果设置为 True，则在为 K 个类别创建 K 个虚拟变量后，会**丢弃第一个**类别对应的列。

   - **用法**：drop_first=True

   - **目的**：

     这是为了 **避免“虚拟变量陷阱”（多重共线性）**

     对于线性模型来说，K-1个虚拟变量已经包含了所有信息，保留 K 个会导致列之间完全线性相关，可能会对模型造成问题。**在为线性模型准备数据时，强烈建议设置为 True**。

5. dummy_na

   1. 布尔值，默认为 False。如果设置为 True，并且数据中包含缺失值 (NaN)，它会为 NaN 也创建一个专门的虚拟变量列。
   2. dummy_na=True
   3. 有时候  “数据缺失”  本身就是一种信息（例如，用户拒绝回答某个问题）。通过为 NaN 创建一列，可以让模型学习到缺失值是否与预测目标有关。

## pandas数据结构转为np

### `pd.to_numpy()`

```python
def to_numpy(self, dtype=None, copy=False, na_value='<no_default>'): # reliably restored by inspect
    """
    Convert to a NumPy ndarray. 转换为 NumPy ndarray

    This is similar to :meth:`numpy.asarray`, but may provide additional control
    over how the conversion is done.

    Parameters
    ----------
    dtype : str or numpy.dtype, optional
        The dtype to pass to :meth:`numpy.asarray`.
        传递给 :meth:`numpy.asarray` 的数据类型
    copy : bool, default False
        Whether to ensure that the returned value is a not a view on
        another array. Note that ``copy=False`` does not *ensure* that
        ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
        a copy is made, even if not strictly necessary.
    na_value : Any, optional
        The value to use for missing values. The default value depends
        on `dtype` and the type of the array.

    Returns
    -------
    numpy.ndarray
    """
```



## 总结

Pandas 是进行探索性数据分析（EDA）、数据清洗和数据预处理的瑞士军刀。

- **Series 和 DataFrame** 是两大核心数据结构。
- **数据读写** (read_csv) 是起点。
- **查看与检查** (head, info, describe) 是了解数据的第一步。
- **数据选择** ([], .loc, .iloc, 布尔索引) 是最常用、最重要的技能。
- **数据清洗** (dropna, fillna) 是保证数据质量的关键。
- **分组聚合** (groupby) 是进行复杂数据分析和洞察发现的强大工具。