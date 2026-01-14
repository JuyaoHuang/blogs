---
title: sklearn优化器
publishDate: 2025-10-21
description: "Scikit-learn optimizer的介绍和使用"
tags: ['ML']
language: 'Chinese'
first_level_category: "人工智能"
second_level_category: "机器学习理论"
draft: false
---

# sklearn优化器

本篇介绍sklearn优化器里有哪些优化器，以及在代码中怎么使用

---
不是所有 sklearn 模型都使用梯度下降法：

- **明确使用梯度下降的模型**

  - SGDClassifier：  随机梯度下降分类器
  - SGDRegressor：  随机梯度下降回归器
  - MLPClassifier / MLPRegressor：  多层感知机（神经网络），必须使用梯度下降法（或其变体如Adam）进行训练

- **可以选择使用梯度下降作为求解器的模型**

  - LogisticRegression

    ​	有多个 solver 参数可选，其中 'sag' 和 'saga' 是基于梯度下降的快速算法

  - Ridge, Lasso, ElasticNet

    ​	同样可以选择基于梯度下降的求解器

- **不使用梯度下降的模型**

  - inearRegression

    通常使用**普通最小二乘法**，这是一种直接求解的数学方法，而非迭代优化

  - 决策树 (Decision Tree Classifier) 和随机森林 (Random Forest Classifier)

    它们使用基于信息增益或基尼不纯度的贪婪算法来构建树，不涉及梯度

  - 支持向量机 (SVC)

    通常使用称为**序列最小最优化 (SMO)** 的高效算法
---

## 线性模型

这是求解器选项最丰富的模型类别。主要涉及 LogisticRegression, Ridge, Lasso, ElasticNet 等

### LogisticRegression 中的 solver 参数

|     Solver      |                             描述                             |                             优点                             |                            缺点                             |                         适用场景                         |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------------: | :------------------------------------------------------: |
|   **'lbfgs'**   | **(默认)** 拟牛顿法的一种。它使用梯度的二阶导数信息（Hessian矩阵的近似）来加速收敛。 |    **收敛速度快**，内存效率高，是大多数情况下的**首选**。    |             对特征尺度敏感，需要进行特征缩放。              |     多分类问题（multinomial）、小数据集、通用场景。      |
| **'liblinear'** |     基于坐标下降法（Coordinate Descent）和L1/L2正则化。      |        在**小数据集**上表现非常好，支持L1和L2正则化。        | 只能处理**二分类问题**，对于多分类需要用"one-vs-rest"策略。 |    二分类、小数据集、需要L1正则化（特征选择）的场景。    |
| **'newton-cg'** |  **牛顿共轭梯度法**。与'lbfgs'类似，也利用了二阶导数信息。   |                  对大规模数据有效，精度高。                  |                                                             |             多分类问题、对精度要求高的场景。             |
|    **'sag'**    | **随机平均梯度下降 (Stochastic Average Gradient)**。是梯度下降的一种变体，适用于**大规模数据集**。 | 在样本数量和特征数量都**很大**时，通常比其他求解器**快得多**。 |        对特征尺度敏感，需要特征缩放。收敛可能较慢。         |            **大数据集**（样本/特征数上万）。             |
|   **'saga'**    | **'sag'的改进版**。是唯一原生支持**L1正则化（Lasso）**的梯度下降类求解器。 | 兼具'sag'的速度优势，并且**支持L1, L2和ElasticNet正则化**。  |               对特征尺度敏感，需要特征缩放。                | **大数据集**，特别是当你需要**L1正则化**进行特征选择时。 |

**选择建议**:

- **默认用 'lbfgs'**，通常效果最好。
- 数据集很大（例如 > 10,000个样本）时，考虑用 'sag' 或 'saga'。
- 需要L1正则化时，如果数据集小用 'liblinear'，数据集大用 'saga'。
- **始终记得对数据进行标准化处理**

---

### 随机梯度下降模型    SGD Models

这类模型的核心就是随机梯度下降算法本身，因此没有solver参数可选，但可以通过其他参数来配置SGD的行为

#### SGDClassifier 和 SGDRegressor

这两个模型的  “优化器”  就是**随机梯度下降 (SGD)**，可以通过以下参数来微调它：

- **loss**: **损失函数**

  ​	例如，'hinge' (线性SVM), 'log_loss' (逻辑回归), 'squared_error' (线性回归)。

- **penalty**: **正则化项**

  ​	'l2', 'l1', 'elasticnet'。

- **learning_rate**: **学习率衰减策略**

  - 'constant':     学习率固定不变，由 eta0 控制
  - 'optimal':      Scikit-learn根据启发式规则自动计算
  - 'invscaling':      eta = eta0 / pow(t, power_t)，学习率随时间t逐渐减小
  - 'adaptive':     当训练损失不再下降时，自动将学习率除以5

- **eta0**:     初始学习率

- **momentum**: **动量**

  一个0到1之间的值，用于加速SGD在相关方向上的收敛并抑制振荡，通常设置为0.9左右

- **nesterovs_momentum**

  是否使用**Nesterov动量**，是动量法的一个改进版。

**总结**:  对于SGD模型，不是选择优化器，而是**配置SGD这个优化器本身**的各种行为

#### 代码示例

以求解一个 线性回归为例：$y = w*x+b$

```python
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# 1. 准备数据
# 创建一些近似 y = 2x + 1 的数据
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.1, 4.9, 7.2, 8.8]) # y 约等于 2*X + 1

# 注意：对于梯度下降，特征缩放非常重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 实例化模型
model = SGDRegressor(
    loss='squared_error', # 损失函数：均方误差 (MSE)
    penalty=None,         # 不使用正则化
    max_iter=1000,        # 最大迭代次数 (相当于 PyTorch 的 epochs)
    tol=1e-3,             # 如果损失改善小于这个值，就提前停止
    eta0=0.01,            # 初始学习率 (相当于 PyTorch 的 learning_rate)
    verbose=1             # 每隔一段时间打印一次训练进度
)

# 3. 训练模型
# 自动进行：预测 -> 计算损失 -> 计算梯度 -> 更新参数 的迭代过程。
model.fit(X_scaled, y)

# 4. 查看结果
print("\n训练完成！")
# SGDRegressor 学习到的参数存储在 .coef_ 和 .intercept_ 属性中
# 需要将缩放后的系数转换回来，才能与原始数据对应
# 转换公式: 
# w_orig = w_scaled / scaler.scale_
# b_orig = b_scaled - sum(w_scaled * scaler.mean_ / scaler.scale_)
original_coef = model.coef_ / scaler.scale_
original_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

print(f"学习到的权重 w: {original_coef[0]:.3f}")
print(f"学习到的偏置 b: {original_intercept[0]:.3f}")

# 5. 进行预测
new_X = np.array([[5.0]])
new_X_scaled = scaler.transform(new_X)
prediction = model.predict(new_X_scaled)
print(f"\n对 x=5 的预测值是: {prediction[0]:.3f}")
```



---

### 神经网络模型

这是Scikit-learn中唯一明确支持Adam等高级优化器的地方

#### MLPClassifier 和 MLPRegressor 中的 solver 参数

|   Solver    |                             描述                             |                             优点                             |                             缺点                             |                        适用场景                        |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------: |
| **'adam'**  | **(默认)** **自适应矩估计 (Adaptive Moment Estimation)**。结合了动量和RMSProp的优点，为每个参数计算自适应的学习率 | **鲁棒性强**，对超参数（如学习率）相对不那么敏感，在大多数情况下**收敛速度快且效果好** |                              无                              |  **通用首选**，尤其适用于较大数据集或较复杂的网络结构  |
|  **'sgd'**  |        **随机梯度下降**。在这里，它还支持配置**动量**        | 简单，可配置动量。在某些精细调优的场景下，可能找到比Adam更好的局部最优解 |            对学习率非常敏感，收敛速度通常比Adam慢            | 教学、研究、或当你需要对优化过程进行更精细的手动控制时 |
| **'lbfgs'** |           **拟牛顿法**。与线性模型中的'lbfgs'类似            |   在**小数据集**上表现非常好，通常能**更快收敛**到很好的解   | 不适用于大规模数据集，因为它需要将整个数据集加载到内存中计算梯度 |       **数据集较小**（通常几千个样本以下）的场景       |

**选择建议**:

- 绝大多数情况下，**直接使用默认的 'adam'**
- 如果数据集非常小，可以尝试 'lbfgs'，可能会更快收敛
- 如果想深入研究学习率和动量对训练的影响，可以换成 'sgd' 进行实验

#### 代码示例

在一个分类任务中如何使用  MLPClassifier  并配置 Adam优化器

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 创建一个模拟数据集
# 1000个样本，20个特征，2个类别
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# 2. 数据预处理
# 神经网络对特征缩放非常敏感，所以标准化是必须的
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 实例化模型，并配置Adam优化器
# MLPClassifier默认就使用Adam，这里显式写出来并调整参数
model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # 定义神经网络的结构：两个隐藏层，分别有100和50个神经元
    activation='relu',            # 激活函数
    solver='adam',                # 核心：在这里指定使用Adam优化器
    alpha=0.0001,                 # L2正则化项的系数
    batch_size='auto',            # mini-batch的大小
    learning_rate_init=0.001,     # 初始学习率 (相当于PyTorch中的lr)
    max_iter=300,                 # 最大迭代次数 (epochs)
    tol=1e-4,                     # 提前停止的容忍度
    random_state=42,              # 保证结果可复现
    verbose=True                  # 打印训练过程中的损失值
)

# 5. 训练模型
# .fit() 方法会调用Adam优化器，在后台进行完整的训练循环
print("开始使用Adam优化器进行训练...")
model.fit(X_train, y_train)

# 6. 进行预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n训练完成！")
print(f"在测试集上的准确率: {accuracy:.4f}")

#                 查看训练过程中的损失曲线
# import matplotlib.pyplot as plt
# plt.plot(model.loss_curve_)
# plt.title("Loss Curve during Training")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()
```



---

### 其他模型

许多其他Scikit-learn模型不使用基于梯度的优化器，因此没有相关的solver参数

- **LinearRegression**: 

  使用**普通最小二乘法**的解析解，不涉及迭代优化。

- **SVC / SVR (支持向量机)**

  使用**LIBSVM库**，内部实现了高效的**序列最小最优化 (SMO)** 算法。

- **DecisionTree / RandomForest / GradientBoostingClassifier**

  使用基于树的构建算法（如贪婪分裂、集成等），不涉及梯度下降

  注意：梯度提升中的  “梯度”  指的是损失函数的梯度，但其优化过程是迭代地添加新树，而非调整一组固定参数
