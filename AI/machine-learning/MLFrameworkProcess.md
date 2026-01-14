---
title: 机器学习框架流程
publishDate: 2025-10-21
description: "机器学习的架构流程描述"
tags: ['ML']
language: 'Chinese'
first_level_category: "人工智能"
second_level_category: "机器学习理论"
draft: false
---

# 机器学习的框架流程

首先明确一点：**机器学习是后世所有AI应用的基础**，深度学习、大模型、LLM、vLLM、agent都是在机器学习基础上建立起来的。因此机器学习的经典内容和实现路线是需要特别熟悉的，尤其是在 **评价标准、数据清洗、特征工程、模型选择、模型训练和评估**方面。

本篇主要介绍机器学习的工作流程，在介绍完基本的工作流后，会对其中的一些重点地方做额外讲解

本篇主要面向的是初入机器学习的学习者，因为不论是算法工程师的应用方向，还是研究算法底层实现的AI算法研究员，都应该掌握基本的机器学习基础 （更多倾向于AI算法研究员），在这之后学习主流的环境框架：PyTorch或TensorFlow，但本篇不做涉及。

## 机器学习的目的

开始之前，应该明确一下机器学习提出的目的。
**机器学习的最终目标都是为了从数据中自动学习出一个能够进行预测的数学函数 $f$** 
**重点**：
1. 自动学习  ----能够自行学习的模型
2. 预测      ----能够完成对未知的可信预测

这个函数 $f$ 接收一些输入（特征值 $X$），然后给出一个输出（预测 $\hat{y} $），这个输出 $hat{y}$ 应该尽可能地接近真实值 $y$。
$$
y \approx \hat{y} = f(X)
$$
例如，糖尿病预测中：

- X = {怀孕次数，BMI，胰岛素水平，年龄...}
- $y$ = {1（患病），0（不患病）}
- 目的：找到一个函数 $f$，把一个人的生理指标 X 输入进去，能够得到一个接近 1或者 0的预测结果 $\hat{y}$

## 实现架构
### 1. 定义问题
在训练获得此模型函数之前，应该**明确我们要做的事**：我们获得这个函数后要做的事情是什么，它要解决的问题是什么。明确了要解决的问题，才知道要选择什么算法模型做训练。

- **明确业务问题**：
     明确要解决的是什么问题。例如，是想预测房价，还是提高用户点击率
     
- **定义任务类型**：
    一般任务类型分为三类：
    - 分类问题： 预测类别，例如 "是否患病"
    - 回归问题： 预测数值，例如 "房价"
    - 聚类问题： 无监督分组
    
- **确定评估指标**：
    如何衡量模型是否训练成功。是追求高准确率，还是追求精确率/召回率
    
    - 准确率 Accuracy：判断某个类别的事物是不是属于目标类别，例如 苹果是不是水果
    
    - 精确率 Precision：判断某个事物是不是自己需要的那一类，例如 苹果是好的还是坏的。
    
      [评估指标介绍](#评价指标)
    
- **可行性分析**：

- **价值分析**：
    分析此业务问题是否具有为了解决它而训练一个模型的价值，或者说，此业务是否能够被解决，如果无法解决，那它就没有实现的价值

### 2. 数据采集和EDA

在确定好问题之后，就要准备训练模型的数据集，并且对数据集进行清洗和分析。
*分析可选，因为这个数据分析太有门道和难度了，甚至有相关的工作岗位：数据分析工程师*
- 数据采集：
    获取所有可能相关的原始数据。数据可以来自公司数据库、公开数据集、爬虫、传感器等。数据的质量和数量直接决定了项目的上限
- **[数据清洗](#数据清洗)**：
    真实世界的数据是很 "脏"的，没有那么完美的数据集给你使用。你需要在这一步解决**原始数据的缺失值、异常值、重复值**等问题，避免模型学习到了垃圾数据。
- 探索性数据分析（EDA）：
    通过可视化和统计的方式来理解数据，目的是为了发现数据中的规律、相关性，为后续特征工程提供灵感和依据。
    *"灵感"就意味着很难找*
    常见的可视化方式有：
    - 直方图
    - 散点图
    - 热力图

### 3. 数据划分

在训练模型之前，会对数据进行划分。将其划分为**训练集、测试集和验证集**，分别用来训练、测试、最终评估模型的效果，以防止[过拟合问题](#过拟合)。

### 4. 特征工程

因为数学模型（算法）通常只理解数值，而原始信息一般是文本信息，或者部分数值（如时间，尺度差异大的特征）也不适合直接送入模型，因此要对数据进行转换，将其变为模型可理解的数据。

例如，将日期拆分为年、月、日，或者对文本进行向量化。

常用的操作为：

- 数据标准化/归一化：解决不同特征尺度不一的问题
- 独热编码：One-Hot Encoding，处理类别特征
- 特征选择：剔除无关或冗余的特征

一个好的特征工程可以直接让简单的模型性能翻倍

---

**补充**

**训练集和测试集都需要进行数据预处理**：

**必须**先划分数据集，再做数据预处理

1. 数据集划分为训练集：$X\_ train, y\_ train$和数据集：$X\_ test, y\_ test$

2. 对训练集进行数据处理，此时数据集对象学习到测试集的数据特征

   实际上我们不是直接把一整个数据集直接喂给模型，而是使用一个对象 $scaler$ 给模型传递数据和该数据集的数据特征。因此该对象学习数据集的数据特征时，不应该学习到测试集的数据特征，不然会出现**数据泄露**的问题，用一个类比：

   训练集是习题集，测试集是考试卷子，现在孩子学习的时候，你应该是把习题集的答案告诉它，让它知道自己是否做错。如果训练的时候你都直接把**考试卷子的答案**都直接给它了，那它还学个蛋，直接抄答案不就完了

   **核心**：

   不能使用测试集的数据特征对模型进行训练，因此必须先划分数据集，再使用训练集的数据特征进行训练

   1. 标准化/归一化

      ```python
      scaler = StandardScaler()
      scaler.fit(X_train) <-- 只在训练集上学习规则
      X_train_scaled = scaler.transform(X_train)
      X_test_scaled = scaler.transform(X_test) <-- 用从训练集学到的规则来转换测试集
      ```

      - **.fit()** 是用来**学习**转换规则的（比如学习平均值和标准差）

        如果在整个数据集上fit，就相当于提前把“期末考试”的统计信息（平均分、难度分布）告诉了正在学习的学生，这会导致模型评估结果过于乐观，是一种作弊行为。

   2. 独热编码

      ```python
      encoder = OneHotEncoder(handle_unknown='ignore') (handle_unknown='ignore' 是一个神器，它能让测试集中出现的新类别被编码为全0，从而避免报错)
      encoder.fit(X_train[['城市']]) <-- 只在训练集上学习所有可能的城市
      X_train_encoded = encoder.transform(X_train[['城市']])
      X_test_encoded = encoder.transform(X_test[['城市']]) <-- 用从训练集学到的类别列表来转换测试集
      ```

3. 将模型学习到的规则

---

### 5. 模型选择

处理好数据集后，就要选择合适的算法模型做训练。

**选择标准**：
- 问题类型：分类还是回归
- 数据特点：线性/非线性、数据量大/小
- 业务需求：高精度还是高速度，质量拉高还是速度为先

例如：
- 数据量不大，问题是二分类，逻辑回归模型就很好
- 数据量很大，特征类型间关系复杂，可能会上神经网络

这一步决定了要使用什么算法模型构建最终的预测函数 $f$

### 6. 模型训练（核心）

选取好模型后，我们就来到了激动人心的炼丹环节。

这一环节涉及到很多技术，国内高校大部分和机器学习有关的课程教的就是这部分内容的原理，例如逻辑回归、线性回归、梯度下降法、损失函数等等。

因此这部分我们也分三个部分解释，并且在后文会给出相关的知识介绍，但受篇幅限制，不会特别详细。

- [算法模型](#机器学习算法分类)（Algorithm）：

    算法模型提供了一个具体的数学结构/框架，或者成为函数模板，它就是要炼丹的对象，可称为基底模型。

    以逻辑回归为例， 逻辑回归的模板是
    $$
    \hat{y} = \sigma(w^T X + b) \iff \hat{y} = \frac{1}{1 + e^{-(w^T X + b)}}
    $$
    这个模板定义了模型能学到的规律的形式，未来预测也是按照此公式去输出值。模型的任务就是在此模板下，找到**最好**的参数 $w$ 和 $b$
- [损失函数](#损失函数)（Loss function）

    损失函数用来**衡量预测误差**
    对每一个训练样本，损失函数会计算出模型预测值 $\hat{y}$ 和真实值 $y$ 之间的差距。它的目的是**提供一个明确的、可量化的优化目标**：这个差距（损失）越小，模型就越好
- 优化器（Optimizer）

    优化器是**驱动模型参数更新的工具**
    它根据损失函数计算得到的误差，找到一个能让这个误差缩小的方向，然后告诉模型进行参数（例如逻辑回归的 $w$ 和 $b$）的微调
    最常用的就是[梯度下降法](#梯度下降法及其变种)（尤其是Adam算法）

**总结**：模型训练的过程就是：**优化器**不断调整**算法**的内部参数，它调整的唯一依据是如何让**损失函数**的值变得越来愈小。

### 7. 模型评估与参数回调

**目的**：
    检验模型的**泛化能力**。大白话就是，让模型面对新的数据，看看它的输出结果是否理想，误差是否达到预期值

**操作**：
    使用从未参与过训练的测试集/验证集评估模型的性能，根据预先选定的评价指标和目测效果，评价模型的优劣，是否需要继续迭代。
    常见的指标：

    1. 准确率 Accuracy
    2. 精确率 Precision
    3. 召回率 Recall
    4. 还有 TN、TP、FP、FN等
  评价指标介绍[看此处](#评价指标)
  如果模型效果不理想，那么需要回来调整第六步模型训练里的[超参数](#超参数)，例如梯度下降法的学习率、神经网络的隐藏层层数
  如果仍旧不理想，就要考虑第三步、第四步（换模型）了。

### 8. 模型部署与监控

训练好模型后，最后一步就是部署上线，应用到实际的业务中，不然就纯纯是空中楼阁。一个不能解决实际问题的算法，毫无价值

1. 部署：将模型集成到后端中，供给前端调用
2. 监控：
   持续监控模型在真实世界中的表现。因为世界是变化的，过去的数据规律可能在未来失效（称为“模型漂移”）。监控的目的是及时发现模型性能衰退，以便进行重新训练或更新。

## 完整示例：构建垃圾邮件过滤器

### 需求分析

用户收到垃圾邮件骚扰，需要一个自动过滤器，滤掉垃圾邮件。

### 1.问题定义

- **明确业务问题**：

    构建一个过滤器，过滤垃圾邮件
- **机器学习问题**：
- 
    找到一个函数 $f$，输入一封电子邮件的内容（$X$），输出一个预测值 $\hat{y}$，判断这封邮件是垃圾邮件（Spam）还是正常邮件（Ham）
    分析可知，这是一个**二分类问题**，令 "垃圾邮件"编码为 1，"正常邮件" 编码为 0.
- **确定评价指标**：

    我们自然希望准确率 Accuracy最高，但是**更不希望一封重要的正常邮件被错判为垃圾邮件**。即，在预测为"垃圾邮件"的结果中，我们希望它是真正的垃圾邮件的比例能够达到最高。因此，相比于准确率，精确率（该封邮件是否是垃圾邮件）指标更加关键和重要

### 2.数据采集和EDA

1. 获取数据集：
   从网上下载一个公开的邮件数据集，里面有成千上万封邮件，并且都标注好了是 Spam 还是 Ham。
2. 清洗数据：
   检查数据是否有缺失、损坏或重复的条目。
   - 如果是少量的单个邮件的某条数据缺失，可直接删除此封邮件，或者使用一些值如 ' '来代替内容。
3. EDA：
   - 统计 Spam 和 Ham 的邮件各有多少封，看看数据是否平衡。如果不平衡，可采用两种方法：
        1. 数据量足够大时，可删除数量较多的一方的数据，直到两方平衡
        2. **更常见的做法是**：先用数量较少的一类数据集（假设是 Spam）进行训练，生成新的Spam，直到 Spam 的数量和 Ham相差不多，这称为 SMOTE，[详情看此部分](#SMOTE)
   - 制作 "词云"：
    查看 Spam邮件中最常出现的词（例如 "free", "viagra", "offer", "winner"），和 Ham中最常出现的词（如 "meeting", "report", "hello", "team"）。这能给我们一个直观的感受，认为二者的确有较为明显的差别，并且能够获得训练的数据

### 3.特征工程

由于算法模型，例如逻辑回归并不知道 "free viagra"这样的文本内容，只认识数字。
因此需要对文本内容采用向量化形式，将其转换（编码）为数学语言

**文本向量化**里常用的方法是 **词袋模型（Bag-of-Words）**：
1. 建立词汇表
   统计数据集中所有出现过的独立单词，形成一个巨大的词典（比如有5000个单词）
2. 向量化：
   对于每一封邮件，都创建一个长度为 5000 的向量。向量的每一个位置对应词典中的一个单词。如果词典中第 100 个单词是 "free"，而某封邮件里 "free" 出现了 2 次，那么这个向量在第100个位置的值就是 2。其他没出现的单词位置就是 0
   如此，通过词袋模型，就成功将每一封文本邮件，转换成了算法可处理的数字向量 $X$

### 4.模型选择

这是一个经典的二分类问题，并且特征（词向量）维度很高

逻辑回归算法就很不错，快速、简单。
因此选择的函数模板就是
$$
\hat{y} = \sigma(w^T X + b) \iff \hat{y} = \frac{1}{1 + e^{-(w^T X + b)}}
$$

### 5.模型训练

**前提**：已经划分好了训练集，拥有处理好的邮件向量 $X_train$ 和对应的标签 $y_train$ （0 或 1），即前文定义的 Spam是 1，Ham是 0

1. 初始化
    模型初始时是 无知 的。它内部的参数，即每个单词的权重 $w$，都是一个随机值
    
2. 前向传播 Prediction
   1. 从训练集获取一封邮件的向量 $X$
   2. 模型根据当前的权重 $w$，计算一个综合得分 score $z = wX+b$
   3. 使用 Sigmoid函数 $\sigma$ 将得分 $z$ 转换为一个概率 $\hat(y)$。例如，模型输出 $\hat{y}=0,4$，意思是 "模型认为这封邮件有40%的可能是垃圾邮件"
   
3. 计算损失函数 Loss function
    1. 现在，我们查看这封邮件的真实标签，发现它其实是垃圾邮件，即 $y = 1$
    
    2. 模型预测值 $\hat{y} = 0.4$，为了计算/量化真实值与预测值的误差，因此使用**损失函数**来量化该指标
    
    3. 分类问题常用的损失指标是 **交叉熵损失 Cross-Entropy Loss**
       1. 如果真实是1，预测值0.4，会有一个损失值。
       2. 如果真实是1，预测值0.1，损失值会急剧增大
       3. 如果真实是1，预测值0.9，损失值会非常小
       
       可以理解为，损失值是用来量化**误差**的指标，损失函数是用来计算损失值得
    4. 通过计算，我们得到了一个具体的损失值，比如 $Loss = 0.91 $。这个数字代表了模型在这一个样本上的表现有多差。
    
4. 反向传播和优化 Optimization
   
    现在，模型知道自己的损失值大小，但它不知道怎么去减小损失值，因此需要使用一个算法：梯度下降法来完成这件事
    梯度下降法的作用是计算出损失函数对每一个权重 $w$的梯度，将参数朝着梯度的反方向下降，进而损失值下降。
    - 例如，如果权重 $w\_ free$（即单词 free 的权重）的梯度是正的，那么它就会稍微增加 $w\_ free$的权重，进而使损失减小。因为增加$w\_ free$可以让 $z$变大，进而让概率 $\hat{y}$ 更接近 1
    - 如果权重 $w\_ meeting$的梯度是负的，意味着减小$w\_ meeting$可以让损失减小。
    可以想象为一个凹凸不平的曲面，曲面最低处为 0，上面有一小球，小球所处位置的 z 轴值就是损失值 $Loss$，要做的就是让小球滚向最低处。
    梯度下降法做的事是让小球滚向最低处的同时，滚下的速度最快
    对于每一个权重 $w$，对应的是一个独立的小球，所处的位置和其他小球无关
    
5. 迭代 Epoch
    不断重复第二步到第四步，处理下一封邮件里的单词的权重，直到完成一次 Batch_size，例如以64封邮件作为一个小批量训练样本的大小，这样就算为完成了一次参数的更新。当模型遍历完一遍所有的训练数据后，就完成了一次 Epoch。
    一般都会采用小批量样本训练，而不是直接把所有训练样本塞给模型，显卡也吃不消
    **结果**：
    经过一轮轮的迭代，模型的权重 $w$会达到一个较好的状态。例如正常邮件 Ham的词 "free"对应的权重很高，而 Spam里频繁出现的词 "meeting" 对应的权重很低，这样模型就学会了怎么区分垃圾邮件

### 6.模型评估与参数回调

1. **评估**：
    训练好模型后，下一步就是对模型进行测试，评估它的能力。
    第一步中已经确定核心评价指标为 **Precision**
    使用前面划分数据集得到的测试集对模型进行测试，得到输出 $\hat{y}\_ test$，再与标签 $y\_ test$ 对比，计算得到 $Precision$
2. **回调参数**
    根据 Precision值是否满足预先设置的误差以内，如果效果不好，需要进行参数回调
    - 回调梯度下降法的超参数：学习率
    - 更换词袋模型，换为 TF-IDF 特征等等

### 7.模型部署与监控

拿到模型后自然要部署上线使用。
个人技术栈为 fastAPI + VUE，因此很容易就集成到后端里

## 关键技术要点

### 机器学习算法分类

- **分类算法**

    目标：预测一个**离散的类别标签**。
    例如：“是垃圾邮件/不是垃圾邮件”、“是猫/是狗/是鸟”、“信用良好/信用一般/信用差”。
  - 逻辑回归 Logistic Regression
  - 支持向量机 Support Vector Machine - SVM
  - 决策树 Decision Tree
  - 随机森林 Random Forest
  - K-近邻 K-Nearest Neighbors - KNN
  - 朴素贝叶斯 Naive Bayes
  - 梯度提升决策树 Gradient Boosting Decision Trees - GBDT, XGBoost, LightGBM
  - 神经网络/多层感知机  Neural Networks/MLP
  
- **回归算法**

    目标：预测一个**连续的数值**
    例如：预测房价、预测股票价格、预测明天的气温。
  - 线性回归  Linear Regression
  - 岭回归  Ridge Regression
  - Lasso 回归  Lasso Regression
  - 支持向量回归  Support Vector Regression - SVR
  - 决策树回归 / 随机森林回归 / GBDT回归
  - 神经网络回归

- **聚类算法**

    目标：在**没有标签**的情况下，自动将数据分成不同的组（簇），使得同一组内的数据点相似，不同组之间的数据点相异
    这是一种无监督学习
  - K-均值  K-Means
  - 次聚类  Hierarchical Clustering
  - DBSCAN  Density-Based Spatial Clustering of Applications with Noise
  - 高斯混合模型  Gaussian Mixture Model - GMM

---

### 评价指标

详情<a href="https://www.juayohuang.top/posts/ai/machine-learning/mlevaluationmertrics" target = "_blank" rel="noopener noreferrer">请点击此处查看机器学习评价指标文档</a>，此处仅给出简要介绍

#### 分类指标

**评价的基础**：  混淆矩阵  Confusion Matrix

所有分类模型的评估指标，都源于一个简单的表格，叫做**混淆矩阵**，它用于告诉开发者模型在哪些地方做得好，哪些地方是错误的

**假设**：现在训练好了一个模型，用于检测邮件是否是垃圾邮件 Spam

那么有两个术语，用来描称呼 Spam和 Ham：

- 正类 Positive： 是垃圾邮件
- 负类 Negative： 不是垃圾邮件

现在，我们使用模型预测了100封邮件，混淆矩阵看起来是这样的：
|                         | 预测为：垃圾  Positive | 预测为：正常  Negative |
| :---------------------: | :--------------------: | :--------------------: |
| **实际：垃圾 Positive** |      **TP = 10**       |       **FN = 5**       |
| **实际：正常 Negative** |       **FP = 2**       |      **TN = 83**       |

这四个格子是所有计算的核心：

- **TP True Positive - 真阳性**: 
    ✅️ 模型正确地将垃圾邮件预测为了垃圾邮件（10封）
- **TN True Negative - 真阴性**： 
    ✅️ 模型正确地将正常邮件预测为了正常邮件（83封）
- **FP False Positive - 假阳性 / Type I Error**: 
    ❌️ 模型错误地将正常邮件预测为了垃圾邮件。这是 “误报”，把重要的邮件错杀进了垃圾箱（2封）
- **FN False Negative - 假阴性 / Type II Error**: 
    ❌️ 模型错误地将垃圾邮件预测为了正常邮件。这是 “漏报”，让垃圾邮件溜进了你的收件箱（5封）

拥有了以上四个数据，就能够用来计算分类算法里的各种评价指标：

1. **准确率 Accuracy**’
   $$
     Accuracy = \frac{所有预测正确的样本}{总样本数}
   
     \\ Accuracy = \frac{TP + TN}{TP+TN+FP+FN}
   $$
   
2. **精确率 Precision**
   $$
       Precision = \frac{真正是正类的}{所有被预测为正类的}
       \\ Precision = \frac{TP}{TP+FP}
   $$

3. **召回率  Recall/ Sensitivity/ True Positive Rate**
   $$
       Recall = \frac{真正是正类的}{所有实际上是正类的}
       \\ Recall = \frac{TP}{TP+FN}
   $$

4. **F1分数 F1-Score**
   $$
       F1-Score = 2 * \frac{Precision * Recall}{Precision + Recall}
       \\ F1-Score = 2 * (\frac{1}{Precision} + \frac{1}{Recall})
   $$

**代码示例**

Scikit-learn 提供了非常方便的工具

```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# 假设 y_true 是真实标签, y_pred 是模型预测的标签
y_true = [...] 
y_pred = [...]
# 1. 直接打印所有核心指标
print(classification_report(y_true, y_pred, target_names=['Not Diabetic', 'Diabetic']))
# 2. 生成并可视化混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Diabetic', 'Diabetic'], yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

#### 回归指标

1. **平均绝对误差** Mean Absolute Error - MAE
   $$
   MAE = \frac{1}{n} * \sum (|y_i - \hat{y}_i|)
   $$

2. **均方误差** Mean Squared Error - MSE
   $$
   MSE = \frac{1}{n} * \sum (y_i - \hat{y}_i)^2
   $$
   
3. **均方根误差** RMSE
   $$
    MSE = \sqrt{MSE} = \sqrt{\frac{1}{n} * \sum (y_i - \hat{y}_i)^2}
   $$

4. **R平方** R-Squared / 决定系数
   $$
           R^2 = 1-\frac{模型误差平方}{基准模型误差平方和}
           \\ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_i)^2}
   $$

**代码示例**

Scikit-learn 提供了所有这些工具

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 假设 y_true 是真实值, y_pred 是模型预测的值
y_true = [30, 50, 100, 22]
y_pred = [28, 55, 90, 25]
# 计算 MAE
mae = mean_absolute_error(y_true, y_pred)
# 计算 MSE
mse = mean_squared_error(y_true, y_pred)
# 计算 RMSE
rmse = np.sqrt(mse) # 或者 mean_squared_error(..., squared=False)
# 计算 R-Squared
r2 = r2_score(y_true, y_pred)
```

---

#### 聚类指标

由于聚类学习是**无监督学习**，因此我们是没有**真实的标签**来和模型输出的数据做对比的，它的指标比较特殊

**外部评价指标**

- 调整兰德指数 Adjusted Rand Index - ARI
- V-measure V-度量

**内部评价指标（核心）**

- **轮廓系数 Silhouette Coefficient**
- **方差比标准 Calinski-Harabasz 指数**
- **Davies-Bouldin 指数**

**实践中使用**

内部评价指标最常见的用途就是帮助我们选择**最佳的聚类数量 (k)**，这通常被称为**"肘部法则(Elbow Method)"** 的扩展应用

1. 设定一个 k 的范围，例如从2到10
2. 对每个 k 值，运行聚类算法（如K-Means）
3. 计算一个内部评价指标（例如**轮廓系数**）
4. 将 k 值与对应的指标分数画成折线图。
5. **寻找拐点**:
   - 对于**轮廓系数**或**Calinski-Harabasz指数**，寻找**分数最高**的那个 k 值
   - 对于**Davies-Bouldin指数**，寻找**分数最低**的那个 k 值

**Scikit-learn中的实现**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score # 外部指标示例
# 假设 X 是特征数据
# 假设 labels_true 是真实标签 (如果有的话)
# 运行聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_pred = kmeans.fit_predict(X)
# 轮廓系数 (越高越好)
s_score = silhouette_score(X, labels_pred)
# Calinski-Harabasz 指数 (越高越好)
ch_score = calinski_harabasz_score(X, labels_pred)
# Davies-Bouldin 指数 (越低越好)
db_score = davies_bouldin_score(X, labels_pred)
# 如果有真实标签，计算外部指标
# ari_score = adjusted_rand_score(labels_true, labels_pred)
```

---

### 数据清洗

主要为 缺失值、异常值、重复值的处理，查看
<a href="https://www.juayohuang.top/posts/ai/machine-learning/pandas" target = "_blank" rel="noopener noreferrer">pandas用法</a>

### SMOTE

Synthetic Minority Oversampling Technique，合成少数类过采样技术

**核心思想**:  "不直接复制，而是在现有的较少量样本之间，创造出一些新的、看起来很像真实样本的‘合成’样本。"

**操作方法**：

以健康和患病人数不平衡为例：
假设人数比：健康：患病 = 7：3，为了获得更多的患病样本数，我将采取以下措施：

1. 选取合成样本模板：
   在现有患病样本中（例如 300个），随机选取一个样本 A
2. 寻找相似特征样本
   在剩下的样本中，寻找与 A样本的特征最相似的几个"邻居"样本（比如最近的 5 个）
3. 创造新数据
   从这几个邻居中，随机挑选一个邻居 B，然后在 A 和 B 所连接的线段上的某个随机点，生成一个新样本
4. 重复上述步骤，直到类别数量平衡

**优势**：减轻过拟合问题

**实际工具**

有一个专门处理不平衡数据的库 imbalanced-learn (通常简写为 imblearn)，它可以和 scikit-learn 完美集成，在 PyTorch中也可以使用，只需要在生成 Dataset对象前对原始数据集做处理即可

1. 安装
   ```bash
   pip install imbalanced-learn
   ```
2. 使用
   ```python
   from imblearn.over_sampling import SMOTE
   # 测试集必须保持原样，因为它代表了真实世界的数据分布。
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   print(f"原始训练集样本分布: {Counter(y_train)}")
   # 示例输出: 原始训练集样本分布: Counter({0: 400, 1: 180})
   # 1. 创建 SMOTE实例 random_state是为了结果可复现
   smote = SMOTE(random_state=42)
   # 2. 对数据集采样 .fit_resample() 会返回经过重采样的数据
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
   print(f"经过SMOTE重采样后的训练集样本分布: {Counter(y_train_resampled)}")
   # 示例输出: 经过SMOTE重采样后的训练集样本分布: Counter({0: 400, 1: 400})
   # 3. 使用重采样的数据来训练模型
   # 4. 在原始的、未动过的测试集进行评估
   ```

---

### 过拟合

**过拟合 Overfitting** 是机器学习中最重要的核心概念之一

1. **表现方式**

    一个模型在训练数据上表现得很完美，但在**新的、未见过的数据（测试数据）**上表现非常糟糕。模型没有学到通用的规律，而是 "记忆" 了训练数据中所有的细节，甚至是噪声
    以一个例子说明：
    假设有两个学生 A和 B，它们在准备一场考试。老师给了他们一套**复习题集（训练集）**
    - A：理想模型
    
       - 努力学习 题库的知识，掌握背后的原理和知识点
       - 总结出了自己的一套解题方法
       - 考试结果：
            考场上有很多**新题目（测试集）**，虽然 A没有见过题目，但是它知道这些题目背后的原理，因此能考得很好 
    - B：过拟合模型

        - 死记硬背 题库里的每一道题和标准答案
        - 不会理解背后的原理，而是把每一道题，或者说把这一整本复习资料给背下来，哪一题在哪一页都记得一清二楚
        - 在 **模拟测试（训练集上评估）** 表现完美，每一题都能一字不差地答出来，次次100分
        - 考试结果：
            因为这些新题都没见过，虽然考的知识点一样，但问法和数字都变了。B因为只会背答案，完全无法应对这些新情况，最终考了个很差的分数
   
2. **原因**

   1. 模型过于复杂
        例如使用 一个拥有上百万参数的模型，例如深度神经网络去学习一个十分简单的问题，足以把所有训练数据的细节硬生生记下来
        例如让一个 博士去准备小学生的期末考试
   2. 数据样本太少
        模型得到的 复习题集太小，只有几道题。这还学个蛋，直接背不就完了
   3. 数据噪声太多 
        训练数据中包含很多错误或随机的干扰信息

3. **图像上识别过拟合**

    在训练模型时，我们通常会监控两个关键指标：训练损失 Training Loss 和 验证损失 Validation Loss

   - **训练损失**: 模型在它正在学习的数据集上的表现
   - **验证损失**: 模型在一个它没有见过的、预留出来的验证集上的表现

    **大致表现**
   1. **初始阶段**
        训练刚开始，模型在学习通用规律。此时，训练损失和验证损失都在**一起下降**。这说明模型学到的东西既适用于训练数据，也适用于新数据。
   2. **过拟合开始点 The "Elbow"**
        训练继续进行，模型已经掌握了大部分通用规律。它开始“没事找事”，去学习训练数据中的**噪声和特有细节**。此时，**训练损失继续下降**（因为它在更精细地“背题”），但**验证损失开始停止下降，甚至掉头上升**。
   4. **严重过拟合**
        两个曲线的**差距越来越大**。训练损失趋近于0，而验证损失越来越高。模型已经变成了一个只会“背题”的“书呆子”。


4. **解决办法**

   这些方法统称为**正则化 (Regularization)**，意思是“施加规则，防止模型为所欲为

   1. **获取更多数据**

      **最有效但最昂贵**的方法

   2. **数据增强**

      一种 “免费” 获得更多数据的方式

      比如在图像识别任务中，将一张猫的图片进行轻微的旋转、翻转、裁剪、改变亮度，这些依然是猫，但对模型来说是 “新” 数据。

   3. **简化模型**

      更换更简单的模型训练

   4. **Dropout (随机失活) - [神经网络常用]**

      在训练过程的每一步，随机地 “冻结” 一部分神经元，让它们不参与工作

   5. **L1/L2 正则化**

      在模型的损失函数中增加一个 “惩罚项”，这个惩罚项与模型参数（权重）的大小有关。如果模型的某些参数变得非常大（意味着它过度依赖某个特征），就会受到惩罚

   6. **早停**

      **非常实用**的方法

      在训练过程中，持续监控**验证集**的损失。一旦发现验证损失连续多个轮次不再下降，甚至开始上升，就**立即停止训练**，并保存验证损失最低时的那个模型

---

### 算法详细介绍

具体的推导和介绍<a href="机器学习算法" target = "_blank" rel="noopener noreferrer">请点击此处</a>（未编写文章）

#### 分类算法 

**目标**：预测一个**离散的类别标签**。

例如：“是垃圾邮件/不是垃圾邮件”、“是猫/是狗/是鸟”、“信用良好/信用一般/信用差

|                           算法名称                           |                           核心思想                           |                             优点                             |                             缺点                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                逻辑回归 (Logistic Regression)                | 虽然名字带“回归”，但它通过Sigmoid函数将线性回归的输出映射到0-1之间，用于预测概率，从而进行二分类或多分类。 | **简单、快速、可解释性强 (可以通过权重判断特征重要性)**，适合作为基线模型。 |            对非线性关系拟合能力弱，需要特征工程。            |
|          支持向量机 (Support Vector Machine - SVM)           | 在特征空间中找到一个能将不同类别样本分隔开的“最优超平面”（间隔最大）。 | **在高维空间中非常有效**，对于非线性问题，可以通过**核技巧**实现高效划分。 | 对数据量大的情况训练较慢，对缺失数据敏感，参数和核函数的选择比较讲究。 |
|                    决策树 (Decision Tree)                    | 通过一系列“是/否”问题（类似流程图）来对数据进行划分，最终到达一个叶子节点，即分类结果。 | **模型非常直观，易于理解和解释**，可以处理数值型和类别型数据。 |          **容易过拟合**，单个树的泛化能力可能不强。          |
|                   随机森林 (Random Forest)                   | **集成学习**的代表。通过构建**多棵决策树**，并让它们投票来决定最终的分类结果。 | **极大地减少了单棵决策树的过拟合风险**，性能强大且稳定，不易受到噪声影响。 |  模型的可解释性比单棵决策树差，训练和预测速度比单个模型慢。  |
|              K-近邻 (K-Nearest Neighbors - KNN)              | “物以类聚”。一个新样本的类别由它在特征空间中最近的K个邻居的类别来决定。 |         **算法非常简单，无需训练**，对异常值不敏感。         | **预测速度慢**（需要计算与所有点的距离），对特征尺度敏感（需要标准化），需要手动选择K值。 |
|                   朴素贝叶斯 (Naive Bayes)                   | 基于贝叶斯定理，并假设特征之间相互**独立**（这是“朴素”的来源），来计算样本属于某个类别的概率。 | **算法简单，计算速度快**，在**文本分类**（如垃圾邮件检测）等领域效果非常好。 | “特征独立”的假设在现实中往往不成立，因此在某些情况下性能可能不佳。 |
| 梯度提升决策树 (Gradient Boosting Decision Trees - GBDT, XGBoost, LightGBM) | **集成学习**的另一种形式。它迭代地构建一系列弱的决策树，每一棵新树都试图纠正前面所有树的错误。 | **性能极其强大，是各种数据科学竞赛中的大杀器**，能够处理非常复杂的关系。 |   **训练过程是串行的，可能比较慢**，调参比随机森林更复杂。   |
|          神经网络/多层感知机 (Neural Networks/MLP)           | 模拟人脑神经元结构，通过多个层和非线性激活函数来学习数据中极其复杂的模式。 | **能拟合任意复杂的非线性关系，在图像、语音、文本等非结构化数据上表现最佳**。 |    **需要大量数据，训练成本高，模型是黑箱，可解释性差**。    |

------

#### 回归算法 

**目标**：预测一个**连续的数值**

例如：预测房价、预测股票价格、预测明天的气

|                    算法名称                    |                           核心思想                           |                             备注                             |
| :--------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|          线性回归 (Linear Regression)          | 找到一条直线（或一个超平面）来最好地拟合数据点，使得预测值与真实值之间的误差平方和最小。 |          **最简单、最基础的回归算法**，可解释性强。          |
|           岭回归 (Ridge Regression)            | 在线性回归的损失函数上增加一个**L2正则化项**，用于惩罚过大的模型权重。 |         主要用于**解决多重共线性问题和防止过拟合**。         |
|         Lasso 回归 (Lasso Regression)          |        在线性回归的损失函数上增加一个**L1正则化项**。        | 除了能防止过拟合，L1正则化还能将某些不重要的特征权重变为0，从而**实现特征选择**。 |
| 支持向量回归 (Support Vector Regression - SVR) | 类似于SVM，但它试图找到一个超平面，让尽可能多的数据点落在离这个平面一定距离的“管道”内。 |             对于处理异常值（outliers）比较鲁棒。             |
|      决策树回归 / 随机森林回归 / GBDT回归      | 这些分类算法都有对应的回归版本。它们不是预测类别，而是在叶子节点上预测一个**平均值**。 | 能够很好地捕捉数据中的**非线性关系**。例如，随机森林回归和GBDT回归都是非常强大的回归工具。 |
|                  神经网络回归                  | 神经网络的输出层不接分类函数（如Softmax），而是直接输出一个或多个连续值。 |              能够拟合极其复杂的非线性回归问题。              |

---

#### 聚类算法

**目标**：在**没有标签**的情况下，自动将数据分成不同的组（簇），使得同一组内的数据点相似，不同组之间的数据点相异。这是一种**无监督学习**。

| 算法名称                                                     | 核心思想                                                     | 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| K-均值 (K-Means)                                             | 预先指定要分成的簇数K，然后迭代地将每个数据点分配给最近的簇中心，并更新簇中心的位置，直到簇中心不再变化。 | **算法简单、快速，易于理解**，是应用最广泛的聚类算法。       | **需要预先指定K值**，对初始簇中心的选择敏感，对非球形的簇和异常值处理不好。 |
| 层次聚类 (Hierarchical Clustering)                           | 通过一种层次化的方式来构建簇。可以是“自底向上”的凝聚型（从小簇合并成大簇）或“自顶向下”的分裂型。 | **无需预先指定簇数K**，可以得到一个漂亮的树状图（Dendrogram），有助于理解数据结构。 | 计算复杂度较高，特别是对于大数据集。                         |
| DBSCAN (Density-Based Spatial Clustering of Applications with Noise) | 基于密度的聚类算法。它将密集区域中的点连接起来形成簇，并将稀疏区域中的点识别为噪声或异常点。 | **可以发现任意形状的簇**，并且能**自动识别噪声点**，无需预先指定簇数。 | 对密度差异大的数据集效果不佳，对高维数据效果不佳。           |
| 高斯混合模型 (Gaussian Mixture Model - GMM)                  | 假设数据是由若干个高斯分布混合生成的。算法的目标是找到这些高斯分布的参数。 | 能够处理**重叠的、椭球形**的簇，提供每个点属于各个簇的概率（软聚类）。 | 算法相对复杂，对初始值敏感。                                 |

---

#### 各算法的详细介绍和PyTorch使用方法（未编写文章）

1. <a href="./回归算法" target = "_blank" rel="noopener noreferrer">回归算法</a>
2. <a href="./分类算法" target = "_blank" rel="noopener noreferrer">分类算法</a>
3. <a href="./聚类算法" target = "_blank" rel="noopener noreferrer">聚类算法</a>

---

### 损失函数

损失函数是用于判断 模型当前轮次预测效果好坏 和 指导模型该如何提高预测效果的函数

这里只简单介绍具体有什么损失函数、常用的损失函数、代码中使用

若要查看相关介绍，<a href="https://www.juayohuang.top/posts/ai/machine-learning/lossfunction">请点击此处</a>

#### 分类损失

1. **二元分类问题下的损失函数**

   二元交叉熵损失

   实现代码：

   1. 在 Scikit-learn 中

        在Scikit-learn中，通常**不会直接选择交叉熵损失函数**

        对于像逻辑回归这样的分类模型，交叉熵损失是其内置的、默认的优化目标

        调用 `.fit()` 方法训练一个逻辑回归模型时，它内部的  “求解器 (solver)”  就在努力地最小化交叉熵损失，只需创建和训练模型即可

        ```python
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification
        # 1. 准备数据
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # 2. 创建模型
        # 交叉熵损失是这个模型内在的优化目标
        model = LogisticRegression()
        # 3. 训练模型
        # .fit() 的过程就是在最小化交叉熵
        model.fit(X_train, y_train)
        # 你可以使用 log_loss 来“计算”损失值，但这只是评估，不是训练过程的一部分
        from sklearn.metrics import log_loss
        predictions_proba = model.predict_proba(X_test)
        loss_value = log_loss(y_test, predictions_proba)
        ```

   2. 在 PyTorch 中

        PyTorch中，需要**显式地定义和使用**损失函数 `torch.nn.BCEWithLogitsLoss`

        ```python
        import torch
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss()
        model_output = torch.randn(10, 1) # 10个样本，模型的原始输出 (logits)
        true_labels = torch.randint(0, 2, (10, 1)).float() # 10个真实标签 (0或1)
        loss = criterion(model_output, true_labels)
        ```

2. 多类别分类
   
   分类交叉熵损失使用 `torch.nn.CrossEntropyLoss`
   这个函数**内部集成了 Softmax 和计算负对数损失**两个步骤，这意味着模型最后一层**绝对不能**有Softmax激活函数，直接输出原始得分（logits）即可
   
   ```python
   import torch
   import torch.nn as nn
   criterion = nn.CrossEntropyLoss()
   # 3个样本，模型对4个类别的原始输出 (logits)
   model_output = torch.randn(3, 4) 
   # 3个真实标签 (不是one-hot，直接是类别索引)
   true_labels = torch.tensor([1, 0, 3]) 
   loss = criterion(model_output, true_labels)
   ```

3. 多标签分类
   
   二元交叉熵损失：将  “N选K”  问题分解为 **N 个独立的二元分类问题**

|    任务类型    |   类别关系    |   模型最后一层激活   |              **首选损失函数**              |
| :------------: | :-----------: | :------------------: | :----------------------------------------: |
|  **二元分类**  |    二选一     |     **Sigmoid**      |   **二元交叉熵 (Binary Cross-Entropy)**    |
| **多类别分类** | N选一 (互斥)  |     **Softmax**      | **分类交叉熵 (Categorical Cross-Entropy)** |
| **多标签分类** | N选K (不互斥) | **N个独立的Sigmoid** |   **二元交叉熵 (Binary Cross-Entropy)**    |

---

#### 回归损失

与分类损失函数衡量类别猜得对不对不同，回归损失函数的核心任务是  **衡量预测值与真实值之间的距离或误差有多大**

1. 均方误差 MSE / L2 损失
2. 平均绝对误差 Mean Absolute Error - MAE / L1 损失
3. 均方根误差  Root Mean Squared Error - RMSE
4. Huber 损失  Huber Loss  /   平滑平均绝对误差

**总结**

|    损失函数    |               核心特点               |   对异常值   |               何时使用               |
| :------------: | :----------------------------------: | :----------: | :----------------------------------: |
|  **MSE (L2)**  |       **平方误差**，惩罚大错误       | **非常敏感** | 默认选择，数据干净，希望修正大误差时 |
|  **MAE (L1)**  |        **绝对误差**，一视同仁        | **非常稳健** |          数据中存在异常值时          |
|    **RMSE**    |             MSE的平方根              | **非常敏感** |  主要作为**评估指标**，因其单位直观  |
| **Huber Loss** | **混合体**，小误差用MSE，大误差用MAE | **比较稳健** |     想要两全其美，既稳健又高效时     |

**代码实现**

1. 在 Scikit-learn 中

     与分类类似，损失函数通常是模型内置的。例如 LinearRegression 默认使用MSE

     但你可以使用 metrics 模块来评估模型在不同损失函数下的表现，不是指定 $Loss\ Function$

     ```python
     from sklearn.metrics import mean_squared_error, mean_absolute_error
     # 假设 y_true 和 y_pred 是 NumPy 数组
     y_true = [3, -0.5, 2, 7]
     y_pred = [2.5, 0.0, 2, 8]
     # 计算 MSE
     mse = mean_squared_error(y_true, y_pred)
     # 计算 MAE
     mae = mean_absolute_error(y_true, y_pred)
     # 计算 RMSE
     rmse = mean_squared_error(y_true, y_pred, squared=False) # 或者 np.sqrt(mse)
     ```

2. 在 PyTorch 中

     PyTorch中需要显式地定义训练时使用的损失函数

     ```python
     import torch
     import torch.nn as nn
     # 创建损失函数实例
     criterion_mse = nn.MSELoss()
     criterion_mae = nn.L1Loss() # MAE在PyTorch中被称为L1Loss
     criterion_huber = nn.HuberLoss(delta=1.0)
     # 准备数据
     predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
     targets = torch.tensor([3.0, -0.5, 2.0, 7.0])
     # 计算损失
     loss_mse = criterion_mse(predictions, targets)
     loss_mae = criterion_mae(predictions, targets)
     loss_huber = criterion_huber(predictions, targets)
     ```
---

### 梯度下降法及其变种

受限于篇幅，此处仅介绍

- 梯度下降法的作用
- 在模型训练中扮演的角色
- 常用的梯度下降法
- 代码实现

<a href = "https://www.juayohuang.top/posts/ai/machine-learning/gradientdescentmethod" target="_blank" rel="noopener noreferrer">详细的算法介绍请点击此处</a>

1. **作用**

   梯度下降法的作用就是让模型当前的损失值 $Loss$下降，最终目标是下降到最低点

   假设你位于喜马拉雅山脉的某一点上，你需要以最快的速度下山，并且不需要考虑自身的安全问题

   怎么找到**每一步**都是下山最快的一步？ $\Rightarrow$  梯度下降法就是用来做这个事情的

   具体实现方式就是每一次都沿着梯度$\Delta$ 的反方向前进

2. **扮演的角色**

   根据前文，我们已经得知，$Loss\ Function$会计算得到损失值 $Loss$，模型需要根据 $Loss$ 是否满足精度要求进而判断是否需要进一步的 参数更新

   一轮经典的训练步骤如下：

   1. **模型做出预测**

      模型根据当前输入和当前的参数值，进行计算得到一个预测值 $\hat{y}$

   2. **损失函数评估预测的好坏**

      $Loss\ Function$根据预测值和真实值计算误差，得到一个$Loss$

      注意：误差 $Error \neq Loss$ 

   3. **梯度下降法执行反向传播**

      使用误差反向传播，得到损失函数相对**每一个模型参数**的梯度

   4. **优化器更新模型参数**

      优化器（梯度下降法的执行者）根据上一步计算得到的梯度，朝着梯度的反方向，对模型的**每一个参数**进行一次微小的更新

3. **常用的梯度下降法**

   1. 随机梯度下降法 SGD
   2. **小批量梯度下降法** 
   3. 整体梯度下降法
   4. 动量梯度下降法
   5. AdaGrad算法
   6. RMSProp算法
   7. Adam算法

4. **代码实现**

   1. sklearn

      在 Scikit-learn 中，梯度下降法通常是**隐藏在模型内部**的。Scikit-learn 高度封装，让你更专注于模型选择和超参数调整，而不是训练过程的细节

      在 Scikit-learn 中，使用梯度下降法，实际上就是**实例化一个使用梯度下降的模型，并调整其相关超参数**

      以线性回归为例：$y = w * x + b$

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
      # 2. 实例化模型，并配置梯度下降法
      # 在这里通过设置超参数来控制梯度下降的行为
      model = SGDRegressor(
          loss='squared_error', # 损失函数：均方误差 (MSE)
          penalty=None,         # 不使用正则化
          max_iter=1000,        # 最大迭代次数 (相当于 PyTorch 的 epochs)
          tol=1e-3,             # 如果损失改善小于这个值，就提前停止
          eta0=0.01,            # 初始学习率 (相当于 PyTorch 的 learning_rate)
          verbose=1             # 每隔一段时间打印一次训练进度
      )
      # 3. 训练模型：梯度下降法在此处
      # .fit() 这一行代码会自动进行：预测 -> 计算损失 -> 计算梯度 -> 更新参数 的迭代过程
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
      
      > Scikit-learn中更多的优化器介绍<a href= "https://www.juayohuang.top/posts/ai/machine-learning/sklearn_optimizer" target="_blank" rel="noopener noreferrer">[请点击此处]</a>
      
   2. PyTorch
   
      PyTorch 已经将复杂的梯度计算和参数更新封装得极其简单，使用者只需要配置它，而不需要自己实现。
   
      以线性回归为例：$y = w * x + b$
   
      ```python
      import torch
      import torch.nn as nn
      import torch.optim as optim
      # 1. 准备数据
      X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
      y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32) # y = 2 * X
      # 2. 定义模型
      # 简单的线性模型 y = w * x + b
      model = nn.Linear(1, 1) 
      # 3. 定义损失函数和优化器
      # 超参数学习率
      learning_rate = 0.01
      criterion = nn.MSELoss() # 损失函数：均方误差
      # 梯度下降法在此使用
      # 选择SGD 作为优化器
      # 要优化的参数model.parameters(), 学习率
      optimizer = optim.SGD(model.parameters(), lr=learning_rate)
      # 4. 训练循环
      epochs = 20
      for epoch in range(epochs):
          # 做出预测
          y_pred = model(X)
          # 计算损失
          loss = criterion(y_pred, y)
          # 计算梯度
          # 在PyTorch中，这两行代码会自动完成所有复杂的梯度计算
          optimizer.zero_grad() # 清空上一轮的梯度
          loss.backward()     # 反向传播，计算当前损失下所有参数的梯度
          # 更新参数
          # 优化器根据计算好的梯度，更新模型的所有参数
          optimizer.step()
          if (epoch+1) % 2 == 0:
              print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
      # 5. 查看结果
      # 理想情况下，w应该接近2，b应该接近0
      [w, b] = model.parameters()
      print(f"训练后的权重 w: {w[0][0].item():.3f}, 偏置 b: {b[0].item():.3f}")
      ```
   
      - 不需要手动计算梯度，`loss.backward()`会自动完成
   
      - 不需要手动更新参数
   
        `optimizer.step()`会根据`loss.backward()`计算出的梯度，自动执行 $新参数 = 旧参数 - 学习率 * 梯度$ 这个操作。
   
      - 开发者，核心工作就是选择并配置优化器
   
        除了SGD，还可以选择  `optim.Adam`、`optim.RMSprop`等更高级的梯度下降法变体，但它们的核心角色都是一样的：读取梯度，更新参数



