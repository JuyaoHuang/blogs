---
title: 10-12学习路线
author: JuyaoHuang
published: 2025-10-21
description: "25年10~12月学习路线"
first_level_category: "知识库"
second_level_category: "学习路线"
tags: ['AI agent']
draft: false
---

# PyTorch学习路线

## 1. 经典机器学习

- **机器学习核心概念**:
  - 监督学习 vs. 无监督学习
  - 分类问题 vs. 回归问题
  - 特征 (Feature) 和标签 (Label)
  - 训练集 (Training Set) 和测试集 (Test Set)
- **数据预处理**:
  - **特征工程**: 理解其重要性，如识别和处理无效数据（例如将0值替换为均值/中位数）。
  - **数据标准化/归一化**: 理解为什么需要它，以及 StandardScaler 的工作原理。
- **Scikit-learn 库**:
  - **数据划分**: model_selection.train_test_split
  - **数据预处理**: preprocessing.StandardScaler
  - **模型**: linear_model.LogisticRegression (逻辑回归)
  - **模型训练与预测**: .fit() 和 .predict() 方法
  - **模型评估**: metrics.accuracy_score (准确率), metrics.confusion_matrix (混淆矩阵)
  - **模型解释**: 如何从训练好的模型中提取信息，如 .coef_ (特征权重)

## 2. 深度学习与PyTorch

- **PyTorch入门**:
  - **官方教程是最好的老师**: 跟随PyTorch官网的 "60 Minute Blitz" (60分钟入门) 教程。它会带你走过Tensor、数据加载、模型构建和训练的全过程。
  - **拆解学习**:
    1. **数据加载**: 专门学习torchvision.datasets.CIFAR10和DataLoader如何配合使用，加载并批量化数据。
    2. **模型构建**: 练习用nn.Module搭建一个简单的CNN。先模仿，再尝试自己修改。
    3. **训练循环**: 把训练循环的五个核心步骤背下来，并理解每一步的作用。这是PyTorch的灵魂。

- **深度学习核心概念**:
  - 神经网络、多层感知机 (MLP)
  - **卷积神经网络 (CNN)**: 核心组件及其作用（卷积层 Conv2d, 池化层 MaxPool2d, 激活函数 ReLU）
  - 前向传播 (Forward Pass) 和反向传播 (Backpropagation)
  - 损失函数 (Loss Function)，特别是**交叉熵损失 (Cross-Entropy Loss)**
  - 优化器 (Optimizer)，如 SGD 和 **Adam**
  - Epoch, Batch Size, Iteration 的概念
  - 过拟合 (Overfitting) 和欠拟合 (Underfitting)
- **PyTorch 框架**:
  - **Tensor (张量)**: PyTorch的基本数据结构，及其与NumPy数组的转换。
  - **数据集处理**:
    - torchvision.datasets (如 CIFAR10)
    - torchvision.transforms (数据预处理和增强，如 ToTensor, Normalize)
    - torch.utils.data.Dataset 和 DataLoader (构建数据加载管道)
  - **模型构建**:
    - torch.nn.Module: 构建自定义网络模型的基类
    - 常用层: nn.Conv2d, nn.MaxPool2d, nn.Linear, nn.ReLU, nn.Flatten
  - **训练流程**:
    - 定义损失函数 (如 nn.CrossEntropyLoss)
    - 定义优化器 (如 optim.Adam)
    - **训练循环 (Training Loop)**:
      1. optimizer.zero_grad() # 清空梯度
      2. outputs = model(inputs) # 前向传播
      3. loss = criterion(outputs, labels) # 计算损失
      4. loss.backward() # 反向传播
      5. optimizer.step() # 更新权重
  - **模型评估与使用**:
    - model.train() 和 model.eval() 模式的切换
    - torch.no_grad(): 在评估时关闭梯度计算
    - 模型保存与加载 (torch.save, model.load_state_dict)
  - **可视化**:
    - torch.utils.tensorboard.SummaryWriter: 使用TensorBoard记录训练过程

---

# AI agent构建学习路线

## 前提：已学习了机器学习基础和PyTorch

## 至少要掌握的AI算法内容

作为一个工程师，不需要深入学习算法的底层原理，但起码要知道这些算法是什么、怎么工作、以及如何构建并参与到工作流中

### 机器学习

- **过拟合**

  ​	这是最重要的概念。

  ​	需要能判断出Agent在某些任务上表现不佳，是不是因为它“背诵”了某些模式而不是学会了泛化

- **分类 vs. 回归**

  理解Agent在做决策时，本质上是在进行哪种类型的预测。

- **评估指标**

  ​	知道什么时候用**准确率**，什么时候用**精确率/召回率**。比如，一个医疗诊断Agent，你绝对不希望它漏诊（高召回率）

### 神经网络

知道：

- **基本结构**: 

  ​	知道它是由“层”(Layers)和“神经元”组成的，并且通过“激活函数”加入了非线性，这使得它能学习复杂模式

- **输入与输出**

  ​	明白一个神经网络接收什么样的数据（通常是数字向量/张量），输出什么样的数据

- **“深度”的意义**

  ​	直观理解为什么网络越深，通常能学习到的特征越抽象、越高级（比如从像素点到边缘，再到物体的部件，最后到整个物体）

### NLP中的词嵌入技术：核心

这部分和 agent 里的 **RAG**深度相关，如果要构建一个高质量的RAG系统，就要熟练掌握 Embedding技术的实现

**必须**掌握以下几点：

- **核心思想**

  ​	深刻理解**“词嵌入就是把词语/句子映射到一个高维的数学空间，在这个空间里，意思相近的词语/句子，它们的向量在空间中的距离也相近”**

  ​	这是构建所有RAG应用和语义搜索的基石。

- **工作原理**

  ​	知道Embeddings是如何**生成**的（通过调用OpenAI text-embedding-ada-002 或其他模型的API），以及它们是如何**使用**的（通过计算**向量间的相似度**，如余弦相似度，来找到最相关的文本片段）

- **实践应用**

  ​	能够熟练地将一段文本转换成向量，将这些向量存入**向量数据库**，并根据一个新的查询向量，从数据库中高效地检索出最相似的几个向量。

## 知识清单

### 第一部分：掌握大脑核心：大语言模型(LLM)的API和工程化

1. **LLM API熟练使用**:

   - OpenAI API (GPT-4, GPT-4o)

     这是行业标准，必须掌握。学习如何进行API调用、处理响应、管理API密钥

   - Anthropic Claude API / Google Gemini API

     了解其他主流模型的API，知道它们各自的特点（例如Claude的长上下文窗口）

   - 开源模型

     学习如何通过Hugging Face、Ollama或Replicate等平台，在本地或云端调用开源LLM（如Llama 3）

2. **提示工程**

   - 核心技能: 

     这是与LLM沟通的语言。你不是在训练模型，而是在指导模型。

   - 基础: 

     ​角色扮演、零样本 (Zero-shot)、少样本 (Few-shot) 提示

   - 进阶: 

     ​思维链 (Chain-of-Thought, CoT) 引导模型进行复杂推理、生成结构化输出 (JSON)

3. **核心概念理解**:

   - **Tokens**: 理解文本是如何被量化和计费的

   - **Context Window**: 知道每个模型的“短期记忆”有多长，以及超出后会发生什么

   - **Embeddings**: 理解它是一种将文本转换为“意义向量”的技术，是实现记忆和知识库搜索的基础

### 第二部分：赋予大脑记忆和知识：RAG与向量数据库

1. **RAG (Retrieval-Augmented Generation)**:

   - **核心架构**: 

     ​这是目前最重要、最流行的LLM应用架构。它解决了LLM不知道私有数据/近期数据的问题

   - **流程**: 

     ​必须能清晰地解释并实现 Query -> Embed -> Search -> Retrieve -> Augment -> Generate 的完整流程

2. **向量数据库**:

   - **作用**: 存储和高效检索Embeddings。这是RAG的长期记忆模块

   - **技术选型**:

     - 本地/轻量级: 

       ChromaDB, FAISS (一个库，不是数据库)。适合快速原型开发

     - 云服务/生产级: 

       Pinecone, Weaviate了解它们的API和使用场景

### 第三部分：构建智能体的“身体”和“行动能力”：框架与工具

1.  **AI Agent框架**

   - 作用: 

     ​这些框架将上述所有组件（LLM,Prompt, Memory, Tools）粘合在一起，极大简化了Agent的开发

   - LangChain: 

     ​**必学**。最流行、最全面的框架。学习它的核心概念：Chains (执行链), Agents (智能体循环), Tools (工具)

   - LlamaIndex:     专注于RAG，是构建知识库应用的强大工具

   - AutoGen / CrewAI:     专注于多智能体协作的框架

2. **工具使用  / 函数调用**

   - Agent的核心:     

     让LLM从一个聊天机器人变成一个行动者的关键

   - 原理: 

     让LLM能够决定调用哪个外部API或代码函数来获取信息或执行操作

   - 实践: 

     学习OpenAI的Function Calling API，并掌握如何在LangChain中为Agent定义和使用工具（例如，一个能查询天气的API、一个计算器、一个能访问公司数据库的工具）

### 第四部分：将应用落地——部署与工程实践

1. **Web框架**:
   - 作用: 为AI Agent提供一个用户界面或API接口
   - FastAPI: Python的首选。非常适合为Agent构建高性能的API后端
   - Streamlit / Gradio: 用于快速构建漂亮的前端Demo和原型，非常适合展示项目
2. **部署与运维**:
   - 容器化: Docker基础。学习如何将你的AI应用打包成一个Docker镜像
   - 云平台: 了解如何在Hugging Face Spaces、Streamlit Community Cloud或主流云平台（AWS, GCP, Azure）上部署你的应用

## 合适的学习路线

### 第一阶段：成为一个LLM API调用大师

1. **学习LLM API调用**:

   - 技术: 

     学习如何使用Python的requests库或官方的openai库，来调用一个LLM的API（推荐从OpenAI的GPT系列开始，因为它的文档和社区最好）

   - 需要理解: 什么是API Key，如何构建请求，如何解析返回的JSON数据

2. **学习提示工程**:

   - 技术: 这不是写代码，而是学习如何用自然语言精确地指导LLM
   - 需要掌握:
     - **角色设定**: 如何通过系统提示让LLM扮演一个特定角色（例如“你是一个资深的语音助手”）
     - **思维链 (Chain-of-Thought)**: 如何通过一步一步地思考这样的指令，引导LLM解决复杂问题
     - **输出格式化**: 如何要求LLM返回特定格式的输出，比如JSON，这对于你用FastAPI进行后端处理至关重要

- **项目1：构建一个多功能角色聊天机器人**
  
  - 目标: 熟练使用OpenAI API和Prompt Engineering
  - 任务: 创建一个简单的Web界面（用Streamlit），用户可以选择不同的角色（如“Python编程助手”、“英语口语教练”、“苏格拉底”），程序会根据选择，使用不同的System Prompt与用户进行高质量对话
  - 收获: 精通API调用、掌握核心的提示工程技巧
- **项目2：构建一个高级API封装器**

  - 任务: 使用FastAPI，创建一个简单的后端服务。它接收一个任务描述（比如“总结这段文字”或“把这段英文翻译成中文”），然后在内部构建一个高质量的Prompt，调用LLM API，最后将LLM返回的干净结果作为API的响应返回
  - 目的: 将LLM的强大能力，封装成你可以轻松调用的、可靠的后端服务

### 第二阶段：构建一个有记忆的专家

1. **学习词嵌入的概念与应用**:

   - 需要理解: 

     ​Embeddings 是一种将文本（对话历史）转换成数字向量的技术。**在向量空间中，意思相近的文本，它们的向量也相近**。 这就是实现智能记忆搜索的魔法

   - 实践: 

     ​学会如何调用API（如OpenAI的Embedding API）或使用本地库（如sentence-transformers）来将任何一句话变成一个向量

2. **学习本地向量存储与搜索**:

   - 技术: 学习一个轻量级的本地向量索引库，FAISS 或 ChromaDB 是最佳选择

   - 需要理解: 

     ​它们就像一个专门为向量设计的数据库。你可以把成千上万个对话向量存进去，然后用一个新的查询向量，瞬间找出与它最相似的几个历史对话向量

3. **学习RAG架构**:

   - 技术: 这是将记忆和大脑结合起来的黄金准则
   - 流程: `用户提问 -> 向量化 -> 在向量索引中搜索相关历史 -> 将历史和问题一起打包给LLM -> LLM给出带上下文的回答`

- **项目3：与JSON对话**

  - **任务**: 编写一个Python脚本，它可以：
    1. 读取一个包含多段对话的 JSON 文件
    2. 使用 Embeddings 将每一段对话向量化，并存入一个本地FAISS索引文件
    3. 接收一个新的用户输入，向量化后在FAISS中进行搜索，找出最相关的历史对话
    4. 打印出这些相关的历史对话

- **项目4：与PDF对话——构建一个RAG应用**

  - 目标: 掌握RAG架构、Embeddings和向量数据库。
  - 任务: 创建一个应用，用户可以上传一份PDF文档（比如一份年报或一篇论文），应用会对其进行切割、向量化并存入 ChromaDB。之后，用户可以就这份文档的内容进行提问，系统会从文档中找到相关信息来回答
  - 收获: 真正理解并实现了目前最主流的LLM应用架构

### 第三阶段：迈向真正的智能体

1. **学习一个Agent框架 (LangChain)**:

   - 技术: 

     ​LangChain是一个“粘合剂”，它把LLM、记忆系统、工具（未来的扩展）等所有组件都封装好了，让你不必编写大量重复的底层代码

   - 需要学习: 

     ​LangChain中的Memory模块和Chains的概念。它能帮你自动化上面提到的RAG流程

2. **整合现有系统**:

   - 技术:     将现有的 `ASR -> vLLM -> TTS` 流程进行改造
   - 新的流程:
     `ASR => FastAPI后端 => LangChain Agent (内部执行RAG记忆检索) => vLLM => FastAPI返回结果 => TTS`

- **项目5：构建一个简单的生活助理 Agent**

  - 目标: 掌握LangChain框架和Tool Use
  - 任务: 使用LangChain构建一个Agent，并为它提供至少两个工具：
    1. 一个能调用天气API的工具
    2. 一个能执行简单数学计算的工具
  - 交互示例:
    - 你问：“北京今天天气怎么样？” -> Agent判断需要使用天气工具 -> 调用API -> 返回结果
    - 你问：“(35+19)*2等于多少？” -> Agent判断需要使用计算工具 -> 执行计算 -> 返回结果
  - 收获: 理解了Agent的“思考-行动”循环 (ReAct)，让你的AI第一次能够与外部世界交互

### 第四阶段：成为一个全栈AI工程师

- **项目6：将生活助理 Agent 产品化**
  - 目标: 掌握应用的封装和部署
  - 任务:
    1. 使用FastAPI将你的Agent封装成一个API端点
    2. 创建一个简单的React或Vue前端（或者继续用Streamlit）来与这个API交互
    3. 将整个应用Docker化，并将其部署到Hugging Face Spaces或云服务器上，让你的朋友可以通过公网访问
  - 收获: 走完了从想法到原型再到上线部署的全流程，你已经具备了成为一名合格的AI应用工程师的核心能力

## 学习深度是 T型

- **横向 (广度)**: 

  ​	对机器学习、神经网络等领域有宏观的、概念性的理解。知道它们是做什么的，有什么优缺点，能解决什么问题。

- **纵向 (深度)**: 

  ​	在与AI Agent 应用开发直接相关的技术上，进行深入的、实践性的学习

  这包括：

  - LLM API & Prompt Engineering
  - RAG 架构
  - Embeddings 的原理与应用
  - 向量数据库
  - LangChain等Agent框架
  - Function Calling / Tool Use



1. 快速回顾，不要深陷: 

   ​花少量时间快速回顾一下机器学习和神经网络的核心概念，确保你对“过拟合”、“评估指标”等有清晰的认识。可以使用一些高质量的视频（如StatQuest, 3Blue1Brown）来建立直观理解，然后就立即停下

2. 把主要精力投入到Embeddings: 

   ​这是连接“理论”和“Agent应用”最重要的一座桥梁。找一篇专门讲解和实践Embeddings与向量搜索的文章或教程，动手实现一个简单的语义搜索系统

3. 然后，立即转向Agent开发

   ​按照项目驱动路线图，开始构建第一个应用。在实践中遇到问题时，再回过头来查阅相关的理论知识，这样的学习效率是最高的