---
title: 'torch.save模块'
author: Alen
published: 2025-10-28
description: "PyTorch模型持久化工具 torch.save的介绍"
first_level_category: "AI"
second_level_category: "PyTorch"
tags: ['ML','DL']
draft: false
---


# torch.save持久化模型

## 引言：模型持久化的重要性

模型持久化是指将训练好的模型的状态保存到磁盘，以便将来可以重新加载和使用

这至关重要，原因如下：

1. **避免重复训练**：  训练模型（尤其是大型模型）可能需要数小时甚至数天，持久化可随时保存成果。
2. **模型部署 (Inference)**：  在生产环境中，需要加载训练好的模型来对新数据进行预测
3. **断点续练 (Checkpointing)**：  如果训练过程意外中断，可以从上次保存的状态（称为  “检查点”  ）继续训练，而不是从头开始
4. **模型分享与复现**：  可以将模型文件分享给他人，方便复现研究结果

PyTorch 提供了两种核心的持久化方法，它们都使用 `torch.save()` 和 `torch.load()`，但保存的对象不同

------

## 方法一：保存模型的状态字典 (tate Dictionary

这是最常用、最灵活的方法

**核心思想**：只保存模型的可学习参数（权重 W 和偏置 b），而不保存模型的结构

- **状态字典 state_dict**：  
   一个 Python 字典，将模型每一层映射到其对应的参数张量
   例如，键可能是 'ln1.weight'，值就是第一层线性层的权重张量

**优点**：

- **灵活性和可移植性**：  这是最大的优点。只要有模型的 Python 类定义（`class MLP(...)`），就可以在任何项目、任何文件结构下重新创建模型实例并加载这些参数
- **轻量级**：  只保存了必要的参数数据，文件体积更小
- **安全性**：  加载的文件不包含可执行代码，更安全

### 封装方法

1. **保存：torch.save(obj, f)**
   - **作用**：  将一个 Python 对象序列化并保存到磁盘
   - **参数**：
     - obj:   要保存的对象；在这里传入 model.state_dict()
     - f: 一个字符串，表示文件路径（例如 `'my_model.pth'`），或者一个文件类对象
2. **加载：model.load_state_dict(state_dict)**
   - **作用**：  将一个状态字典中的参数加载到当前模型实例中；PyTorch 会按键名匹配参数，并将 state_dict 中的值复制到模型的参数张量中
   - **参数**：
     - state_dict:   一个包含模型参数的 Python 字典，通常由 torch.load(f) 从文件中读取得到

### 示例代码

假设我们已经有了 MLP 类的定义和训练好的模型 trained_model

<a href="" target="_blank">MLP实例实践请点击此处</a>

**1. 保存模型参数**

```python
# 假设 trained_model 是已经训练好的模型实例
# 例如: trained_model, _ = train_model(...)

# 定义模型保存路径
model_save_path = "fashion_mnist_mlp_statedict.pth"

# 只保存模型的状态字典
torch.save(trained_model.state_dict(), model_save_path)
print("Model state dictionary saved successfully in {model_save_path}")
```

**2. 加载模型参数并进行推理**

```python
# --- 在一个新的脚本或环境中 ---

# 1. 必须先重新创建模型实例，并且结构必须和保存时完全一样
loaded_model = MLP()
print("Created a new instance of the MLP model.")

# 2. 加载状态字典
print(f"Loading model state dictionary from {model_save_path}...")
state_dict = torch.load(model_save_path)

# 3. 将状态字典加载到模型实例中
loaded_model.load_state_dict(state_dict)
print("Model state dictionary loaded successfully.")

# 4. 将模型设置为评估模式
loaded_model.eval()

# 现在 loaded_model 和 trained_model 的参数完全一样，可以用于预测
# with torch.no_grad():
#     # ... 进行预测 ...
```

------

## 方法二：保存整个模型

这种方法将整个模型对象（包括结构和参数）一起序列化保存

**核心思想**：  使用 Python 的 pickle 模块将整个 nn.Module 对象保存下来

**优点**：

- **简单直接**：  保存和加载的代码都只有一行，非常方便

**缺点**：

- **脆弱**：  序列化的数据与特定的类和文件目录结构绑定。如果你重构了项目，移动了模型类的定义文件，或者修改了类名，加载时就可能会失败。
- **可移植性差**：  在其他项目中复用模型会很困难
- **安全风险**：  加载 pickle 文件可能会执行任意代码，存在安全隐患

### 封装的方法

1. **保存：torch.save(obj, f)**
   - **作用**：  同方法一
   - **参数**：
     - obj: 在这里直接传入**整个模型对象** model
2. **加载：torch.load(f)**
   - **作用**：  从文件中反序列化对象
   - **返回值**：  直接返回保存时的那个对象，在这里就是**加载好的模型实例**

### 示例代码

**1. 保存整个模型**

```python
# 假设 trained_model 是已经训练好的模型实例
model_save_path = "fashion_mnist_mlp_whole.pth"

# 直接保存整个模型对象
torch.save(trained_model, model_save_path)
print("Entire model saved successfully in {model_save_path}.")
```

**2. 加载整个模型并进行推理**

```python
# --- 在一个新的脚本或环境中 ---

# 1. 直接加载模型，无需预先创建实例
print(f"Loading the entire model from {model_save_path}...")
loaded_whole_model = torch.load(model_save_path)
print("Entire model loaded successfully.")

# 2. 将模型设置为评估模式
loaded_whole_model.eval()

# 现在 loaded_whole_model 就可以直接使用了
# with torch.no_grad():
#     # ... 进行预测 ...
```

------

## 补充：保存检查点 (Checkpoint) 以恢复训练

这是一个更高级、更实用的应用，它结合了方法一的优点，检查点不仅保存模型参数，还保存了恢复训练所需的一切信息。

**核心思想**：  创建一个字典，包含所有需要恢复训练状态的信息，然后保存这个字典

**需要保存的内容**：

- **当前 epoch**
- **模型的状态字典 (`model.state_dict()`)**
- **优化器的状态字典 (`optimizer.state_dict()`)**：  这很重要，因为它包含了优化器的状态，如动量（Momentum）的缓存、Adam 的一阶和二阶矩估计等
- **当前的损失值（可选）**

### 示例代码

**1. 在训练循环中保存检查点**

```python
# 在 train_model 函数或训练循环的末尾
# 假设 optimizer 是优化器实例
check_point = "training_checkpoint.pth"

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': last_loss, # 假设记录最后一个批次的损失
}, check_point)
  
```

**2. 加载检查点以恢复训练**

```python
# 在开始训练之前

# 1. 像往常一样创建模型和优化器实例
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. 加载检查点
checkpoint = torch.load(check_point)

# 3. 将保存的状态加载到模型和优化器中
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 4. 恢复训练的 epoch 和 loss
start_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']

print(f"Resuming training from epoch {start_epoch+1}")

# 5. 将模型设置为训练模式
model.train()

# 6. 现在可以从 start_epoch + 1 开始继续训练循环
# for epoch in range(start_epoch + 1, num_epochs):
#     # ... a normal training loop ...
```

---

## 总结对比

|       特性       |                方法一：保存 state_dict (推荐)                |                    方法二：保存整个模型                    |
| :--------------: | :----------------------------------------------------------: | :--------------------------------------------------------: |
|   **保存内容**   |               仅模型的可学习参数 (权重和偏置)                |          整个模型对象 (结构 + 参数)，使用 pickle           |
|     **优点**     |               **灵活、可移植、安全、标准做法**               |                      简单，代码量最少                      |
|     **缺点**     |                需要先手动创建模型实例才能加载                |               脆弱，依赖文件结构，有安全风险               |
| **torch.save()** |             torch.save(model.state_dict(), PATH)             |                  torch.save(model, PATH)                   |
|   **加载方式**   |   model = MLP()<br>model.load_state_dict(torch.load(PATH))   |                  model = torch.load(PATH)                  |
|   **适用场景**   |          **所有场景**，特别是部署、分享和长期保存。          |      快速、临时的模型保存，且不打算在其他项目中使用。      |
|    **检查点**    | 是保存检查点的**基础**，通过将多个 state_dict 存入一个字典实现 | 不适用于保存检查点，因为它没有方便地分离出优化器等其他状态 |