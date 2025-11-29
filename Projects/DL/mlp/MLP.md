---
title: 'MLP实践'
author: Alen
published: 2025-10-28
description: "PyTorc第一个实践：MLP的简单实验"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['ML','DL']
draft: false
---

# MLP 简单实践

这是一个简单的MLP实践，或者说，一个完整的深度学习实例。

## 深度学习实践流程

完整的深度学习实践应当包含以下步骤：

1. **数据处理**

   1. **加载数据集**
   2. **切分数据集：训练集、验证集、测试集**
   3. **创建 DataLoader准备给模型喂数据**
   4. **必要时自定义 Dataset**

2. **选择 GPU还是 CPU进行训练**

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

   **如果选择 GPU训练，记得将以下内容迁移到 GPU上**：

   1. **模型训练前将实例化的模型转移到 gpu上**
   2. **每一轮训练中，将训练的数据 X,y转移到 gpu上**

   分别对应以下代码：

   1. ```python
      model.to(device) # model为实例化模型：model = MLP()
      ```

   2. ```python
      for X, y in train_loader:
          X, y = X.to(device), y.to(device)
      ```

3. **构建神经网络**

   1. **创建一模型子类，集成 nn.Module**
   2. **在初始化函数`__init__()`中使用  nn.Sequential()构建网络框架，定义数据流(可选)**
   3. **在 forward()中定义模型的数据流（和第二步的操作二选一），并返回模型的输出**

4. **训练模型**

   **实例化一个优化器**

   **在训练循环`for epoch in range(num_epochs):`中**

   1. **模型训练环节**
      1. **将模型切换到训练模式**：`model.train()`
      2. **将训练数据 X,y转移到正确的设备上**
      3. **前向传播**
      4. **计算Loss**
      5. **梯度清零并进行反向传播**
      6. **更新网络参数**
   2. **模型验证环节**
      1. **将模型切换为测试模式**：`model.eval()`
      2. **不在使用梯度下降法更新参数**
      3. **将验证数据 X,y 转移到正确的设备上**
      4. **计算损失值**

5. **测试模型性能**

   1. **将模型转为测试模式**
   2. **计算Loss与 精确率等评价指标**

6. **存储训练过程得到的loss和precision，便于可视化**

   1. **使用一字典存储需要的数据**
   2. **转为 json保存**

7. **模型持久化**

   1. **使用 torch.save存储状态字典**

---

## 实践

实践目的是使用 *Fashion_MNIST*数据集实现一个多类别图像分类器，判断输入的图像类别：

- **输入**：一张 28x28 像素的灰度图像
- **处理**：通过 MLP 模型进行学习和转换
- **输出**：从10个预设的类别（如 'T-shirt/top', 'Trouser', 'Pullover' 等）中预测出图像最属于哪一类

### 1. 获取数据集

`torchvision`库里的数据集已自带`Fashion_MNIST`数据集，从此加载即可，此处重点应该是：
1. 怎么从一个 Dataset获取数据
2. **怎么使用 DataLoader加载训练模型需要的数据**

```python
def get_dataloader_workers():
    """使用 4 个进程读取数据"""
    # 注意：在Windows上，这必须在 if __name__ == '__main__': 块中运行
    return 4

def load_fashion_mnist_datasets(resize=None):
    """
    下载Fashion-MNIST数据集，返回 PyTorch Dataset 对象
    返回 Dataset 对象而不是 DataLoader，以便我们能进行切分练习
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    full_train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    return full_train_dataset, test_dataset
```

返回的数据类型是：`<class 'torchvision.datasets.mnist.FashionMNIST'>`

### 2. 数据预处理

拿到数据后，对其进行划分

#### 2.1. 划分数据集

划分验证集 validation_dataset、训练集 train_dataset、测试集 test_dataset

[为什么需要验证集请查看](https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html#id6)

```python
   full_train_dataset, test_dataset = load_fashion_mnist_datasets()

   # 计算切分大小
   train_samples_chunk = len(full_train_dataset)
   val_samples_chunk = int(validation_split * train_samples_chunk)
   train_samples_chunk -= val_samples_chunk

   # 使用 random_split 切分数据集
   train_dataset, val_dataset = random_split(
       full_train_dataset, [train_samples_chunk, val_samples_chunk]
   )
```

#### 2.2. 创建 DataLoader

DataLoader相关的参数介绍<a href="https://juayohuang.top/posts/ai/pytorch/torchutilsdata" target="_blank" >请查看此处</a>
```python
    # 为切分后的数据集创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(),pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=get_dataloader_workers(),pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=get_dataloader_workers(),pin_memory=True)
```

### 3. 选择设备进行训练

**为加快训练，选择 GPU进行处理**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 4. 构建神经网络

神经网络中数据流的构建主要有两种等效的方法：
1. 在初始化时就定义网络中数据的流动方式：使用 `nn.Sequential`

```python
class MLP(nn.Module):
   def __init__(self,input_size=784,hidden_size=256,num_classes=10):
      super(MLP,self).__init__()
      self.faltten = nn.Flatten()
      self.network = nn.Sequential(
         nn.Linear(input_size, hidden_size),
         nn.ReLU(),
         nn.Linear(hidden_size, hidden_size // 2),
         nn.ReLU(),
         nn.Linear(hidden_size // 2, num_classes)
         # 输出层不加 Softmax，因为 nn.CrossEntropyLoss 会处理
      )
   
   def forward(x):
      x = self.faltten(x)
      return self.network(x) # 把数据 x传入网络即可

```
2. 初始化时只定义要用到的网络组件，在设计前向传播计算图时再定义数据流

```python
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.ln1 = nn.Linear(input_size,hidden_size)
        self.ln2 = nn.Linear(hidden_size,hidden_size//2)
        self.ln3 = nn.Linear(hidden_size//2,num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.relu(x)
        return self.ln3(x)
```

#### 4.1. 网络结构设计

网络结构包含：
- 一个平坦层 Flatten：  用于将输入的28x28的图像展平为784维的向量
- 两个隐藏层 Linear + ReLU：  实现非线性激活
- 一个输出层 Linear：  输出层有10个神经元，对应Fashion-MNIST的10个类别，CrossEntropyLoss()内部已有了 Softmax函数，因此在使用 CrossEntropyLoss计算 Loss时，不需要手动添加 Softmax()。

网络结构每一层的神经元依次为：
$$
(28*28)\text{Flatten\_layer}(784)\ \Rightarrow \ (784)\text{fst\_ hidden\_ layer}(256) \Rightarrow \ (256)\text{sec\_ hidden\_ layer}(128) \Rightarrow \ (128)\text{output\_layer}(10)
$$

```python
class MLP(nn.Module):
   def __init__(self,input_size=784,hidden_size=256,num_classes=10):
      super(MLP,self).__init__()
      self.faltten = nn.Flatten()
      self.network = nn.Sequential(
         nn.Linear(input_size, hidden_size),
         nn.ReLU(),
         nn.Linear(hidden_size, hidden_size // 2),
         nn.ReLU(),
         nn.Linear(hidden_size // 2, num_classes)
      )
```

#### 4.2. 前向传播计算图设计

```python
    def forward(self, x):
        x = self.flatten(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.relu(x)
        return self.ln3(x)
```

**注意**：这是为了方便看到数据流动的过程，实际上可以使用前文的方式，平坦化后直接传入定好的 `network`中即可

### 5. 训练模型

需要的参数（包括超参数）：
1. 模型
2. 训练集的 DataLoader（训练集）
3. 验证集的 DataLoader（验证集）
4. 优化器
5. batch_size ---- 用于划分数据集，实例化 DataLoader，训练模型时用不到
6. 学习率 learning rate
7. 训练轮数 num_epoches
8. 损失函数
9. 运行设备 device
   

这里的参数取值（不包括 DataLoader）为：
1. model = MLP()
2. batch_size = 256
3. lr = 0.001
4. num_epoches = 60
5. loss_function = nn.CrossEntropyLoss()
6. device = torch.device('cuda')

#### 5.1. 实例化模型和损失函数

将定义好的网络模型 MLP()进行实例化，并且选择合适的优化器

```python
model = MLP()

loss_function = nn.CrossEntropyLoss()
```

将模型转移到 gpu上

```python
model.to(device)
```

#### 5.2 实例化优化器，以 Adam为例

```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

#### 5.3 训练循环里

在训练循环`for epoch in range(num_epochs):`里，分成两个阶段：
1. 训练阶段：  训练模型

      1. 将模型切换到训练模式
      2. 将训练数据 X,y转移到正确的设备上
      3. 前向传播
      4. 计算Loss
      5. 梯度清零并进行反向传播
      6. 更新网络参数
2. 验证阶段
      1. 将模型切换为测试模式
      2. 不再使用梯度下降法更新参数
      3. 将验证数据 X,y 转移到正确的设备上
      4. 计算损失值

**5.3.1 模型训练阶段**

```python
        # 训练阶段
        # 1.将模型切换到训练模式
        model.train()
        # 用于计算每一轮的平均损失和精确率
        train_loss, accuracy_count = 0, 0
        for X, y in train_loader:
            # 2.将训练数据 X,y转移到正确的设备上
            X, y = X.to(device), y.to(device)

            # 3.前向传播
            outputs = model(X)
            # 4.计算Loss
            loss = loss_fn(outputs, y)

            # 5.梯度清零
            optimizer.zero_grad()
            # 6.误差反向传播
            loss.backward()
            # 7.更新权重
            optimizer.step()

            train_loss += loss.item()
            accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

        average_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_count / len(train_loader.dataset)
```

**5.3.2 模型验证阶段**

```python
        # 验证阶段：不需要再进行反向传播
        # 1. 将模型切换为测试模式
        model.eval()
        # 同训练阶段
        validation_loss, validation_accuracy_count = 0, 0
        # 不需要再进行反向传播，因此没必要继续记录梯度
        with torch.no_grad():
            for X, y in val_loader:
                # 2.将验证数据 X,y 转移到正确的设备上
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                # 3.计算损失值
                validation_loss += loss_fn(outputs, y).item()
                validation_accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

        avg_validation_loss = validation_loss / len(val_loader)
        validation_accuracy = validation_accuracy_count / len(val_loader.dataset)
```

**完整函数代码**

```python
def train_model(model, train_loader, val_loader, optimizer_name, loss_fn, num_epochs, learning_rate, device):
    """
    集成的训练函数，可以根据 optimizer_name 选择不同的优化器
    """
    print(f"\n--- Starting Training ---")
    print(f"Optimizer: {optimizer_name}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Device: {device}")

    # 将模型移动到指定设备
    model.to(device)

    # 根据参数选择优化器
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 用于记录训练历史
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练阶段
        # 1.将模型切换到训练模式
        model.train()
        # 用于计算每一轮的平均损失和精确率
        train_loss, accuracy_count = 0, 0
        for X, y in train_loader:
            # 2.将训练数据 X,y转移到正确的设备上
            X, y = X.to(device), y.to(device)

            # 3.前向传播
            outputs = model(X)
            # 4.计算Loss
            loss = loss_fn(outputs, y)

            # 5.梯度清零
            optimizer.zero_grad()
            # 6.误差反向传播
            loss.backward()
            # 7.更新权重
            optimizer.step()

            train_loss += loss.item()
            accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

        average_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_count / len(train_loader.dataset)

        # 验证阶段：不需要再进行反向传播
        # 1. 将模型切换为测试模式
        model.eval()
        # 同训练阶段
        validation_loss, validation_accuracy_count = 0, 0
        # 不需要再进行反向传播，因此没必要继续记录梯度
        with torch.no_grad():
            for X, y in val_loader:
                # 2.将验证数据 X,y 转移到正确的设备上
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                # 3.计算损失值
                validation_loss += loss_fn(outputs, y).item()
                validation_accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

        avg_validation_loss = validation_loss / len(val_loader)
        validation_accuracy = validation_accuracy_count / len(val_loader.dataset)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
              f"Val Loss: {avg_validation_loss:.4f}, Val Acc: {validation_accuracy:.4f} | "
              f"Duration: {epoch_duration:.2f}s")

        history['train_loss'].append(average_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_validation_loss)
        history['val_accuracy'].append(validation_accuracy)

    print("--- Finished Training ---")
    return model, history
```

### 6. 测试模型性能

获取到训练好的模型后，根据评价指标对其进行评估。

```python
def evaluate_model(model, test_loader, device):
    """在测试集上评测最终模型性能"""
    #  1. 将模型转为测试模式
    model.to(device)
    model.eval()
    test_accuracy_count = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            # 2. 计算Loss
            outputs = model(X)
            test_accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()
    # 3. 计算精确率
    test_acc = test_accuracy_count / len(test_loader.dataset)
    return test_acc
```

### 7. 存储训练过程得到的Loss

```python
    import json
    file_path = 'train_history.json'
    with open(file_path, 'w') as f:
        json.dump(history,f,indent=4,ensure_ascii=False)
```

### 8.模型持久化

模型持久化有两种，这里只保存模型状态字典，而不是整个模型。

<a href="" target="_blank">具体可查看此篇文章</a>

```python
model_save_path = 'fashion_mnist_model.pth'
torch.save(trained_model.state_dict(), model_save_path)
```

---

## 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import time


# --- 1. 数据处理 Dataset ---
def get_dataloader_workers():
    """使用 4 个进程读取数据"""
    # 注意：在Windows上，这必须在 if __name__ == '__main__': 块中运行
    return 4


def load_fashion_mnist_datasets(resize=None):
    """
    下载Fashion-MNIST数据集，返回 PyTorch Dataset 对象
    返回 Dataset 对象而不是 DataLoader，以便我们能进行切分练习
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    full_train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    return full_train_dataset, test_dataset


# --- 2. 神经网络构建 nn.Module ---
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # 与forward()的数据流二选一
        # self.network = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, num_classes)
        #     # 输出层不加 Softmax，因为 nn.CrossEntropyLoss 会处理
        # )
        self.ln1 = nn.Linear(input_size,hidden_size)
        self.ln2 = nn.Linear(hidden_size,hidden_size//2)
        self.ln3 = nn.Linear(hidden_size//2,num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.relu(x)
        return self.ln3(x)

# --- 3. 训练与评测函数 ---
def train_model(model, train_loader, val_loader, optimizer_name, loss_fn, num_epochs, learning_rate, device):
    """
    集成的训练函数，可以根据 optimizer_name 选择不同的优化器
    """
    print(f"\n--- Starting Training ---")
    print(f"Optimizer: {optimizer_name}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Device: {device}")

    # 将模型移动到指定设备
    model.to(device)

    # 根据参数选择优化器
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 用于记录训练历史
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练阶段
        # 1.将模型切换到训练模式
        model.train()
        # 用于计算每一轮的平均损失和精确率
        train_loss, accuracy_count = 0, 0
        for X, y in train_loader:
            # 2.将训练数据 X,y转移到正确的设备上
            X, y = X.to(device), y.to(device)

            # 3.前向传播
            outputs = model(X)
            # 4.计算Loss
            loss = loss_fn(outputs, y)

            # 5.梯度清零
            optimizer.zero_grad()
            # 6.误差反向传播
            loss.backward()
            # 7.更新权重
            optimizer.step()

            train_loss += loss.item()
            accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

        average_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_count / len(train_loader.dataset)

        # 验证阶段：不需要再进行反向传播
        # 1. 将模型切换为测试模式
        model.eval()
        # 同训练阶段
        validation_loss, validation_accuracy_count = 0, 0
        # 不需要再进行反向传播，因此没必要继续记录梯度
        with torch.no_grad():
            for X, y in val_loader:
                # 2.将验证数据 X,y 转移到正确的设备上
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                # 3.计算损失值
                validation_loss += loss_fn(outputs, y).item()
                validation_accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

        avg_validation_loss = validation_loss / len(val_loader)
        validation_accuracy = validation_accuracy_count / len(val_loader.dataset)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
              f"Val Loss: {avg_validation_loss:.4f}, Val Acc: {validation_accuracy:.4f} | "
              f"Duration: {epoch_duration:.2f}s")

        history['train_loss'].append(average_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_validation_loss)
        history['val_accuracy'].append(validation_accuracy)

    print("--- Finished Training ---")
    return model, history


def evaluate_model(model, test_loader, device):
    """在测试集上评测最终模型性能"""
    model.to(device)
    model.eval()
    test_accuracy_count = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            test_accuracy_count += (outputs.argmax(1) == y).type(torch.float).sum().item()

    test_acc = test_accuracy_count / len(test_loader.dataset)
    return test_acc


# 主执行流程 - 必须放在这个保护块下以支持多进程数据加载
if __name__ == '__main__':
    # 超参数
    batch_size = 256
    validation_split = 0.15  # 从训练集中划分 15% 作为验证集
    lr = 0.001
    epoches = 60
    # 优化器切换
    OPTIMIZER_NAME = 'Adam'  # 可选: 'Adam', 'RMSProp', 'AdaGrad', 'SGD'

    # 1. ---- 数据处理 & 数据集切分 ----
    print("--- 1. Loading and Preparing Data ---")
    full_train_dataset, test_dataset = load_fashion_mnist_datasets()

    # 计算切分大小
    train_samples_chunk = len(full_train_dataset)
    val_samples_chunk = int(validation_split * train_samples_chunk)
    train_samples_chunk -= val_samples_chunk

    # 使用 random_split 切分数据集
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_samples_chunk, val_samples_chunk]
    )

    print(f"Total training samples: {len(full_train_dataset)}")
    print(f"Split into: {len(train_dataset)} for training, {len(val_dataset)} for validation")
    print(f"Total test samples: {len(test_dataset)}")

    # 为切分后的数据集创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(),pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=get_dataloader_workers(),pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=get_dataloader_workers(),pin_memory=True)

    # ---- 2. 选择GPU进行训练 ----
    print("\n--- 2. Setting up Device ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 3. 神经网络构建 ----
    print("\n--- 3. Building Model ---")
    model = MLP()

    # ---- 4. 训练神经网络模型 ----
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 调用集成的训练函数
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=OPTIMIZER_NAME,
        loss_fn=loss_function,
        num_epochs=epoches,
        learning_rate=lr,
        device=device
    )

    # ---- 5. 进行评测 ----
    print("\n--- 5. Evaluating on Test Set ---")
    test_accuracy = evaluate_model(trained_model, test_loader, device)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # ---- 6.存储训练数据 ----
    import json
    file_path = 'train_history.json'
    with open(file_path, 'w') as f:
        json.dump(history,f,indent=4,ensure_ascii=False)

    # ---- 7.模型持久化 ----
    model_save_path = 'fashion_mnist_model.pth'
    torch.save(trained_model.state_dict(), model_save_path)
```



---

## 手动实现MLP

若想深入底层实现，就要知道输入层、隐藏层、输出层、激活函数、损失函数怎么实现的。

这些实际上就是 PyTorch 封装好的每一个组件，像本次实验用到的 Linear全连接层、Flatten平坦层、ReLU和交叉熵损失函数

而接下来分几个部分介绍每一层的实现

### 1.全连接层的手动创建

```python
num_inputs,num_outputs,num_hiddens = 784,10,256
# 隐藏层的权重 (W1) 和偏置 (b1)
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

# 输出层的权重 (W2) 和偏置 (b2)
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
```

代码没有使用 nn.Linear 层：该层会自动创建和管理权重 W 和偏置 b。相反，它手动创建了每一层所需的参数，并将它们封装在 nn.Parameter 中

- **nn.Parameter 的作用**：
   nn.Parameter 是 torch.Tensor 的一个特殊子类；当一个 Tensor被 nn.Parameter 包装后，它就**自动被注册为模型的可学习参数**
   
   这意味着调用 loss.backward() 时，PyTorch 会自动计算这些参数的梯度，并且优化器 (torch.optim.SGD) 能够找到并更新它们

- **参数初始化**：权重被初始化为小的随机数 (randn * 0.01)，偏置被初始化为零。这是一种常见的初始化策略

### 2.前向传播设计

`net(X)`函数设计的是数据流动的过程：

```python
    def net(X):
        X = X.reshape((-1,num_inputs)) # 平坦化：将空间结构展平成向量
        H = relu(X@W1 + b1) # 隐藏层 => 激活函数
        return (H @ W2 + b2)
```

1. 将数据展平

   ```python
   X = X.reshape((-1, num_inputs))
   ```

   - Fashion-MNIST 数据集中的每个样本 X 是一个 (batch_size, 1, 28, 28) 的4D张量

   - MLP 处理的是向量数据，而不是具有空间结构的图像数据

     因此，第一步必须将每个 (1, 28, 28) 的图像展平为一个 784 维的向量 (num_inputs = 28 * 28 = 784)

   - reshape((-1, num_inputs)) 中的 -1 是一个占位符，PyTorch 会自动计算出 batch_size 的大小

2. 隐藏层计算

   ```python
   H = relu(X @ W1 + b1)
   ```

   - X @ W1：  这是**线性变换**的核心，即输入向量 X 与隐藏层的权重矩阵 W1 进行矩阵乘法；@ 是 PyTorch 中矩阵乘法的运算符。
   - \+ b1：  将偏置向量 b1 加到结果上（利用了 PyTorch 的广播机制）
   - relu(...)：  将线性变换的结果通过自定义的 relu **非线性激活函数**进行处理

3. 输出层计算

   ```bash
   return (H @ W2 + b2) 
   ```

   H @ W2 + b2：  隐藏层的输出 H 再与输出层的权重矩阵 W2 进行矩阵乘法，并加上偏置 b2

   这一步计算出了最终的 **logits**；因为后面使用的损失函数是 nn.CrossEntropyLoss，所以这里不需要接 Softmax 激活函数

### 3. 自定义激活函数

```python
    def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x,a)
```

这段代码从零开始手动实现了一个 ReLU 函数，其功能与 `torch.nn.ReLU()` 或 `torch.nn.functional.relu()` 完全相同

它通过将输入张量 x 与一个同形状的全零张量 a 进行逐元素比较，取其中的最大值，从而实现了 max(0, x) 的效果

### 4. 与使用 PyTorch组件构建对比

使用 PyTorch构建上面这个 一个输入层、一个隐藏层、一个输出层的代码：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(784, 256)
        self.output_layer = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.hidden_layer(x)) # 使用 functional.relu
        return self.output_layer(x)

model = MLP()
# 优化器会自动找到所有可学习参数
op = torch.optim.SGD(model.parameters(), lr=lr)
```
