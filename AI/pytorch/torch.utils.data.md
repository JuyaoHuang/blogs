---
title: 'utils.data模块'
author: Alen
published: 2025-10-28
description: "PyTorch数据加载模块torch.utils.data的介绍"
first_level_category: "人工智能"
second_level_category: "深度学习框架"
tags: ['ML','DL']
draft: false
---

# Dataset and DataLoader

## Dataset：数据的 cook


Dataset是torch.utils.data 模块中负责**数据组织和访问**的核心组件，这是 DataLoader的上位，是连接原始数据和 PyTorch模型训练的桥梁

在 PyTorch 的数据加载体系中，Dataset 和 DataLoader 分工明确：

- **Dataset**:

    负责**存储和访问**单个数据样本
    它知道总共有多少数据（`__len__`），并且知道如何根据给定的索引获取某一个特定的数据样本及其对应的标签（`__getitem__`）
    可以将其看为一份 cook，DataLoader依据这份 cook上菜
- **DataLoader**: 

    负责**打包和提供**数据
    从 Dataset 中获取数据，然后将它们整理成一份小批次 (batches)，并提供：
    
    1. 多进程加载
        ```python
        num_workers=get_dataloader_workers()
        ```
    2. 数据打乱
        ```python
        shuffle=True
        ```
        等高级功能
        可以把它想象成一个 **上菜服务员**，它按照菜单（DataLoader 的配置）从厨房（Dataset）取菜，然后端上餐桌（模型训练循环）

torch.utils.data.Dataset 是一个**抽象类**，这意味着你通常不会直接实例化它；相反，你需要**继承**这个类，并根据自己的数据格式来实现它的特定方法，从而创建一个自定义的 Dataset。

### Dataset 的核心：定义cook清单与交互协议

当你创建一个自定义 Dataset 类时，你必须实现以下两个方法，这是 Dataset 类与 DataLoader 的通信协议

**所以，如果原始的数据格式已满足需求，例如一个csv文件，就无需自定义数据格式**

#### 1. __len__(self)

- **作用**：
  
    这个方法必须返回数据集中样本的总数；DataLoader 需要通过此方法来知道总共有多少数据，以便确定迭代的次数、最后一个批次的大小等
- **参数**：
  - self:   类实例本身
- **返回值**：  一个整数，表示数据集的大小

#### 2. __getitem__(self, index)

- **作用**：
  
    负责**根据给定的索引 index，获取并返回数据集中对应的一个样本**；这是 Dataset 的核心所在，所有的数据加载、预处理、转换等逻辑都发生在这里
- **参数**：
  - self:   类实例本身
  - index (int):   一个从 0 到 len(dataset)-1 的整数索引；DataLoader 会自动为你生成和传入这个索引
- **返回值**：  通常是一个元组 (data, label)
  - data:   一个数据样本，通常是一个 torch.Tensor（例如，一张图片、一段文本的编码）
  - label:   该数据样本对应的标签，可以是一个整数（用于分类）或一个浮点数/张量（用于回归）

### 自定义 Dataset 的通用结构

一个典型的自定义 Dataset 类通常包含第三个方法 __init__：

#### 3. __init__(self, ...)

- **作用**：
    构造函数；在这里执行所有只需要进行一次的初始化操作
    例如：
  - 加载数据清单文件（如 CSV、JSON）
  - 将文件名和标签加载到内存中（通常是 list 或 pandas DataFrame）
  - 定义数据转换（transformations）
- **参数**：  通常会接收数据文件路径、标签文件路径、以及一个可选的 transform 对象

------

### 示例：创建一个自定义图像数据集

假设数据组织形式为：

```bash
/path/to/your/data/
├── images/
│   ├── 001.jpg
│   ├── 002.jpg
│   ├── ...
│   └── 100.png
└── labels.csv
  
```

labels.csv 文件的内容如下：

```bash
image_filename,label
001.jpg,cat
002.jpg,dog
...
100.png,cat  
```

下面是如何为这个数据集创建一个自定义的 CustomImageDataset：

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision import transforms

# --- 为了让示例可独立运行，创建假的数据集 ---
def create_dummy_dataset(root_dir="dummy_data", num_samples=20):
    """创建一个假的图像数据集用于演示"""
    img_dir = os.path.join(root_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # 创建假的CSV文件
    labels = []
    for i in range(num_samples):
        filename = f"{i:03d}.png"
        label = "cat" if i % 2 == 0 else "dog"
        labels.append({"image_filename": filename, "label": label})
        
        # 创建假的图像文件 (10x10的随机像素)
        dummy_img = Image.fromarray((torch.rand(10, 10, 3) * 255).numpy().astype('uint8'))
        dummy_img.save(os.path.join(img_dir, filename))

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(root_dir, "labels.csv"), index=False)
    print(f"Dummy dataset created at '{root_dir}'")

create_dummy_dataset()
# -----------------------------------------------------------------


# 1. 定义自定义 Dataset 类，继承 torch.utils.data.Dataset
class CustomImageDataset(Dataset):
    # (1) __init__: 初始化，加载数据清单
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        Args:
            annotations_file (string): 包含标签的CSV文件路径
            img_dir (string): 包含所有图像的目录路径
            transform (callable, optional): 应用于图像样本的可选转换
            target_transform (callable, optional): 应用于标签的可选转换
        """
        # 读取CSV文件
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # 为了演示，我们将文本标签映射为整数
        self.class_map = {"cat": 0, "dog": 1}

    # (2) __len__: 返回数据集的总大小
    def __len__(self):
        return len(self.img_labels)

    # (3) __getitem__: 根据索引加载并返回一个样本
    def __getitem__(self, idx):
        # 构造图像文件的完整路径
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # 加载图像
        image = Image.open(img_path).convert("RGB") # 确保图像是RGB格式
        
        # 获取标签
        label_text = self.img_labels.iloc[idx, 1]
        label = self.class_map[label_text] # 转换为整数标签

        # 应用转换
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

# 2. 实例化自定义 Dataset
data_root = "dummy_data"
csv_file = os.path.join(data_root, "labels.csv")
image_directory = os.path.join(data_root, "images")

# 定义一些图像转换
# 调整大小 -> 转换为Tensor -> 标准化
image_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建数据集实例
custom_dataset = CustomImageDataset(
    annotations_file=csv_file,
    img_dir=image_directory,
    transform=image_transforms
)

# 3. 验证 Dataset 的功能
print(f"\nDataset size: {len(custom_dataset)}") # 调用 __len__

# 获取第一个样本
first_image, first_label = custom_dataset[0] # 调用 __getitem__(0)
print(f"\nFirst sample - Image shape: {first_image.shape}, Label: {first_label}")
# 检查数据类型
print(f"Image dtype: {first_image.dtype}, Label type: {type(first_label)}")

# 4. 将 Dataset 传递给 DataLoader
# 这是 Dataset 的最终用途
data_loader = DataLoader(
    dataset=custom_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0 # 在Windows上，多进程需要特殊处理，先设为0
)

# 从 DataLoader 中获取一个批次的数据
# 这会在内部多次调用 custom_dataset.__getitem__()
images_batch, labels_batch = next(iter(data_loader))
print(f"\nBatch of images shape: {images_batch.shape}") # (batch_size, C, H, W)
print(f"Batch of labels shape: {labels_batch.shape}")
print(f"Labels in the batch: {labels_batch}")
  
```

### 总结

- Dataset 是 PyTorch 数据加载机制的**基础**，它负责**组织和访问**原始数据
- 创建自定义 Dataset 需要继承 torch.utils.data.Dataset 并实现 `__len__` 和 `__getitem__` 两个核心方法
- `__init__` 用于一次性的初始化工作，如加载文件列表
- `__len__` 返回数据集的大小
- `__getitem__` 是核心，负责根据索引加载单个数据、应用转换，并返回 (data, label) 对
- 创建好的 Dataset 对象最终会被传递给 DataLoader，以便进行高效的批处理、打乱和多进程加载，为模型训练提供源源不断的数据流

---

## DataLoader：数据加载工具

如果 Dataset 是一个包含所有菜品并知道如何制作每一道菜的cook，那么 DataLoader 就是一个高效的"上菜服务员团队"

DataLoader 的核心职责是**从 Dataset 中取出单个样本，并将它们高效地组合成一个个批次**，然后将这些批次提供给训练循环。它封装了所有复杂的后台逻辑，如：

- **数据批处理  Batching**：  将多个样本打包成一个批次张量
- **数据打乱  Shuffling**：  在每个 epoch 开始时随机打乱数据顺序，以增强模型的泛化能力
- **并行加载  Parallel Loading**：  使用多个子进程 (workers) 在后台预加载数据，这样 GPU 在进行计算时，CPU 已经在准备下一个批次的数据了，从而避免了数据加载成为训练瓶颈
- **自定义整理  Custom Collation**：  允许开发者自定义如何将多个样本合并成一个批次，这在处理可变长度数据（如文本序列）时非常有用

DataLoader 是一个 Python 的**可迭代对象**，可以像**遍历列表**一样在 for 循环中轻松地使用它

------

### DataLoader 的核心参数详解

DataLoader 的功能主要通过其构造函数中的参数来配置；以下是最重要和最常用的参数：

1. **dataset (Dataset)**:
   - **作用**：  这是 DataLoader 的数据源，必须是一个 torch.utils.data.Dataset 的对象；DataLoader 将从这个 dataset 中拉取数据。这是**唯一一个必需的参数**
   
2. **batch_size (int, optional)**:
   - **作用**：  指定每个批次包含的样本数量。默认为 1
   - **重要性**：  

        批处理是深度学习训练的基础；它可以在一次前向/反向传播中处理多个样本，这样计算出的梯度更稳定，并且能充分利用 GPU 的并行计算能力。
        batch_size 是一个需要根据 GPU 显存大小和模型进行调整的关键超参数。=
   
3. **shuffle (bool, optional)**:
   - **作用**：  如果设置为 True，则在每个 epoch 开始前都会重新打乱数据的顺序；默认为 False。
   - **重要性**：  

        在**训练**时，打乱数据至关重要；它可以防止模型学习到数据的特定顺序，从而避免过拟合，提高泛化能力
        在**验证或测试**时，通常设置为 False，因为不需要打乱，而且保持顺序一致有助于结果的复现和比较
   
4. **num_workers (int, optional)**:
   - **作用**：  指定用于数据加载的子进程数量；默认为 0。
     - num_workers=0 (默认): 数据将在主进程中加载
     - num_workers > 0: PyTorch 会启动指定数量的子进程在后台并行加载数据

   - **重要性**：  这是提升训练效率的关键参数；如果数据预处理（如图像解码、数据增强）比较耗时，设置 num_workers > 0 可以显著加快训练速度
   - **Windows 用户注意**：  在 Windows 系统上，使用多进程 (num_workers > 0) **必须**将主执行代码放在 `if __name__ == '__main__':` 块中，否则会引发错误
   
5. **pin_memory (bool, optional)**:
   - **作用**：  如果设置为 True，DataLoader 会将加载的数据张量复制到**锁页内存 (pinned memory)** 中；默认为 False
   - **重要性**：  
    
      这是在 **使用 GPU 训练时**的一个性能优化选项；从锁页内存将数据传输到 CUDA 设备的 GPU 显存通常会快得多
      因此，当 num_workers > 0 并且在 GPU 上训练时，建议将此项设置为 True
   
6. **drop_last (bool, optional)**:
   
   - **作用**：
   
     如果数据集的总大小不能被 batch_size 整除，那么最后一个批次的大小会小于 batch_size
   
     如果 drop_last 设置为 True，这个不完整的最后一个批次将被丢弃；默认为 False
   
   - **重要性**：  在某些模型（如需要固定输入大小的RNN）或分布式训练中，可能需要所有批次的大小都相同
   
7. **collate_fn (callable, optional)**:
   - **作用**：  指定一个函数，用于将从 Dataset 中获取的样本列表（list of samples）合并成一个批次
   
   - **默认行为**：  PyTorch 的默认 collate_fn 会尝试将样本中的每个元素（如图像张量、标签）用 torch.stack 堆叠起来
   
   - **重要性**：
   
     当处理的数据样本无法被简单堆叠时（例如，在 NLP 中，每个句子长度不同），你需要提供一个自定义的 collate_fn 来进行填充 (padding) 等操作，以使它们形状一致；这是一个非常强大的高级功能。

------

### 示例：使用 DataLoader 包装自定义 Dataset

继续使用上一节中创建的 CustomImageDataset，并展示如何用 DataLoader 来包装它：

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision import transforms

# --- 重用上一节的 Dataset 定义和虚拟数据创建函数 ---
def create_dummy_dataset(root_dir="dummy_data", num_samples=50): # 增加样本数以便演示
    img_dir = os.path.join(root_dir, "images")
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    labels = [{"image_filename": f"{i:03d}.png", "label": "cat" if i % 2 == 0 else "dog"} for i in range(num_samples)]
    for item in labels:
        dummy_img = Image.fromarray((torch.rand(10, 10, 3) * 255).numpy().astype('uint8'))
        dummy_img.save(os.path.join(img_dir, item["image_filename"]))
    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(root_dir, "labels.csv"), index=False)
    print(f"Dummy dataset created at '{root_dir}'")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_map = {"cat": 0, "dog": 1}
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.class_map[self.img_labels.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------------------------------------------------------------------
# 主执行代码 - 必须放在 if __name__ == '__main__': 保护块中，以便安全使用 num_workers
# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. 创建虚拟数据集
    create_dummy_dataset()

    # 2. 实例化 Dataset
    image_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    full_dataset = CustomImageDataset(
        annotations_file="dummy_data/labels.csv",
        img_dir="dummy_data/images",
        transform=image_transforms
    )

    # 3. 实例化 DataLoader (分别为训练和验证)
    # 训练 DataLoader: 打乱数据，使用多进程
    train_dataloader = DataLoader(
        dataset=full_dataset,
        batch_size=8,
        shuffle=True,      # 打乱数据
        num_workers=2,     # 使用2个子进程加载数据
        pin_memory=True,   # 如果使用GPU，建议开启
        drop_last=True     # 丢弃最后一个不完整的批次
    )

    # 验证 DataLoader: 不打乱，通常可以使用更大的 batch_size
    val_dataloader = DataLoader(
        dataset=full_dataset,
        batch_size=16,
        shuffle=False,     # 验证时不需要打乱
        num_workers=0      # 验证时数据量小，0也可以
    )

    # 4. 演示如何使用 DataLoader
    print(f"\nTotal samples in dataset: {len(full_dataset)}")
    
    # 训练 DataLoader 演示
    print("\n--- Iterating through train_dataloader ---")
    print(f"Number of batches in train_dataloader: {len(train_dataloader)}")
    
    # 从 DataLoader 中获取一个批次
    # 这是一个典型的训练循环的开始
    images_batch, labels_batch = next(iter(train_dataloader))
    
    print(f"Shape of one batch of images: {images_batch.shape}") # [batch_size, C, H, W]
    print(f"Shape of one batch of labels: {labels_batch.shape}")
    print(f"Labels in this batch: {labels_batch.numpy()}")
    
    # 模拟一个训练 epoch
    print("\nSimulating a training epoch loop...")
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        # 在这里，你会将 data 和 targets 移动到 GPU
        # data, targets = data.to(device), targets.to(device)
        # 然后进行模型的前向传播、计算损失、反向传播...
        if batch_idx < 2: # 只打印前两个批次的信息
            print(f"  Batch {batch_idx+1}: Image batch shape {data.shape}, Label batch shape {targets.shape}")
    print("Training epoch simulation finished.")
  
```



## 总结：Dataset vs DataLoader

| 特性              | torch.utils.data.Dataset                                 | torch.utils.data.DataLoader                                 |
| ----------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| **角色**          | **数据的“蓝图”或“食谱”**                                 | **数据的“服务员”或“管道”**                                  |
| **核心功能**      | 存储数据源信息，并定义如何**获取单个样本** (__getitem__) | 从 Dataset 中获取数据，并**打包成批次** (batch)             |
| **主要方法/参数** | __len__, __getitem__                                     | dataset, batch_size, shuffle, num_workers                   |
| **输出**          | 一次返回**一个**样本 (data, label)                       | 一次返回**一个批次**的样本 (batch_of_data, batch_of_labels) |
| **关注点**        | **数据表示和访问** (What and How to get one)             | **数据加载效率和策略** (How to serve efficiently)           |

正确地组合使用 Dataset 和 DataLoader 是构建高效、可读性强的 PyTorch 数据输入管道的关键。