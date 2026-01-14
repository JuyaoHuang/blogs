---
title: os库
publishDate: 2025-11-27
description: "os库的介绍"
tags: ['python']
language: 'Chinese'
first_level_category: "Python"
second_level_category: "工程化与工具"
draft: false
---

## 导入  os 库

```python
import os
```

## `os.path` 子模块

这是 `os` 模块中最常用的部分

它最大的作用是**解决不同操作系统路径分隔符不一样的问题**（Windows 用 `\`, Mac/Linux 用 `/`）

|              函数              |                             作用                             |
| :----------------------------: | :----------------------------------------------------------: |
| **os.path.join(path1, path2)** | 智能拼接路径：自动根据系统补充 `/` 或 `\`。**千万别用字符串 `+` 拼接路径** |
|      os.path.exists(path)      |          判断文件或文件夹是否存在。返回 True/False           |
|     os.path.basename(path)     |                    从完整路径中提取文件名                    |
|      os.path.split(path)       |                 把路径拆分为 (目录, 文件名)                  |
|     os.path.splitext(path)     |                      分离文件名和扩展名                      |

**代码示例**

```python
root_dir = "data/training"
filename = "Bread_001.jpg"

full_path = os.path.join(root_dir, filename) 
print(full_path) 
# Windows输出: data\training\Bread_001.jpg
# Mac/Linux输出: data/training/Bread_001.jpg
```

---

## 文件与目录操作

这部分用于查看文件夹里有什么，或者创建新文件夹。

|        函数         |                         作用                         |
| :-----------------: | :--------------------------------------------------: |
| **os.listdir(path** | **列出指定目录下的所有文件和子目录名**，返回一个列表 |
|     os.getcwd()     |                   获取当前工作目录                   |
|  os.makedirs(path   |      递归创建目录（如果父目录不存在会自动创建）      |

**代码示例**

```python
data_path = "./data/training"

# 获取所有图片文件名
all_files = os.listdir(data_path)
print(f"该目录下有 {len(all_files)} 个文件")
print(all_files[:3]) # 打印前3个看看 ['Bread_001.jpg', 'Bread_002.jpg', ...]
```

