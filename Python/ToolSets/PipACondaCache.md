---
title: pip与conda包缓存介绍
author: Alen
published: 2025-10-10
description: "pip与conda包缓存介绍"
first_level_category: "Python"
second_level_category: "工程化与工具"
tags: ['python']
draft: false
---


## 一、Pip 缓存管理

`pip` 主要使用一个缓存目录来存储下载的 wheel 文件和源码包。

### 1. 查看 Pip 缓存目录

使用以下命令来查看当前 `pip` 缓存目录的路径：

```bash
pip cache dir   
```

**示例输出 (Windows):**

```
C:\Users\YourUsername\AppData\Local\pip\Cache
```

**示例输出 (Linux/macOS):**

```
~/.cache/pip
```

### 2. 查看 Pip 缓存信息 (占用空间等)

你可以查看缓存中存储的包信息和总体积：

```
pip cache info
```



**示例输出:**

```bash
Cache directory: /home/user/.cache/pip
Total number of items in cache: 150
Total size of cache: 1.2 GB
Number of wheels: 100
Number of http files: 50    
```

### 3. 列出 Pip 缓存中的包



可以列出缓存中特定包或所有包：

```
# 列出所有缓存的包
pip cache list

# 列出名为 'requests' 的包的缓存项 (支持通配符)
pip cache list requests    
```

### 4. 清除 Pip 缓存

#### a. 清除所有缓存

这将删除缓存目录下的所有文件：

```
pip cache purge
```

执行此命令后，pip 会询问你是否确认删除。

#### b. 清除特定包的缓存

如果你只想删除某个特定包的缓存文件：

```
pip cache remove <package_name>
```

例如，删除 requests 包的缓存：

```
pip cache remove requests
```

### 5. (可选) 更改 Pip 缓存目录位置

如果你希望将 pip 的缓存目录移动到其他磁盘（例如 D 盘），可以设置环境变量 PIP_CACHE_DIR。

**Windows (临时设置，仅当前 CMD/PowerShell 会话有效):**

```
 $env:PIP_CACHE_DIR = "D:\pip_cache"
# 或者在 CMD 中:
# set PIP_CACHE_DIR=D:\pip_cache
```

**Linux/macOS (临时设置，仅当前终端会话有效):**

```
export PIP_CACHE_DIR="/path/to/your/d_drive_cache" 
```

要永久更改，需要将此环境变量添加到系统环境变量或用户的 shell 配置文件中（如 .bashrc, .zshrc, 或通过 Windows 的系统属性设置）。

------



## 二、Conda 缓存管理



conda 也有自己的包缓存系统，通常位于 Anaconda/Miniconda 安装目录下的 pkgs 文件夹，或者用户目录下的 .conda/pkgs。

### 1. 查看 Conda 缓存信息



使用 conda info 命令可以查看包括包缓存（package cache）在内的多种 conda 配置信息：

```
conda info    
```

在输出中找到 package cache 或 pkgs directories 相关的行，会显示缓存路径。

### 2. 清除 Conda 缓存



conda 提供了 clean 命令来管理和清除缓存。

#### a. 清除已下载的包（tarballs）

这将删除 pkgs 目录中下载的但尚未解压的 .tar.bz2 或 .conda 包文件：

```bash
conda clean --tarballs
# 或者简写
conda clean -t
```

#### b. 清除未使用的已安装包



这将删除在任何环境中都没有被引用的已安装包（即 pkgs 目录中已解压但不再被任何环境需要的包）：

```bash
conda clean --packages
# 或者简写
conda clean -p
```



#### c. 清除索引缓存、锁文件等

```bash
conda clean --index-cache
conda clean --lock
conda clean --tempfiles    
```



#### d. 清除所有类型的缓存（推荐）

最常用的命令是清除所有可清理的缓存，包括以上几种：

```bash
conda clean --all 
```

执行此命令时，conda 可能会列出将要删除的文件和目录，并询问你是否确认。

### 3. (可选) 管理 Conda 的 pkgs_dirs

通常情况下，Conda 的主包缓存目录 (pkgs_dirs 的第一个路径) 会跟随 Anaconda/Miniconda 的安装位置。如果你想添加额外的包缓存目录或更改其优先级，可以修改 .condarc 配置文件中的 pkgs_dirs 设置。但这属于更高级的配置，一般用户较少需要直接修改。

------



**总结：**

定期清理 pip 和 conda 的缓存是一个好习惯，尤其是在磁盘空间紧张或者遇到奇怪的包安装问题时。使用 pip cache purge 和 conda clean --all 是快速释放空间和解决潜在缓存冲突的有效方法。如果有特定的需求，可以根据上述命令进行更细致的缓存管理。

