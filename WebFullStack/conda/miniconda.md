---
title: "conda镜像源配置"
published: 2025-12-02
description: "在云服务器上下载miniconda并配置镜像源"
tags: ['conda']
first_level_category: "项目实践"
second_level_category: "DeepLearning"
draft: false
---

## 镜像源

**华为云**

```bash
# 1. 添加华为云的 Anaconda 仓库
conda config --add channels https://repo.huaweicloud.com/anaconda/pkgs/free/
conda config --add channels https://repo.huaweicloud.com/anaconda/pkgs/main/
# 2. 添加华为云的 PyTorch 仓库
conda config --add channels https://repo.huaweicloud.com/anaconda/cloud/pytorch/
conda config --add channels https://repo.huaweicloud.com/anaconda/cloud/conda-forge/
conda config --add channels https://repo.huaweicloud.com/anaconda/cloud/msys2/
```

**清华源**

```bash
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```

## 实践示例


### 下载miniconda

使用华为镜像源下载 x86 版本

```bash
wget https://repo.huaweicloud.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 配置 pip 全局镜像

使用华为云作为全局的 pip 镜像

```bash
# 全局配置
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple/
```

### 激活环境

```bash
# 激活环境
source ~/.bashrc
```

### 创建conda环境

```bash
conda create -n dl python=3.10 -y
```

### 设置搜索时显示通道地址

```bash
conda config --set show_channel_urls yes
```

### 配置 conda 镜像

```bash
conda config --add channels https://repo.huaweicloud.com/anaconda/pkgs/free/
conda config --add channels https://repo.huaweicloud.com/anaconda/pkgs/main/
conda config --add channels https://repo.huaweicloud.com/anaconda/cloud/pytorch/
conda config --add channels https://repo.huaweicloud.com/anaconda/cloud/conda-forge/
conda config --add channels https://repo.huaweicloud.com/anaconda/cloud/msys2/
```

### 删除镜像配置（可选）

```bash
conda config --remove-key channels
conda config --remove-key show_channel_urls
conda config --show channels # 如果成功恢复默认，你会看到输出里只有 defaults，或者什么都不显示
```

### 激活环境

```bash
conda activate dl
```

### 安装环境依赖


```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
uv pip install matplotlib tensorboard scikit-learn numpy pandas seaborn kagglehub
```

