---
title: 'Linux 服务器安装 Miniconda 完全指南'
publishDate: 2026-02-19
description: '在 Ubuntu 服务器上安装 Miniconda、配置国内镜像源、安装 PyTorch GPU 版本的完整流程。涵盖 conda 与 pip 的关系、虚拟环境管理，以及部署 GPT-SoVITS 等深度学习项目的实践经验。'
tags: ['linux', 'conda', 'virtual-env']
language: 'Chinese'
first_level_category: "Web全栈开发"
second_level_category: "环境与工具"
draft: false
---

# Linux 服务器安装 Miniconda 进行环境配置

> Miniconda 是 Anaconda 的精简版，仅包含 conda 包管理器和 Python，适合在服务器上按需搭建轻量级的 Python 环境。本文记录在 Ubuntu 服务器上安装 Miniconda、配置国内镜像源、安装 PyTorch GPU 版本并部署 GPT-SoVITS 项目的完整流程。

## 1. Miniconda 简介

### 1.1 为什么选择 Miniconda

| 工具 | 特点 | 适用场景 |
|---|---|---|
| Anaconda | 预装 250+ 科学计算包，体积约 3GB | 本地数据分析、教学环境 |
| Miniconda | 仅包含 conda + Python，体积约 80MB | 服务器部署，按需安装 |
| pip + venv | Python 原生方案 | 纯 Python 项目，不涉及 C/C++ 依赖 |

在 GPU 训练服务器上，Miniconda 是最佳选择：体积小、启动快，且 conda 能很好地处理 CUDA、cuDNN 等非 Python 依赖的版本管理。

### 1.2 conda 与 pip 的关系

conda 和 pip 是两套独立的包管理系统，可以共存：

- **conda**：管理 Python 包和非 Python 依赖（如 ffmpeg、cmake、CUDA 库），从 Anaconda/conda-forge 频道安装
- **pip**：仅管理 Python 包，从 PyPI 安装

实践中通常用 conda 安装基础环境和系统级依赖，用 pip 安装项目特定的 Python 包。

## 2. 安装 Miniconda

### 2.1 下载安装脚本

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 2.2 执行安装

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中的选择：
- 阅读许可协议后输入 `yes` 同意
- 安装路径：默认 `/root/miniconda3`（或 `/home/<user>/miniconda3`）
- 是否初始化 conda：选择 `yes`（会在 `~/.bashrc` 中添加 conda init 块）

### 2.3 手动激活 conda（如未自动初始化）

如果安装时未选择自动初始化，或 `.bashrc` 中没有 conda init 块，可以手动激活：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
```

可将此行添加到 `~/.bashrc` 以实现每次登录自动激活：

```bash
echo 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base' >> ~/.bashrc
```

### 2.4 验证安装

```bash
conda --version    # 例：conda 26.1.0
python --version   # 例：Python 3.13.2
```

## 3. 配置国内镜像源

默认的 Anaconda 仓库服务器在国外，国内访问速度慢。配置清华 TUNA 镜像可以大幅提升下载速度。

### 3.1 编辑 ~/.condarc

创建或编辑 `~/.condarc` 文件：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

**配置说明：**

- `default_channels`：替换默认的 main、r、msys2 频道为清华镜像
- `custom_channels`：将 conda-forge 和 pytorch 频道也指向清华镜像，安装 PyTorch 时无需单独指定镜像地址
- `show_channel_urls: true`：安装时显示包的下载来源，方便确认是否走了镜像

### 3.2 验证镜像配置

```bash
conda config --show channels
conda config --show default_channels
```

确认输出的 URL 指向 `mirrors.tuna.tsinghua.edu.cn`。

## 4. 安装系统级依赖

部分项目依赖非 Python 的系统工具，使用 conda 从 conda-forge 频道安装：

```bash
conda install --yes --quiet -c conda-forge ffmpeg cmake make unzip
```

这些工具在音频处理（ffmpeg）、编译 C 扩展（cmake/make）等场景中会用到。通过 conda 安装可以避免与系统包管理器的版本冲突。

## 5. 安装 PyTorch（GPU 版本）

### 5.1 确认 CUDA 版本

```bash
nvidia-smi
```

查看右上角的 `CUDA Version`（如 12.8），这决定了应安装哪个版本的 PyTorch。

### 5.2 安装 PyTorch

通过 pip 安装 PyTorch GPU 版本（从 PyTorch 官方源）：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

其中 `cu128` 对应 CUDA 12.8。常见的版本对应：

| CUDA 版本 | pip 后缀 |
|---|---|
| 11.8 | cu118 |
| 12.1 | cu121 |
| 12.4 | cu124 |
| 12.8 | cu128 |

### 5.3 验证 GPU 可用性

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

正常输出示例：

```
PyTorch 2.10.0+cu128
CUDA available: True
GPU: Tesla V100-PCIE-32GB
```

## 6. 部署项目依赖

### 6.1 安装 requirements.txt

进入项目目录，使用 pip 安装项目依赖：

```bash
cd /root/GPT-SoVITS-v2pro-20250604
pip install -r requirements.txt
```

GPT-SoVITS 的依赖包括：

| 类别 | 主要包 |
|---|---|
| 模型框架 | `transformers`、`peft`、`pytorch-lightning` |
| 语音处理 | `librosa`、`torchaudio`、`funasr`、`faster-whisper` |
| 文本处理 | `jieba`、`pyopenjtalk`、`g2p_en`、`g2pk2` |
| Web 服务 | `fastapi`、`gradio` |
| 推理加速 | `onnxruntime-gpu`、`ctranslate2` |

### 6.2 常见安装问题

| 问题 | 原因 | 解决方式 |
|---|---|---|
| `pyopenjtalk` 编译失败 | 缺少 cmake/make | `conda install -c conda-forge cmake make` |
| `onnxruntime-gpu` 版本不兼容 | CUDA 版本不匹配 | 指定兼容版本 `pip install onnxruntime-gpu==1.x.x` |
| numpy 版本冲突 | 某些包要求 numpy<2.0 | `pip install "numpy<2.0"` |
| 下载超时 | pip 默认源在国外 | 使用镜像 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...` |

## 7. 环境管理

### 7.1 创建独立环境（推荐）

为不同项目创建隔离环境，避免依赖冲突：

```bash
# 创建新环境
conda create -n gpt-sovits python=3.13

# 激活环境
conda activate gpt-sovits

# 在新环境中安装依赖
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 7.2 常用 conda 命令

```bash
conda env list                  # 列出所有环境
conda activate <env>            # 激活环境
conda deactivate                # 退出当前环境
conda remove -n <env> --all     # 删除环境
conda list                      # 查看当前环境的包列表
conda clean --all               # 清理缓存，释放磁盘空间
```

## 8. 启动项目

环境配置完成后，激活 conda 环境并启动 GPT-SoVITS 的 WebUI：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/GPT-SoVITS-v2pro-20250604
python webui.py zh_CN
```

`zh_CN` 参数指定 WebUI 使用中文界面。启动后终端会输出访问地址（默认 `http://0.0.0.0:7860`），通过浏览器即可访问训练和推理界面。
