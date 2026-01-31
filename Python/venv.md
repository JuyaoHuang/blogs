---
title: python 的三种虚拟环境
publishDate: 2026-01-31
description: "介绍 py 中的三种虚拟环境：conda、venv、uv 的创建方式"
tags: ['python']
language: 'Chinese'
first_level_category: "Python"
second_level_category: "基础语法"
draft: false
---

在 Python 开发中，使用虚拟环境可以帮助我们隔离不同项目的依赖，避免包版本冲突。常用的三种虚拟环境工具有 conda、venv 和 uv。下面介绍它们的创建方式。

## 1. conda 虚拟环境

conda 虚拟环境的介绍可查看[此篇文章](https://www.juayohuang.top/blog/webfullstack/conda/condavirtualenvconfigguidance)

## 2. venv 虚拟环境

venv 是 Python 标准库自带的虚拟环境工具，适用于大多数项目。由于它不需要安装额外的软件，是目前 Python 项目中最基础、最通用的选择。

### 2.1. 创建 venv 虚拟环境

打开终端或命令行，导航到你的项目目录，然后运行以下命令来创建一个新的虚拟环境：

```bash
# 通用语法：python -m venv <虚拟环境名称>

# 推荐做法（将环境命名为 .venv 或 venv）
python -m venv .venv
```
`.venv`：这是虚拟环境文件夹的名字。
*   *习惯 1*：命名为 `.venv`（前面有个点），这样它默认是隐藏的，且通常会被编辑器自动识别。
*   *习惯 2*：命名为 `venv`，也是非常常见的做法。

---

**注意**：

与 conda 不同， venv 不支持使用类似  `python -m venv --python=3.8` 来指定 Python 版本。

如果你想创建一个 Python 3.8 的虚拟环境，你必须先在你的电脑上找到并运行 Python 3.8 的解释器。

在 Windows 上安装 Python 时，通常会默认安装一个叫 `Python Launcher` (`py.exe`) 的工具。它可以帮你轻松指定版本。

创建一个 Python 3.9 的环境：

```cmd
# 语法：py -[版本号] -m venv [环境名]
py -3.9 -m venv .venv
```

在 Mac 或 Linux 中，不同版本的 Python 通常以 `python3.x` 的命令存在。

假设你想创建一个 Python 3.10 的环境：

```bash
# 直接调用该版本的命令
python3.10 -m venv .venv
```

---

### 2.2. 激活 venv 虚拟环境

激活虚拟环境：

#### **Windows（cmd）**

```bash
# 如果你的虚拟环境目录是 .venv
.\.venv\Scripts\activate
```

#### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

**注意**：如果 PowerShell 提示“禁止运行脚本”，你需要先执行一次 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` 来允许脚本运行。
#### **macOS 或 Linux**

```bash
# 如果你的虚拟环境目录是 .venv
source .venv/bin/activate
```

### 2.3. 在 venv 中安装依赖

导出依赖列表（便于分享给他人），生成 `requirements.txt` 文件。
```bash
pip freeze > requirements.txt
```

### 2.4. 退出 venv 虚拟环境

```bash
deactivate
```

### 2.5. 删除 venv 虚拟环境

`venv` 创建的虚拟环境本质上就是一个文件夹（例如 `.venv`）。如果不再需要这个环境，或者环境坏了想重来，**直接删除这个文件夹即可**。

```bash
# Linux / macOS
rm -rf .venv

# Windows
rmdir /s /q .venv
```

> 千万不要把虚拟环境文件夹（`.venv`）提交到 Git 仓库中
> 
> 你应该在项目根目录创建一个 `.gitignore` 文件，并在其中添加一行：
> 
> ```text
> .venv/
> ```
## 3. uv 虚拟环境

uv 是一个轻量级的虚拟环境管理工具，适合快速创建和切换虚拟环境。可在此篇文章获得 uv 的详细介绍和使用方法：[uv 虚拟环境介绍](https://www.juayohuang.top/blog/python/toolsets/uv)


uv 是目前 Python 社区最炙手可热的工具，由 Rust 编写。它的核心特点只有一个字：快。

它不仅比 pip 快 10-100 倍，而且是一个大一统工具：它可以替代 pip、pip-tools（锁定依赖）、virtualenv（创建环境）、甚至 pyenv（管理 Python 版本）。

uv 的使用方式主要分为两种模式：现代项目模式和 Pip 兼容模式。

### 3.1. 安装 uv

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

或者通过 pip :
```bash
pip install uv
```

### 3.2. 创建 uv 虚拟环境

如果你不希望在全局安装 uv，那么需要先使用 python 自带的 venv 创建一个虚拟环境，然后在该环境中安装 uv。

但这样会导致无法使用 uv 指定特定的 Python 版本来创建虚拟环境。

**1. 使用 uv 创建一个类似 venv 的虚拟环境：**

```bash
# 创建一个基于 Python 3.10 的虚拟环境（默认目录名为 .venv）
uv venv --python 3.10

# 或者指定环境名称
uv venv my_env --python 3.10
```

这样 uv 会先检查系统是否有 python 3.10，如果没有，它会自动从官方源下载并安装该版本，并使用这个版本创建虚拟环境。

然后和普通的 venv 一样，你需要手动激活该环境：

```bash
# 在 Windows 上
venv\Scripts\activate
# 或者在 macOS / Linux 上
source venv/bin/activate
```

**2. 项目开发的最佳实践**

在正式开发中，通常推荐使用 `uv init` 初始化项目，这样可以将 Python 版本锁定在配置文件中，方便团队协作。

1. 初始化项目并指定版本：

   ```bash
   # 创建新项目文件夹并初始化
   uv init my_project --python 3.10
   
   # 或者在已有文件夹中初始化
   cd my_existing_project
   uv init --python 3.10
   ```

   这会生成一个 `.python-version` 文件（内容只有版本号，如 `3.9.18`）和一个 `pyproject.toml` 文件。

2. 修改现有项目的 Python 版本

   把一个现有项目的 Python 版本从 3.11 切换到 3.12：

   ```bash
   uv python pin 3.12
   ```
   这会更新 `.python-version` 文件，并自动下载所需的 Python 版本。

3. 运行代码：

  在项目目录下，不需要手动激活环境，直接运行：

  ```bash
  # uv 会自动识别 .python-version 指定的版本来运行
  uv run python app.py
  ```

### 3.3. 项目管理模式

uv 目前主推的用法，类似于 Poetry 或 npm，全自动管理虚拟环境，不需要手动激活环境。

#### **初始化项目**
```bash
# 创建一个新文件夹并初始化
mkdir my-project
cd my-project
uv init
```
生成一个 `pyproject.toml` 文件（现代 Python 项目标准配置文件）。

#### **安装依赖**
不需要手动激活环境，直接运行：
```bash
uv add requests pandas
```
uv 会**自动**做以下几件事：
1.  自动创建一个虚拟环境（默认在 `.venv`）
2.  下载并安装 `requests` 和 `pandas`
3.  将依赖写入 `pyproject.toml`
4.  生成一个 `uv.lock` 文件（锁定版本，确保安全）

#### **运行代码**

这是 `uv` 最爽的功能。你不需要 `source .venv/bin/activate`，直接用 `uv run`：
```bash
# 运行脚本
uv run main.py

# 或者临时运行一个命令
uv run python -c "import requests; print(requests.__version__)"
```
`uv` 会自动检测并在它管理的虚拟环境中运行该命令。

注意：uv run 默认用 .venv，而你的虚拟环境是 wx-vtuber

```bash
wx-vtuber\Scripts\python.exe test_api.py
# 或者
uv run --active test_api.py
```

#### **同步环境**
如果你下载了别人的代码（含有 `uv.lock`），只需运行：
```bash
uv sync
```
它会瞬间把环境还原成和作者一模一样的状态。

### 3.4. uv 下载的 py 在哪

使用  `uv python install 3.12` 或 `uv init --python 3.12` 时，uv 会将 py 安装在一个全局目录中。

具体路径（默认情况）：

*   macOS：~/Library/Application Support/uv/python/
    或者 ~/.local/share/uv/python/
*   Linux：~/.local/share/uv/python/
*   Windows：%LOCALAPPDATA%\uv\python\
    (通常是 C:\Users\用户名\AppData\Local\uv\python\`

## 三者优劣

了解了 conda、venv 和 uv 的基本用法后，我们从创建机制、依赖管理、性能等维度对它们进行深度比对。

### 4.1. 核心特性对比表

| 维度                 | **venv (标准库)**             | **conda (Anaconda)**              | **uv (新一代)**                      |
| :------------------- | :---------------------------- | :-------------------------------- | :----------------------------------- |
| **安装成本**         | Python 内置                   | 安装 Miniconda/Anaconda           | 单文件/pip 安装                      |
| **Python 版本管理**  | ❌ 弱 (仅限当前系统安装的版本) | ✅ **强** (可任意指定并下载版本)   | ✅ **极强** (自动下载 Managed Python) |
| **非 Python 库支持** | ❌ 不支持 (需系统预装 C 库)    | ✅ **强** (支持 CUDA, GCC, FFmpeg) | ❌ 不支持 (仅限 PyPI)                 |
| **依赖锁定机制**     | 弱 (仅靠 requirements.txt)    | 中 (environment.yml)              | ✅ **强** (uv.lock 全局锁定)          |
| **运行速度**         | 普通                          | 较慢 (尤其是依赖解析)             | 🚀 **极快** (Rust 编写)               |
| **磁盘占用**         | 高 (每个环境复制一套包)       | 较高 (但有硬链接优化)             | ✅ **极低** (全局缓存 + 硬链接)       |

### 4.2. 深度解析

#### **1. 创建与版本管理的灵活性**
*   venv：最大的痛点在于它看天吃饭。如果你的系统里只装了 Python 3.8，你就造不出 Python 3.10 的环境。想切版本，你得先去官网下载安装包，配置环境变量，比较繁琐。
*   conda：自带版本库。一句 `conda create -n myenv python=3.9` 就能解决问题，非常适合需要频繁在不同 Python 版本间横跳的用户。
*   uv：它不仅能像 Conda 一样自动下载安装 Python 版本，而且下载速度和环境构建速度是毫秒级的。

#### **2. 依赖管理的侧重点**
*   conda：数据科学的神。很多科学计算包（如 NumPy, PyTorch）依赖底层的 C/C++ 库。在 Windows 上用 pip 编译这些源码往往会报错，而 Conda 提供的是预编译好的二进制包，省去了编译痛苦，且能管理 CUDA 等非 Python 依赖。
*   venv：适合大多数纯 Python 项目，但在处理复杂依赖树时，容易出现“依赖地狱”。
*   uv：引入了类似 `npm` 或 `Poetry` 的 `uv.lock` 文件机制，确保“在我机器上能跑，在你机器上也能跑”。同时，它利用硬链接技术，多个项目共用同一个包缓存，极大地节省了磁盘空间。

#### **3. 速度与体验**
*   uv 是绝对的王者。在安装大型依赖（如 PyTorch）或解析复杂的版本冲突时，uv 比 pip 快 10 到 100 倍。
*   conda 的依赖解析器在包很多时会变得非常慢（虽然 libmamba 改善了这一点，但仍不及 uv）。

### 4.3. 最终选取建议

根据你的开发场景，对号入座：

1. 如果你是数据科学家、AI 研究员，或者是 Windows 初学者：

   👉 首选 Conda (Miniconda)。

   *理由*：你不想折腾 C++ 编译器错误，你需要一行命令装好 PyTorch 和 CUDA 环境。

2. 如果你是 Web 后端开发者、开源库维护者，追求工程规范：

   👉 首选 uv。*理由*：你需要极速的 CI/CD 构建速度，严格的 `uv.lock` 依赖锁定，以及便捷的项目管理体验。它是 Poetry 的完美极速替代品。

3. 如果你只是写个小脚本，或者在 Docker 容器里部署应用：

   👉 首选 venv。

   *理由*：Docker 容器本身已经是隔离环境，再装 uv 或 conda 略显臃肿。系统自带的 venv 足够轻量，开箱即用。
