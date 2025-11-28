---
title: uv
author: Alen
published: 2025-10-10
description: "包环境下载和管理工具：UV的介绍"
first_level_category: "Python"
second_level_category: "工程化与工具"
tags: ['python']
draft: false
---


# uv

## 介绍

可以把 uv 理解为一个**速度极快的 Python 包安装器和虚拟环境管理器**，它是一个“瑞士军刀”式的工具，旨在取代 pip, pip-tools, venv, virtualenv 等多个工具。

**uv 的核心功能 = python -m venv + pip + pip-tools**

- **创建虚拟环境：** 像 venv 一样，但速度快到几乎瞬时。
- **安装/卸载包：** 像 pip 一样，但由于其智能缓存和并行处理，速度快 10-100 倍。
- **依赖解析和锁定：** 像 pip-tools (pip-compile, pip-sync) 一样，可以帮你管理项目的依赖关系，生成锁定的 requirements.txt 文件，确保环境可复现。

## 使用方法

### 安装

uv 本身是一个独立的二进制文件，官方推荐使用 pip3 来安装，这样可以避免污染你全局的 Python 环境。

- **如果你没有 pip3，先安装它：**

  ```bash
  python -m pip install --user pip3
  python -m pip3 ensurepath
  ```    (安装后可能需要重启终端)
  ```

- **使用 pip3 安装 uv：**

  ```bash
  pip3 install uv
  ```

- **或者，如果你不想用 pipx，也可以直接用 pip：**

  ```bash
  pip install uv
  ```

安装后，通过运行 uv --version 来验证是否成功。

### 核心命令

uv 的命令设计得非常直观。我们来看几个最常见的场景。

#### 场景一：从零开始一个新项目

这是最能体现 uv 一体化优势的流程。

**1. 创建虚拟环境 (替代 python -m venv .venv)**

```
# 在你的项目根目录下运行
uv venv
```

这个命令会**瞬间**创建一个名为 .venv 的虚拟环境。比 python -m venv 快非常多。

**2. 激活虚拟环境**

这步和传统方式一样。

- **macOS / Linux (bash/zsh):**

  ```
  source .venv/bin/activate
  ```

- **Windows (PowerShell):**

  ```
  .venv\Scripts\Activate.ps1
  ```

- **Windows (CMD):**

  ```
  .venv\Scripts\activate.bat
  ```

**3. 安装包 (替代 pip install)**

```bash
# 安装单个包
uv pip install requests

# 安装多个包
uv pip install "fastapi[all]" "uvicorn[standard]"

# 从 requirements.txt 文件安装
uv pip install -r requirements.txt
```

你会发现，即使是第一次安装 pandas 或 torch 这样的大型包，uv 的速度也比 pip 快得多，因为它会并行下载和构建依赖。

#### 场景二：管理项目依赖 (替代 pip-tools)

这是 uv 非常强大的功能，可以帮助你创建可复现的、干净的开发环境。

**1. 在 pyproject.toml 中定义依赖**

现代 Python 项目推荐使用 pyproject.toml 来声明项目依赖。

在你的项目根目录创建 pyproject.toml 文件：

```toml
# 文件名: pyproject.toml

[project]
name = "my-awesome-project"
version = "0.1.0"
dependencies = [
    "fastapi",
    "requests",
    "pydantic<2",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff", # 你的好朋友 Ruff
]
```

**2. 生成锁定的 requirements.txt (替代 pip-compile)**

这个命令会读取 pyproject.toml，解析所有依赖（包括依赖的依赖），并生成一个精确的版本锁定文件。

```bash
uv pip compile pyproject.toml -o requirements.txt
```

如果你有开发依赖，可以这样生成 requirements-dev.txt：

```bash
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```

**3. 同步虚拟环境 (替代 pip-sync)**

uv pip sync 是一个非常有用的命令。它会确保你的虚拟环境**严格地**与 requirements.txt 文件中的内容保持一致。不多不少。

```bash
# 安装/卸载包，使环境与 requirements.txt 完全匹配
uv pip sync requirements.txt
```

- **和 install -r 的区别：**
  - install -r 只会**安装或升级**包，不会删除你手动安装但不在文件里的包。
  - sync 会**安装、升级、并删除**任何不在文件里的包，保证环境的纯净。

### 其他常用命令

- **卸载包：**

  ```bash
  uv pip uninstall requests
  ```

- **查看已安装的包：**

  ```bash
  uv pip list
  ```

- **冻结当前环境的包版本：**

  ```
  uv pip freeze
  ```

- **清理 uv 的全局缓存：**

  ```
  uv cache clean
  ```

------



## 完整uv 工作流示例

1. **创建项目并初始化环境**

   ```bash
   mkdir my_uv_project
   cd my_uv_project
   uv venv
   source .venv/bin/activate
   ```

2. **定义依赖**
   创建 pyproject.toml 并写入：

   ```toml
   [project]
   name = "my_uv_project"
   version = "0.1.0"
   dependencies = [
       "requests",
   ]
   ```

3. **编译和同步**

   ```bash
   # 编译依赖
   uv pip compile pyproject.toml -o requirements.txt
   
   # 同步环境
   uv pip sync requirements.txt
   ```

4. **写代码并运行**
   创建 main.py：

   ```py
   import requests
   
   def get_astral_blog():
       try:
           response = requests.get("https://astral.sh/blog")
           response.raise_for_status() # 检查请求是否成功
           print("成功获取 Astral 博客页面！")
           print(f"状态码: {response.status_code}")
       except requests.exceptions.RequestException as e:
           print(f"请求失败: {e}")
   
   if __name__ == "__main__":
       get_astral_blog()
   ```

   运行代码：

   ```bash
   python main.py
   ```