---
title: Ruff库
author: Alen
published: 2025-10-10
description: "代码美化工具：Ruff库的介绍"
first_level_category: "python"
second_level_category: "开发工具"
tags: ['python']
draft: false
---

# Ruff

## 介绍

Ruff 是一个“代码美化大师”，一个“能抵千军万马的工具”。它是一个用 Rust 语言编写的、速度极快的 **Python Linter（代码检查器）** 和 **Code Formatter（代码格式化器）**。

它的核心理念是整合多个经典工具的功能于一身，所以才说 **Ruff = Black + Flake8 + isort + pyupgrade**。

- **Flake8**:     用于检查代码风格错误（比如变量未使用、行太长等）。
- **Black**:     用于自动格式化代码，统一代码风格。
- **isort**:     用于自动排序 import 语句。
- **pyupgrade**:     用于自动将代码语法升级到更新的 Python 版本。

Ruff 把以上所有功能都实现了，并且速度比它们快 10-100 倍。

---

## 使用方法

使用 Ruff 非常简单，主要分为以下几个步骤：

---

### 安装

首先，通过 pip 安装 Ruff。

```
pip install ruff
```

---

### 基本使用（命令行）

Ruff 的主要功能通过两个命令实现：ruff check 和 ruff format。

**A. 代码检查 (Linting - 替代 Flake8, isort)**

这个命令会检查你的代码是否存在风格问题、潜在错误、import 顺序问题等。

- **检查单个文件：**

  ```bash
  ruff check your_file.py
  ```

  它会打印出文件中所有的问题，类似 Flake8。

- **检查整个项目（当前目录及所有子目录）：**

  ```bash
  ruff check .
  ```

**B. 自动修复 (Fixing)**

这是 Ruff 最强大的功能之一。它可以自动修复很多检查出来的问题（比如自动排序 import、删除未使用的变量、修复一些简单的语法问题）。

- **检查并尝试自动修复所有问题：**

  ```bash
  ruff check . --fix
  ```

  运行这个命令后，Ruff 会直接修改你的文件，修复它能修复的所有问题。

**C. 代码格式化 (Formatting - 替代 Black)**

这个命令会根据统一的风格（类似 Black）重新格式化你的代码。

- **格式化单个文件：**

  ```
  ruff format your_file.py
  ```

- **格式化整个项目：**

  ```
  ruff format .
  ```

**D. 实时监控（Watch Mode）**

在你写代码的时候，让 Ruff 在后台实时监控文件变化并自动检查和修复，非常方便。

```
ruff check . --watch
```

### 配置文件 (pyproject.toml)

为了让整个团队保持一致的规则，或者自定义 Ruff 的行为，你需要在你的项目根目录下创建一个 pyproject.toml 文件来配置它。

这是一个推荐的配置示例：

```toml
# 文件名: pyproject.toml

[tool.ruff]
# 设置每行的最大长度，和 Black 默认值一样
line-length = 88
# 指定你的项目兼容的 Python 版本
target-version = "py310"

[tool.ruff.lint]
# 选择要启用的规则集。
# E: pycodestyle (错误)
# W: pycodestyle (警告)
# F: Pyflakes (逻辑错误，如未使用变量)
# I: isort (import 排序)
# UP: pyupgrade (语法升级)
select = ["E", "W", "F", "I", "UP"]

# 忽略特定的规则。例如，E501 是“行太长”，
# 如果你设置了 line-length，通常不需要忽略它，这里只是示例。
# ignore = ["E501"]

[tool.ruff.format]
# 格式化器的配置。Ruff 的格式化器和 Black 一样，可配置项很少。
# 例如，可以指定字符串引号风格。
# quote-style = "double"  # 使用双引号
# quote-style = "single"  # 使用单引号
```

**有了这个配置文件后，你再运行 ruff check . 或 ruff format .，Ruff 就会自动读取这些规则。**

### 集成到 IDE

这才是提升效率的关键！让 Ruff 在你保存文件时自动工作。

#### 在 VS Code 中使用

1. 在 VS Code 的扩展商店中搜索并安装 **“Ruff”** 扩展（由 Astral 公司发布）。

2. 安装后，打开你的 VS Code 设置 (settings.json)，添加以下配置：

   ```json
   {
     "[python]": {
       "editor.defaultFormatter": "charliermarsh.ruff", // 将 Ruff 设置为默认格式化器
       "editor.formatOnSave": true, // 开启保存时自动格式化
       "editor.codeActionsOnSave": {
         "source.fixAll": "explicit", // 开启保存时自动修复
         "source.organizeImports": "explicit" // (可选) Ruff 已经处理了 import 排序
       }
     },
     "ruff.lint.args": [],
     "ruff.path": ["/path/to/your/ruff/executable"] // 通常不需要，如果 Ruff 在你的 PATH 中
   }
   ```

   配置好后，你在保存 Python 文件时，VS Code 就会自动调用 Ruff 进行格式化和修复。

#### 在 PyCharm 中使用

PyCharm 的集成需要通过 "File Watcher" 功能。

1. **安装 Ruff**：确保你已在项目的 Python 解释器中安装了 ruff。
2. **设置 File Watcher**：
   - 打开 Settings/Preferences -> Tools -> File Watchers。
   - 点击 + 号，选择` <custom>`。
   - **Name:** Ruff Format
   - **File type:** Python
   - **Scope:** Project Files
   - **Program:** 填写 ruff 的路径。如果你的虚拟环境已激活，通常可以直接写 ruff。或者使用宏 $PyInterpreterDirectory/ruff $来确保使用的是项目解释器中的 Ruff。
   - **Arguments:** format $FilePath$
   - **Output paths to refresh:** $FilePath$
   - 点击 **OK**。
3. 现在，当你修改并保存一个 Python 文件时，PyCharm 就会自动运行 ruff format 来格式化它。

**如果想在保存时自动修复 (check --fix)，可以再创建一个 File Watcher：**

- **Name:** Ruff Fix
- **Program:** ruff
- **Arguments:** check --fix --exit-zero $FilePath$
  - --exit-zero 是为了防止 PyCharm 在 Ruff 发现无法修复的错误时弹出报错窗口。

---

## 总结

1. **安装:** pip install ruff
2. **配置:** 在 pyproject.toml 中设置规则。
3. **使用:**
   - **检查:** ruff check .
   - **修复:** ruff check . --fix
   - **格式化:** ruff format .
4. **集成:** 在 VS Code 或 PyCharm 中设置保存时自动运行，彻底解放双手。
