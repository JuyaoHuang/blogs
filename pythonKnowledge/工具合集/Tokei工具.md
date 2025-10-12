---
title: tokei
author: Alen
published: 2025-10-10
description: "代码统计工具：Tokei的介绍"
first_level_category: "python"
second_level_category: "开发工具"
tags: ['python']
draft: false
---

# Tokei

## 介绍

**Tokei 是一个用 Rust 编写的、独立运行的命令行工具 (Command-Line Tool)。**

它是目前公认的用于代码统计的**最快、最强大**的工具之一，是无数开发者的心头好。

**Tokei** (日语“時計”，意为“时钟”，引申为“统计”) 是一个**代码统计程序**。它能飞快地扫描你的项目文件夹，识别出其中包含的各种编程语言，并精确地统计出每种语言的**文件数、总行数、代码行、注释行和空白行**。

可以看作是一个超级加强版的 cloc (Count Lines of Code)。

## 核心优势

1. **极致的速度 (Blazingly Fast)**
   - 由于它是用 Rust 编写的，并充分利用了多核处理器进行并行处理，Tokei 的扫描速度快得惊人。对于大型项目，它可能比其他同类工具（如 cloc）快 10 倍甚至 100 倍。
2. **超广的语言支持**
   - Tokei 能识别超过 150 种编程语言和文件格式，并且这个列表还在不断更新。无论是常见的前后端语言，还是配置文件、文档标记语言，它都能精确识别。
3. **精确的统计**
   - 它不仅仅是简单地数行数。Tokei 内置了对各种语言**注释语法**（单行注释、多行注释、嵌套注释等）的精确解析。因此，它能非常准确地区分出哪些是真正的**代码**，哪些是**注释**，哪些是**空白行**。
4. **多种输出格式**
   - 除了默认在命令行打印出漂亮的表格外，Tokei 还支持将统计结果输出为 **JSON**, **YAML**, CBOR 等多种机器可读的格式。这使得将 Tokei 集成到自动化脚本、CI/CD 流程或项目报告中变得非常容易。
5. **跨平台**
   - 作为一个独立的二进制文件，它可以在 Windows, macOS, Linux 上完美运行。

---

## 使用方法

### 安装

**macOS (使用 Homebrew)**

```bash
brew install tokei
```

**Windows (使用 Scoop 或 Chocolatey)**

```
# Scoop
scoop install tokei

# Chocolatey
choco install tokei
```

**通用 (使用 Rust 的包管理器 Cargo)**
如果你安装了 Rust 环境，这是最直接的方式。

```
cargo install tokei
```

**直接下载**
你也可以直接从 Tokei 的 [GitHub Releases 页面](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2FXAMPPRocky%2Ftokei%2Freleases) 下载对应你操作系统的、已经编译好的二进制文件，然后把它放到你的系统路径下即可。

### 基本用法

用法极其简单。打开你的终端，cd 进入到你的项目根目录，然后执行：

```
tokei .
```

（. 代表当前目录）

Tokei 会立刻扫描当前目录（并递归扫描所有子目录），然后输出一份漂亮的报告。

### 理解输出结果

执行后，你会看到类似这样的表格：

```bash
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 BASH                    2           63           47           10            6
 JSON                    5          734          734            0            0
 Markdown                3          420          310           80           30
 Python                 25         2800         2100          350          350
 TOML                    1           85           71            5            9
-------------------------------------------------------------------------------
 HTML                    2         1200         1050           50          100
 (Inaccurate)
-------------------------------------------------------------------------------
 Total                  38         5302         4312          495          495
===============================================================================
```

**列的含义**:

- **Language**: 识别出的语言。
- **Files**: 该语言对应的文件数量。
- **Lines**: 该语言所有文件的物理总行数。
- **Code**: **真正的代码行数** (这是最有价值的指标)。
- **Comments**: 注释行数。
- **Blanks**: 空白行数。

### 常用进阶参数

- **排除目录**: 这是最常用的功能之一！比如你不想统计 node_modules 或虚拟环境 venv。

  ```bash
  tokei . --exclude venv,node_modules,target
  ```

- **排序**: 让输出结果按某一列排序。

  ```bash
  # 按代码行数（Code）降序排列
  tokei . --sort code
  ```

- **更改输出格式**: 输出为 JSON，方便脚本处理。

  ```bash
  tokei . --output json > stats.json
  ```

### python中使用

虽然 Tokei 不是一个 Python 库，但可以用 Python 的 subprocess 模块来调用它，并捕获其输出。

这是一个非常常见的模式：

```python
import subprocess
import json

def get_code_stats(path: str) -> dict:
    """
    使用 tokei 工具来统计指定路径的代码信息。
    
    :param path: 要统计的项目文件夹路径。
    :return: 一个包含统计信息的字典。
    """
    try:
        # 构建 tokei 命令，排除常用目录并要求 JSON 输出
        command = [
            "tokei",
            path,
            "--exclude", "venv,node_modules,__pycache__",
            "--output", "json"
        ]
        
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # 将捕获到的 JSON 字符串解析成 Python 字典
        stats_data = json.loads(result.stdout)
        return stats_data
        
    except FileNotFoundError:
        print("错误: 'tokei' 命令未找到。请确保已经安装并将其添加到了系统 PATH。")
        return None
    except subprocess.CalledProcessError as e:
        print(f"执行 tokei 时出错: {e.stderr}")
        return None

if __name__ == "__main__":
    # 假设你的项目在当前目录下的 'my_project' 文件夹
    project_path = "." 
    stats = get_code_stats(project_path)
    
    if stats:
        # 你现在可以像操作普通字典一样操作统计结果了！
        python_stats = stats.get("Python")
        if python_stats:
            print(f"Python 统计:")
            print(f"  - 文件数: {python_stats['stats']['files']}")
            print(f"  - 代码行: {python_stats['stats']['code']}")
            print(f"  - 注释行: {python_stats['stats']['comments']}")
```

---

## 总结: 

Tokei 是一个“小而美”的典范工具。它专注于一件事——代码统计，并将其做到了极致。对于任何需要快速了解项目代码规模、评估代码健康度或生成项目报告的开发者来说，它都是一个必备利器。