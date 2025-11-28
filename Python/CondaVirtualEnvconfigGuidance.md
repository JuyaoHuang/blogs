---
title: "Conda环境配置指南"
published: 2025-10-01
tags: ['conda']
first_level_category: "Web全栈开发"
second_level_category: "运维与Linux"
author: "Alen"
draft: false
---

# Conda 虚拟环境指南

​	虚拟环境是数据科学和软件开发中的一个重要概念，它允许你在隔离的环境中管理项目依赖项。这意味着不同项目可以使用不同版本的库，而不会相互冲突。Conda 是一个开源的包管理系统和环境管理系统，它不仅可以管理 Python 包，还可以管理其他语言的包。

## 为什么使用虚拟环境？

- **隔离性：** 每个项目都有自己独立的依赖项集合，避免不同项目之间库版本冲突
- **可重复性：** 你可以轻松地共享环境配置，确保其他人在其机器上也能复现你的开发环境
- **干净的工作区**： 保持你的系统 Python 环境清洁，只安装你需要全局使用的工具

## 0. 前提条件

​	你需要安装 Miniconda 或 Anaconda。如果你还没有安装，请访问 [Conda 官方网站](https://www.google.com/url?sa=E&q=https%3A%2F%2Fdocs.conda.io%2Fen%2Flatest%2Fminiconda.html) 下载并安装适合你操作系统的版本

## 1. 创建虚拟环境

使用 conda create 命令创建新的虚拟环境。你可以指定 Python 版本和需要预安装的包。

**基本语法：**

```
conda create -n <环境名称> python=<版本号> [包1] [包2] ...
```

- -n 或 --name：指定环境的名称。
- python=<版本号>：指定该环境使用的 Python 版本，例如 python=3.9。

**示例：**

1. **创建一个名为 myenv 的环境，并指定 Python 3.9：**

   ```bash
   conda create -n myenv python=3.9
   ```

   执行此命令后，Conda 会列出将要安装的包，并询问你是否继续（Proceed ([y]/n)?），输入 y 并回车即可

2. **创建一个名为 new_env 的环境，并安装 Python 3.10 和 PyTorch：**

   ```bash
   conda create -n new_env python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch
   ```
   
- -c pytorch：    指定从 PyTorch 官方 channel 安装包，通常用于获取最新的或特定的深度学习库版本

## 2. 激活虚拟环境

使用虚拟环境之前，必须先激活它。激活后，你在该终端会话中运行的所有命令都将在该环境中执行。

**语法：**

```bash
conda activate <环境名称>
```

**示例：**

```bash
conda activate myenv
```

**激活成功标志：**
激活后，终端提示符前会显示环境名称，例如：

```bash
(myenv) C:\Users\YourUser>
```

或者在 Git Bash 中：

```
(myenv) Alen@Alen MINGW64 /d/Coding/GitHub_Resuorse (main)  
```

## 3. 停用虚拟环境

当你完成在虚拟环境中的工作后，可以使用 conda deactivate 命令退出当前环境

**语法：**

```bash
conda deactivate  
```

**示例：**

```bash
(myenv) C:\Users\YourUser>conda deactivate
C:\Users\YourUser>  
```

## 4. 管理虚拟环境中的包

### 4.1 安装包

在激活的环境中，可以使用 conda install 或 pip install 安装包

**使用 conda install (Conda 会处理依赖关系)：**

```bash
(myenv) conda install numpy pandas scikit-learn
```

**使用 pip install (如果 Conda 找不到特定包，或需要安装 PyPI 上的包)：**

```bash
 (myenv) pip install some-package-from-pypi 
```

### 4.2 列出已安装的包

```bash
(myenv) conda list  
```

这将列出当前激活环境中所有已安装的包及其版本。

### 4.3 更新包

```bash
(myenv) conda update <包名称>
```

**示例：**

```bash
(myenv) conda update numpy
```

### 4.4 删除包

```bash
(myenv) conda remove <包名称> 
```

**示例：**

```bash
(myenv) conda remove pandas 
```

## 5. 查看所有虚拟环境

查看创建的所有 Conda 虚拟环境

**语法：**

```bash
conda env list
# 或者
conda info --envs   
```

**示例输出：**

```bash
# conda environments:
#
base                     C:\Users\YourUser\miniconda3
myenv                 *  C:\Users\YourUser\miniconda3\envs\myenv
llm_v_tuber_env          C:\Users\YourUser\miniconda3\envs\llm_v_tuber_env   
```

带 * 号的表示当前激活的环境。

## 6. 删除虚拟环境

如果你不再需要某个虚拟环境，可以将其删除

**语法：**

```bash
conda env remove -n <环境名称>
# 或者
conda remove --name <环境名称> --all
```

**示例：**

```bash
conda env remove -n myenv
```

Conda 会询问你是否继续，输入 y 并回车确认。

## 7. 导出和导入环境（环境复现）

为了让其他人或自己在其他机器上复现相同的环境，你可以将环境配置导出为 YAML 文件，然后从该文件创建环境。

### 7.1 导出环境

在激活你想要导出的环境后，将其导出为 environment.yml （名字可自定，但一般都这么写）件

**语法：**

```bash
(myenv) conda env export > environment.yml
```

这将生成一个名为 environment.yml 的文件，其中包含了该环境的 Python 版本、所有 Conda 和 Pip 包及其版本、以及 Conda channels 信息。

**注意**：这只适合同系统下的环境迁移。如果是跨系统迁移环境，例如从 Windows 迁移到 Linux，应该使用此指令：

```bash
(myenv) conda env export --from-history > environment.yml
```

这会只保存使用 conda 指令 `conda install`下的包，并且不指定特定版本，适合环境迁移。

> 如果本身conda 环境被污染了，例如本身创建的 conda 环境（设为 B ）使用的就是用原始环境 A 使用指令 `conda env export > environment.yml` 导出的包依赖，那么创建环境是会 **继承**此依赖关系。

### 7.2 从文件创建/导入环境

需要复现一个环境时，可以使用导出的 environment.yml 文件来创建它

#### 7.2.1 创建新环境

1. 使用 environment.yml 文件的 name：字段定义新的环境名称：

   ```bash
   conda env create -f environment.yml    
   ```

   Conda 会读取 environment.yml 文件，并根据文件中的配置创建一个新的环境。**新环境的名称**将由 environment.yml 文件中的 name: 字段定义

2. 自定义新的环境名称： 

   ```bash
   conda env create -f path/to/your/environment_to_import.yml -n llmvtuber
   ```

   - 将 `path/to/your/environment_to_import.yml` 替换为 `.yml` 文件的实际路径
   - `-n llmvtuber` 参数指定新创建的环境名称为 llmvtuber
   - **如果 YAML 文件内部通过 name: 字段定义了环境名**，并且你想使用 YAML 文件中定义的名字，可以省略 -n llmvtuber 参数

3. 激活新环境:

   ```bash
   conda activate <environment_name_from_yml_or_-n>
   ```

#### 7.2.2 环境已存在，使用 yaml文件更新环境配置

​	如果你已经有一个名为 llmvtuber 的环境，并且希望用 YAML 文件中的配置来更新它（可能会覆盖或添加包），请执行以下操作：

1. 激活 llmvtuber 环境：

   ```bash
   conda activate llmvtuber
   ```

2. 使用 conda env update 命令进行更新：

   ```bash
   conda env update --name llmvtuber --file path/to/your/environment_to_import.yml --prune
   ```

   - --name llmvtuber:   明确指定要更新的目标环境
   - --file path/to/your/environment_to_import.yml:   指定用于更新的 YAML 文件
   - --prune:    该选项会删除当前 llmvtuber 环境中存在、但**不包含**在所提供的 YAML 文件中的那些包。这有助于确保目标环境与 YAML 文件定义的配置完全一致。如果你不希望删除多余的包，可以省略此选项，Conda 将只添加或更新 YAML 文件中列出的包

3. 等待环境配置完成

4. 激活并验证环境

5. 重要注意事项

   1. **YAML 文件内容**： 

      ​	确保使用的 `.yml` 文件来源可靠，并且其内容确实是你想要复制的环境配置。可以先用文本编辑器打开 YAML 文件查看其 name, channels, 和 dependencies 部分。

   2. **Conda 通道**： 

      ​	YAML 文件可能指定了特定的 Conda 下载通道。如果你的 Conda 配置中缺少这些通道，或者网络无法访问，安装某些包时可能会失败。Conda 通常会自动尝试添加 YAML 中声明的通道

   3. **操作系统和架构兼容性**： 

      ​	从不同操作系统或 CPU 架构生成的 YAML 文件在导入时可能会遇到兼容性问题，尤其是对于包含已编译代码的库。Conda 会尽力解决，但有时可能需要手动调整

   4. **Python 版本**：YAML 文件通常会指定一个 Python 版本，Conda 会尝试安装此版本的 Python 到新环境中。

   5. **覆盖风险 (使用 conda env update --prune 时)**：

      ​	当使用 --prune 选项更新现有环境时，请务必确认你希望用 YAML 文件的内容完全替换当前环境的配置，因为不包含在 YAML 中的包将被移除。

## 常用指令总结

|                 指令                  |                作用                |               示例               |
| :-----------------------------------: | :--------------------------------: | :------------------------------: |
| `conda create -n <name> python=<ver>` | 创建新的虚拟环境，指定 Python 版本 | conda create -n myenv python=3.9 |
|        `conda activate <name>`        |          激活指定虚拟环境          |       conda activate myenv       |
|          `conda deactivate`           |          停用当前虚拟环境          |         conda deactivate         |
|       `conda install <package>`       |       在当前激活环境中安装包       |       conda install numpy        |
|        `pip install <package>`        |  在当前激活环境中安装 PyPI 上的包  |       pip install requests       |
|             `conda list`              |  列出当前激活环境中所有已安装的包  |            conda list            |
|       `conda update <package>`        |       更新当前激活环境中的包       |       conda update pandas        |
|       `conda remove <package>`        |       删除当前激活环境中的包       |        conda remove scipy        |
| `conda env list / conda info --envs`  |      列出所有 Conda 虚拟环境       |          conda env list          |
|     `conda env remove -n <name>`      |         删除指定的虚拟环境         |    conda env remove -n myenv     |
| `conda env export > environment.yml`  | 导出当前激活环境的配置到 YAML 文件 |    conda env export > env.yml    |
| `conda env create -f environment.yml` |        从 YAML 文件创建环境        |   conda env create -f env.yml    |
