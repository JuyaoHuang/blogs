---
title: python语法-模块
author: Alen
published: 2025-10-10
description: "python合集第七集：模块-包语法"
first_level_category: "Python"
second_level_category: "基础语法"
tags: ['python']
draft: false
---

# 模块与包

## 模块的介绍

#### 1. 什么是模块？

**一个模块就是一个 Python 文件**。任何一个以 .py 结尾的 Python 文件都可以被看作是一个模块。

模块的主要作用是：

- **代码组织**：将相关的代码（函数、类、变量）组织在一个文件中，使项目结构更清晰。
- **代码重用**：你可以在不同的程序中导入并使用同一个模块，避免重复编写代码。
- **命名空间**：模块创建了一个独立的命名空间，可以避免不同模块间的命名冲突。

例如，Python 内置的 math 模块就是一个名为 math.py 的文件，里面定义了 pi、sqrt() 等数学相关的变量和函数。

也就是说，**模块就是一个py文件，里面有类、函数、变量等，我们可以拿过来用**

#### 2. 如何使用模块 (import)

使用 import 语句来引入一个模块。有几种常见的导入方式：

**方式一：import module_name (推荐)**

这是最常用和推荐的方式。它导入整个模块，并通过 module_name.member_name 的方式访问模块内的成员。

```python
import math

print(math.pi)              # 输出: 3.141592653589793
print(math.sqrt(16))        # 输出: 4.0

import os # 导入操作系统接口模块
print(os.getcwd())          # 输出: 当前工作目录
```

**方式二：import module_name as alias**

给导入的模块起一个别名，通常是为了简化较长的模块名。

```py
import math as m

print(m.pi)
print(m.sqrt(16))

import pandas as pd # 这是一个非常普遍的约定
```

**方式三：from 模块名 import 功能名**

从模块中只导入特定的成员（函数、类或变量），这样就可以直接使用该成员，而不需要模块名前缀。

```python
from math import pi, sqrt

print(pi)              # 直接使用 pi，不需要 math.pi
print(sqrt(16))        # 直接使用 sqrt
```

**方式四：from 模块名 import * **

将模块里的所有方法（*）全部导入进来用。

不推荐使用：因为它会污染你的当前命名空间，你可能不知道导入了哪些变量和函数，如果恰好有同名的函数或变量，就会被覆盖，导致难以追踪的 bug。

### 自定义模块

**步骤 1：创建一个 Python 文件**

假设创建一个名为 my_utils.py 的文件，内容如下：

```python
# 文件名: my_utils.py

PI = 3.14159

def greet(name):
    """一个简单的问候函数"""
    return f"Hello, {name}! Welcome."

def calculate_area(radius):
    """计算圆的面积"""
    return PI * (radius ** 2)
```

**步骤 2：在另一个文件中导入并使用它**

在**同一个目录**下，创建另一个文件，例如 main.py。

```py
# 文件名: main.py

# 导入我们自己创建的 my_utils 模块
import my_utils

# 使用模块中的函数和变量
message = my_utils.greet("Alice")
area = my_utils.calculate_area(10)

print(message)
print(f"半径为 10 的圆的面积是: {area}")
print(f"模块中的 PI 值是: {my_utils.PI}")
```

**运行 main.py，得到输出：**

```py
Hello, Alice! Welcome.
半径为 10 的圆的面积是: 314.159
模块中的 PI 值是: 3.14159
```

**重要概念：if __name__ == '__main__':**

这是一个非常重要的技巧。放在这个代码块下的代码，只有当该 .py 文件被直接运行时才会执行。如果它作为模块被导入，这个代码块下的内容则不会执行。这常用于为模块编写测试代码。

修改 my_utils.py：

```py
# 文件名: my_utils.py

PI = 3.14159

def greet(name):
    return f"Hello, {name}! Welcome."

def calculate_area(radius):
    return PI * (radius ** 2)

# --- 添加测试代码 ---
if __name__ == '__main__':
    # 这部分代码只有在直接运行 "python my_utils.py" 时才会执行
    print("正在运行 my_utils.py 的测试代码...")
    test_name = "Bob"
    test_radius = 5
    print(greet(test_name))
    print(f"半径为 {test_radius} 的圆面积是: {calculate_area(test_radius)}")
```

- 运行 python main.py 时，my_utils.py 是被导入的，所以它的测试代码不会执行。
- 运行 python my_utils.py 时，它的测试代码会被执行。

## Python包

#### 1. 什么是包package？

当项目越来越大，模块文件越来越多时，你就需要一种更好的方式来组织它们。**包就是一种组织模块的方式。**

简单来说，**一个包就是一个包含多个模块的目录（文件夹）**。这个目录必须包含一个特殊的文件` __init__.py `(在现代 Python 中，这个文件可以是空的)。如果没有，python解释器就不会将此文件夹认为是一个包，会当作普通的文件夹处理。

**目录结构示例：**

```bash
my_project/
├── main.py
└── my_app/              <-- 这是一个包
    ├── __init__.py      <-- 必须有这个文件，可以是空的
    ├── string_ops.py    <-- 模块1
    └── math_ops.py      <-- 模块2
```

#### 2. `__init__.py` 的作用

- **标识包：** 它的存在告诉 Python，这个目录应该被当作一个包来处理。
- **初始化代码 (可选):** 你可以在` __init__.py` 文件中编写代码。当包被导入时，这个文件会自动执行。这可以用来进行一些包级别的初始化设置。

#### 3. 创建和使用自定义包

**步骤 1：创建目录和文件**

按照上面的结构创建目录和文件。

- **my_app/string_ops.py 内容：**

- ```py
  def reverse_string(s):
      return s[::-1]
  ```

- **my_app/math_ops.py 内容**

  ```py
  def add(a, b):
      return a + b
  ```

- **my_app/`__init__.py` 内容：**

  ```py
  # 这个文件可以是空的，或者我们可以加一句打印来观察它何时执行
  print("my_app 包正在被初始化...")
  ```

**步骤 2：在 main.py 中导入和使用包中的模块**

```py
# 文件名: main.py

from rich import print
from my_package import string_ops
from my_package.math_ops import add

reverse_text = string_ops.reverse_string("Alen is pig!")
res = add(4, 4)
print(reverse_text)
print(res)
```

**运行 main.py，你将得到输出：**

```
my_app 包正在被初始化...
!gip si nelA
8
```

可以看到，当 main.py 第一次从 my_app 包中导入任何东西时，`__init__.py` 文件就被执行了。
