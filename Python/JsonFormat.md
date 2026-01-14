---
title: json数据格式
publishDate: 2025-10-10
description: "json数据格式介绍"
tags: ['python']
language: 'Chinese'
first_level_category: "Python"
second_level_category: "基础语法"
draft: false
---

# Json

## 定义

**JSON (JavaScript Object Notation)** 是一种轻量级的、人类易于阅读和编写、同时也易于机器解析和生成的数据交换格式。

尽管它的名字来源于 JavaScript，但 JSON 是完全独立于语言的，几乎所有现代编程语言都有解析和生成 JSON 数据的库。可以把它想象成一种 **通用数据语言**，Python 可以用它和C++、 Java、JavaScript、Go 等任何其他语言进行无障碍的数据交流。本质上是一个**特定格式的字符串**

### JSON 的基本结构和规则

JSON 有两种结构：

1. **对象 (Object):** 一系列 **键/值对 (key/value pairs)** 的无序集合。-------就是**python的字典**
   - 一个对象以 {（左花括号）开始，以 }（右花括号）结束。
   - 每个“键”都是一个**字符串**，并且必须用**双引号 "** 括起来。
   - 键和值之间用 :（冒号）分隔。
   - 键/值对之间用 ,（逗号）分隔。
   - 这在 Python 中几乎完全对应于**字典 **。
2. **数组 (Array):** 一系列值的有序集合。-----------就是 **python的列表**
   - 一个数组以 [（左方括号）开始，以 ]（右方括号）结束。
   - 值之间用 ,（逗号）分隔。
   - 这在 Python 中完全对应于**列表 **。

### JSON 支持的数据类型

JSON 的“值”可以是以下几种类型之一：

- **字符串:**     必须用双引号 " 括起来。例如 "hello world"。
- **数字:**     整数或浮点数，不需要引号。例如 101 或 3.14。
- **对象 :**     可以是另一个 JSON 对象，实现数据嵌套。
- **数组:**     可以是一个 JSON 数组。
- **布尔值:**     必须是 true 或 false（注意是小写）。
- **空值:**     必须是 null（注意是小写）。

### 完整示例

这个例子展示了一个用户对象，其中包含了各种数据类型：

```json
{
  "userId": 1001,
  "username": "coder_alex",
  "isActive": true,
  "profile": {
    "realName": "Alex Smith",
    "avatarUrl": "/images/alex.png"
  },
  "tags": [
    "Python",
    "API",
    "Web Development"
  ],
  "lastLogin": null
}
```

## Python 中的 JSON 格式转换

Python 内置了一个强大的 json 模块，可以轻松地在 Python 对象和 JSON 格式的字符串之间进行转换。							

```python
import json
```

这个过程主要涉及两对核心操作：

1. **序列化 (Serialization):**      将 Python 对象转换为 JSON 格式的字符串。
   - json.dumps(): 将 Python 对象转换成**字符串 (string)**。
   - json.dump(): 将 Python 对象写入**文件对象 (file object)**。
2. **反序列化 (Deserialization):**     将 JSON 格式的字符串转换为 Python 对象。
   - json.loads(): 从**字符串 (string)** 中加载数据，转换成 Python 对象。
   - json.loads(): 从**文件对象 (file object)** 中加载数据，转换成 Python 对象。

**记忆技巧：dump / load 后面带 s 的 (dumps, loads) 都是处理字符串 (string) 的。**

### 将 Python 字典转换为 JSON 字符串 (json.dumps)

这通常用于将数据准备好以便通过网络发送（例如，在 API 响应中）。

```py
import json

# 一个 Python 字典
python_dict = {
    "name": "Alice",
    "age": 30,
    "is_student": False,
    "courses": ["Math", "Physics"],
    "address": None
}

# indent=4 表示使用4个空格进行缩进
json_string = json.dumps(python_dict, indent=4)
print(json_string)

# 处理中文字符
chinese_dict = {"姓名": "张三"}
# 默认情况下，非 ASCII 字符会被转义
escaped_json = json.dumps(chinese_dict, indent=4)
print("\n--- 中文字符（默认转义） ---")
print(escaped_json)

# 使用 ensure_ascii=False 来正确显示中文字符
correct_chinese_json = json.dumps(chinese_dict, indent=4, ensure_ascii=False)
print("\n--- 中文字符（ensure_ascii=False） ---")
print(correct_chinese_json)

```

### 将 JSON 字符串转换为 Python 字典(json.loads)

​	

json.loads() 的作用就是接收一个**字符串**作为输入，然后按照 **JSON 语法规则**去解析它。当你把 {...} 整体放进一个字符串里时，true 和 null 就只是字符串里的文本，json.loads() 会正确地将它们识别并转换为 Python 对应的 True 和 None。

这通常用于解析从 API 或文件中读取的 JSON 字符串数据。

```py
import json

# 一个 JSON 列表格式的字符串
json_data_string = """
[
    {
        "name": "Alice",
        "age": 30,
        "is_student": false
    },
    {
        "name": "Bob",
        "age": 25,
        "is_student": true
    },
    {
        "name": "Charlie",
        "age": 35,
        "is_student": false
    }
]
"""

# 将 JSON 字符串解析为 Python 列表
ls = json.loads(json_data_string)

print("--- 解析后的 Python 对象 ---")
print(ls)
print(type(ls))
print(ls[0])
print(type(ls[0]))

# --- 解析后的 Python 对象 ---
# [
#     {'name': 'Alice', 'age': 30, 'is_student': False},
#     {'name': 'Bob', 'age': 25, 'is_student': True},
#     {'name': 'Charlie', 'age': 35, 'is_student': False}
# ]
# <class 'list'>
# {'name': 'Alice', 'age': 30, 'is_student': False}
# <class 'dict'>
```

### JSON 与 Python 的数据类型映射关系

在转换过程中，数据类型会按照下表进行映射：

也就是说，json在py里面是一个字符串，这个字符串的内容、格式是json格式

| JSON          | Python |
| ------------- | ------ |
| object        | dict   |
| array         | list   |
| string        | str    |
| number (int)  | int    |
| number (real) | float  |
| true          | True   |
| false         | False  |
| null          | None   |

### 将 Python 字典写入 JSON 文件 (json.dump)



直接将 Python 字典保存到文件中，非常方便。

```py
import json

user_data = {
    "id": 2024,
    "user": "Bob",
    "permissions": ["read", "write"]
}

file_path = "user_data.json"

with open(file_path, 'w', encoding='utf-8') as f:
    # indent=4 使得文件内容格式化，易于阅读
    # ensure_ascii=False 确保非英文字符正确写入
    json.dump(user_data, f, indent=4, ensure_ascii=False)

print(f"数据已成功写入到 '{file_path}' 文件中。")
```

### 从 JSON 文件中读取数据 (json.load)



直接从文件中读取 JSON 数据并解析为 Python 字典。

```py
from rich import print
import json

file_path = "user_data.json"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    print("--- 从 JSON 文件中读取的数据 ---")
    print(loaded_data)
    print("类型:", type(loaded_data))
    print("用户名:", loaded_data["user"])

except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 不存在。")
# {'id': 2024, 'user': '艾伦', 'permissions': ['read', 'write']}
# 类型: <class 'dict'>
# 用户名: 艾伦
```

**总结：**

*   **JSON** 是一种通用的、人类可读的数据格式，核心是**字典(字典)** 和**数组 (列表)**。
*   Python 的 **`json`** 模块是处理 JSON 的标准工具。
*   处理**字符串**用 **`dumps()`** (Python -> JSON) 和 **`loads()`** (JSON -> Python)。
*   处理**文件**用 **`dump()`** (Python -> 文件) 和 **`load()`** (文件 -> Python)。
*   处理中文或非 ASCII 字符时，记得在 `dump()` / `dumps()` 中设置 **`ensure_ascii=False`**。
*   为了文件美观，设置 **`indent=4`**。