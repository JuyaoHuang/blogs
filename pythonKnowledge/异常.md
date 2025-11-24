---
title: 异常处理
author: Alen
published: 2025-10-10
description: "python合集第六集：异常处理"
first_level_category: "Python"
second_level_category: "基础语法"
tags: ['python']
draft: false
---

# 异常

## 什么是异常？

**异常 (Exception)** 是程序在执行期间发生的错误事件。当 Python 脚本遇到一个它无法处理的情况时（比如用 0 作除数、打开一个不存在的文件），它会“引发”或“抛出”一个异常。

如果不进行处理，这个异常会导致程序立即停止执行并显示一条错误信息（Traceback）。

### 1. 基本的异常捕获：try...except

try...except 语句块是异常处理的核心。

- **try 块**：将可能会引发异常的代码放在这个代码块中。
- **except 块**：如果 try 块中的代码**确实**引发了异常，程序会立即跳转到 except 块中执行。如果 try 块中没有异常发生，except 块将被跳过。

**语法：**

```python
try:
    # 可能会出错的代码
    risky_code()
except:
    # 如果出错，就执行这里的代码
    handle_the_error()
```

**示例：**

```python
try:
    result = 10 / 0
    print("计算结果是:", result) # 这行不会执行
except:
    print("出错了！除数不能为零。")

print("程序继续执行...") # 这行会执行

# 输出:
# 出错了！除数不能为零。
# 程序继续执行...
```

#### 捕获特定类型的异常

一个“裸露”的 except: 会捕获所有类型的异常，这通常不是一个好主意，因为它可能会隐藏你没有预料到的错误。更好的做法是捕获**特定类型**的异常。

常见的异常类型包括：

- ZeroDivisionError
- FileNotFoundError
- ValueError
- TypeError
- IndexError
- KeyError 

**示例：**

```python
try:
    num_str = "abc"
    num = int(num_str) # 这会引发 ValueError
except ValueError:
    print(f"无法将 '{num_str}' 转换为整数。")
```

#### 捕获多种异常

使用一个元组来捕获多种不同类型的异常，或者使用多个 except 块。

**方法一：使用元组**

```py
try:
    # 这里的代码可能引发 ValueError 或 ZeroDivisionError
    x = int(input("请输入一个数字: "))
    result = 100 / x
    print(f"结果是: {result}")
except (ValueError, ZeroDivisionError):
    print("输入无效或除数为零，请重试。")
```

**方法二：使用多个 except 块**

当你想为不同类型的异常提供不同的处理逻辑时，这个方法非常有用。

```py
try:
    x = int(input("请输入一个数字: "))
    result = 100 / x
    print(f"结果是: {result}")
except ValueError:
    print("输入错误，请输入一个有效的整数。")
except ZeroDivisionError:
    print("除数不能为零！")
except Exception as e: # 捕获其他所有未预料到的异常
    print(f"发生了一个未知错误: {e}")
```

**except Exception as e**: 这是一个很好的实践。Exception 是几乎所有常见异常的基类。as e 会将异常对象本身赋值给变量 e，这样你就可以打印出具体的错误信息。

### 2. try...except...else 结构

else 块是可选的，它会在 try 块**没有发生任何异常**的情况下执行。

**为什么使用 else**
它可以帮助你将“成功时才应执行的代码”与 try 块中的“风险代码”分离开，使代码逻辑更清晰。

```py
try:
    f = open('my_file.txt', 'r', encoding='utf-8')
except FileNotFoundError:
    print("文件未找到。")
else:
    # 只有在文件成功打开时，才会执行这里的代码
    print("文件已成功打开，正在读取内容...")
    content = f.read()
    print(content)
    f.close()
```

### 3. try...except...finally 结构

finally 块也是可选的，它的特点是：**无论是否发生异常，它最终总会被执行**。

**为什么使用 finally？**
它非常适合用来执行“清理”操作，比如关闭文件、释放网络连接、解锁资源等，确保这些操作无论程序是否出错都能被执行。

```py
f = None # 在 try 之前初始化
try:
    f = open('my_file.txt', 'r', encoding='utf-8')
    # ... 对文件进行一些操作，可能会出错 ...
    risky_operation(f)
except Exception as e:
    print(f"处理文件时出错: {e}")
finally:
    # 无论 try 成功还是失败，都会执行这里的代码
    if f:
        f.close()
        print("文件已关闭。")
```



**注意：** **对于文件操作，更推荐使用 with 语句**，它能自动处理文件的打开和关闭，内部就利用了类似 finally 的机制。

```py
try:
    with open('my_file.txt', 'r', encoding='utf-8') as f:
        print(f.read())
except FileNotFoundError:
    print("文件未找到。")
```

### 4. 主动抛出异常 raise

除了捕获系统抛出的异常，你也可以在自己的代码中主动抛出异常，以表示发生了某种错误。这在编写函数或库时非常有用，可以向调用者表明输入不符合要求。

```py
def set_age(age):
    if not isinstance(age, int):
        raise TypeError("年龄必须是整数。")
    if age < 0:
        raise ValueError("年龄不能是负数。")
    print(f"年龄已设置为: {age}")

try:
    set_age(25)   # 正常
    set_age(-5)   # 会引发 ValueError
    set_age("二十") # 会引发 TypeError
except (ValueError, TypeError) as e:
    print(f"设置年龄失败: {e}")
```

### 5. 自定义异常

对于大型应用程序，你可能想创建自己的异常类型，以更好地描述程序中特定的错误。自定义异常通常继承自内置的 Exception 类。

```py
class InsufficientFundsError(Exception):
    """当账户余额不足时引发的异常。"""
    pass

class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(f"余额不足。当前余额: {self.balance}, 取款金额: {amount}")
        self.balance -= amount
        return self.balance

# 使用自定义异常
account = BankAccount(100)
try:
    account.withdraw(50)
    print(f"成功取款，剩余余额: {account.balance}")
    account.withdraw(80) # 这里会引发 InsufficientFundsError
except InsufficientFundsError as e:
    print(f"操作失败: {e}")
```

### 总结

1. **具体捕获**：尽量捕获最具体的异常，避免使用裸露的 except:。
2. **try 块最小化**：只在 try 块中放入你认为最可能出错的代码。
3. **使用 finally 或 with 进行清理**：确保资源（如文件、网络连接）总是被释放。
4. **不要滥用**：不要用异常处理来控制正常的程序流程，它应该只用于处理真正的、意外的错误情况。
5. **记录错误**：在 except 块中，通常应该记录下错误信息（例如使用 logging 模块），以便调试。

## 异常类型

### 1. Exception (基类)

- **意义**: 

  ​	这是几乎所有内置的、非系统退出异常的基类。在 except Exception as e: 中捕获它，可以作为一个“万金油”来捕获大多数你没有预料到的程序错误。

- **何时使用**: 

  ​	当你想要捕获所有可能的程序级错误，但又不想捕获像 SystemExit (由 sys.exit() 引发) 或 KeyboardInterrupt (用户按 Ctrl+C) 这样的系统级中断时，捕获 Exception 是一个很好的选择。

### 2. AttributeError

- **意义**: 	尝试访问一个对象上不存在的属性（变量）或方法时引发。

- **触发场景**:

  - **拼写错误**: 	my_list.append(1) 写成了 my_list.apend(1)。
  - **对象类型错误**:     你以为一个变量是字符串，但它实际上是整数，然后你尝试调用字符串的方法，如 num = 123; num.lower()。
  - **访问 None 的属性**:      my_var = None; my_var.some_method()。

  ```py
  x = 10
  print(x.append(5))  # AttributeError: 'int' object has no attribute 'append'
  
  s = "hello"
  print(s.lenght)     # AttributeError: 'str' object has no attribute 'lenght' (拼写错误, 应该是 length)
  ```

### 3. ImportError / ModuleNotFoundError

- **意义**: 

  ​	当 import 语句找不到指定的模块时引发。ModuleNotFoundError 是 ImportError 的一个子类，在 Python 3.6+ 中更具体地表示模块未找到。

- **触发场景**:
  - **模块未安装**:      import pandas，但你没有用 pip install pandas 安装过它。
  - **模块名拼写错误**:     import numpi (应该是 numpy)。
  - **路径问题**:     你尝试导入自己写的模块，但该模块不在 Python 的搜索路径 (sys.path) 中。
  - **循环导入**:     文件 A 导入文件 B，同时文件 B 又导入文件 A。

### 4. IndexError

- **意义**: 

  尝试用一个无效的索引来访问序列（如列表 list、元组 tuple、字符串 str）中的元素时引发。

- **触发场景**:

  - **索引越界**:     列表只有 3 个元素（索引 0, 1, 2），但你尝试访问 my_list[3]。
  - **访问空序列**:     empty_list = []; print(empty_list[0])。

  ```py
  my_list = [10, 20, 30]
  print(my_list[3]) # IndexError: list index out of range
  ```

### 5. KeyError

- **意义**: 

  当你尝试用一个在字典中不存在的键 (key) 来访问字典 (dict) 的值时引发。

- **触发场景**:

  - **键不存在**:      my_dict = {'name': 'Alice'}; print(my_dict['age'])。
  - **键名拼写错误**:      print(my_dict['nane'])。

  ```py
  person = {"name": "Bob", "city": "New York"}
  print(person["age"]) # KeyError: 'age'
  ```

  - **替代方法**: 
  
    ​	为了避免 KeyError，可以使用 .get() 方法，它在键不存在时会返回 None 或指定的默认值：age = person.get('age', 0)。

### 6. FileNotFoundError

- **意义**:     

  尝试打开一个不存在的文件时引发。

- **触发场景**:
  - **文件路径错误**: open('non_existent_file.txt', 'r')。
  - **文件名拼写错误**。
  - **相对路径问题**: 脚本在 A 目录运行，但文件在 B 目录，却使用了相对于 A 目录的路径。

### 7. TypeError

- **意义**: 

  对一个不适当类型的对象执行某个操作或函数时引发。这是最常见的错误之一。

- **触发场景**:

  - **类型不匹配的运算**: 'hello' + 5 (字符串不能和整数相加)。
  - **向函数传递错误类型的参数**: len(123) (len() 函数期望一个序列，而不是整数)。
  - **对不可迭代对象进行迭代**: for i in 12345: ...。
  - **调用函数时传递了错误数量的参数**: def my_func(a, b): ...; my_func(1) (缺少参数)。

  ```py
  result = "5" + 2 # TypeError: can only concatenate str (not "int") to str
  ```

### 8. ValueError

- **意义**: 

  当一个操作或函数的参数类型正确，但其**值**不合适时引发。

- **触发场景**:

  - **类型转换失败**: int('abc') ('abc' 字符串的值无法转换为整数，虽然它的类型是字符串，这是 int() 函数可以接受的类型)。
  - **序列解包数量不匹配**: a, b = [1, 2, 3] (期望解包 2 个值，但列表里有 3 个)。
  - **从列表中移除不存在的值**: my_list = [1, 2]; my_list.remove(3)。

  ```py
  number = int("not a number") # ValueError: invalid literal for int() with base 10: 'not a number'
  ```

  **TypeError vs ValueError 的区别**:

  - len(123) 是 **TypeError**，因为 len() 函数的参数**类型**就不对（它不接受整数）。
  - int('abc') 是 **ValueError**，因为 int() 函数的参数**类型**是正确的（它接受字符串），但这个字符串的**值**是无效的。

### 9. ZeroDivisionError

- **意义**: 

  当除法或模运算的第二个参数（除数）为零时引发。

- **触发场景**:
  - **直接除以零**: 10 / 0。
  - **取模运算除以零**: 10 % 0。
  - **变量为零**: x = 0; result = 100 / x。

### 10. KeyboardInterrupt

- **意义**: 

  当用户在程序运行时按下中断键（通常是 Ctrl+C）时引发。

- **触发场景**:

  - 用户在终端中手动停止一个长时间运行的脚本。
  - 你可以捕获这个异常来执行一些清理工作，比如保存当前进度，然后再退出。

  ```py
  try:
      while True:
          print("Processing...")
          time.sleep(1)
  except KeyboardInterrupt:
      print("\n程序被用户中断。正在退出...")
  ```



### 11. IndentationError / TabError

- **意义**: 

  Python 代码的缩进不正确时引发。这是**语法错误**的一种，通常在程序运行前就会被 Python 解释器发现。

- **触发场景**:
  - if、for、def 等语句块下的代码没有正确缩进。
  - 同一个代码块中混用了 Tab 和空格进行缩进 (TabError)。
  - **IndentationError 通常不能被 try...except 捕获**，因为它在代码解析阶段就出错了。
