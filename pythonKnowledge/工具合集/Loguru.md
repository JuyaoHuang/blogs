---
title: Loguru
author: Alen
published: 2025-10-10
description: "日志记录和分析工具：Loguru的介绍"
first_level_category: "python"
second_level_category: "开发工具"
tags: ['python']
draft: false
---
# Loguru

​	Loguru 是 Python 中一个广受欢迎的第三方日志库，它旨在让日志记录变得简单、愉快且功能强大。相比 Python 内置的 logging 模块，Loguru 提供了更优雅、更现代的 API，并且开箱即用，无需繁琐的配置。

### 1.介绍

Python 内置的 logging 模块功能强大，但配置起来相对复杂，尤其是对于初学者。Loguru 解决了 logging 的许多痛点：

| 特性                | 内置 logging 模块                                            | Loguru 库                                                    |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **上手难度**        | **高**。需要手动创建 logger、handler、formatter 并将它们组合起来，代码量大。 | **极低**。导入即可使用，零配置开箱即用。                     |
| **日志输出**        | 默认不输出任何内容，需要配置 handler。                       | 默认向 sys.stderr 输出带有颜色和格式的日志。                 |
| **文件日志**        | 配置复杂，需要 FileHandler，实现日志轮转（rotation）需要额外代码。 | 一行代码即可实现，并且内置了日志轮转、保留、压缩等高级功能。 |
| **异常捕获**        | 需要在 except 块中手动调用 logging.exception()。             | 提供 logger.catch() 装饰器，可以优雅地自动捕获和记录异常，包含完整的堆栈信息。 |
| **格式化**          | 使用类似 C 语言的 % 格式化或 .format()，不够直观。           | 支持 f-string 风格的 {} 格式化，代码更简洁。                 |
| **线程/多进程安全** | 需要自己处理锁，以保证安全。                                 | **默认线程安全、多进程安全**。                               |

总而言之，Loguru 的设计哲学是**用最少的代码做最多的事，并提供更强大、更人性化的功能**。

### 2.安装

Loguru 是一个第三方库，通过 pip 安装：

    pip install loguru

### 3. 核心用法

#### 3.1 基础入门：开箱即用

你不需要任何配置，只需导入 `logger` 对象即可开始使用。

```python
from loguru import logger

logger.debug("这是一条 debug 信息")
logger.info("这是一条 info 信息")
logger.warning("这是一条 warning 信息")
logger.error("这是一条 error 信息")
logger.critical("这是一条 critical 信息")
  
```

当你运行这段代码时，你会在控制台看到类似下面的输出，并且带有颜色（在支持颜色的终端中）：

```bash
2023-10-27 10:30:00.123 | DEBUG    | __main__:<module>:3 - 这是一条 debug 信息
2023-10-27 10:30:00.124 | INFO     | __main__:<module>:4 - 这是一条 info 信息
2023-10-27 10:30:00.124 | WARNING  | __main__:<module>:5 - 这是一条 warning 信息
2023-10-27 10:30:00.124 | ERROR    | __main__:<module>:6 - 这是一条 error 信息
2023-10-27 10:30:00.125 | CRITICAL | __main__:<module>:7 - 这是一条 critical 信息
```

**默认格式解析：**

```
{time} | {level} | {name}:{function}:{line} - {message}
```

**时间** | **日志级别** | **模块名:函数名:行号 - 日志消息**

#### 占位符处理变量

1. **官方推荐 '{}'占位符**

   ```py
   a = 1
   loggger.info('a={}',a)
   ```

   

2. **使用f'{}'的常见字符串格式化**

   ```py
   a=1
   logger.info(f'a={a}')
   ```

   

#### 3.2 核心概念：Sink（接收器）与 logger.add()

Loguru 的强大之处在于通过 logger.add() 方法配置**接收器 (Sink)**。一个 Sink 就是日志消息的目的地，它可以是：

- 一个文件名（字符串）
- 一个文件对象 (e.g., sys.stdout)
- 一个函数或可调用对象

logger.add() 方法有许多强大的参数，我们来看几个最重要的。

**示例1：将日志记录到文件**

这是最常见的用法。只需一行代码，即可实现强大的文件日志功能。

```py
from loguru import logger

# 添加一个文件接收器
# 这会创建一个名为 "file.log" 的文件，并将 INFO 级别及以上的日志写入
logger.add("file.log", level="INFO")

logger.info("这条信息会同时出现在控制台和 file.log 文件中")
logger.debug("这条信息只会出现在控制台，因为文件日志级别设置为 INFO")
  
```

#### 3.3 文件日志的高级功能

logger.add() 支持非常实用的文件管理功能。

- **日志轮转 (Rotation):** 自动创建新文件，防止单个日志文件过大。

  ```py
  # 当文件大小超过 500 MB 时，自动创建新文件
  logger.add("file_size_rotation.log", rotation="500 MB")
  
  # 每天中午 12:00 创建一个新文件
  logger.add("file_time_rotation.log", rotation="12:00")
  
  # 每周创建一个新文件
  logger.add("file_weekly_rotation.log", rotation="1 week")
    
  ```

- **日志保留 (Retention):** 自动删除旧的日志文件，防止占用过多磁盘空间。

  ```py
  # 只保留最近 10 天的日志文件
  logger.add("file_retention.log", rotation="1 day", retention="10 days")  
  ```

- **日志压缩 (Compression):** 自动将轮转后的旧日志文件压缩，以节省空间。

  ```
  # 轮转后的日志文件会自动压缩为 .zip 格式
  logger.add("file_compressed.log", rotation="10 MB", compression="zip")
    
  ```

**综合示例：**
一个生产环境中常用的配置可能如下：

```json
    logger.add(
    "logs/app.log",          # 日志文件路径
    level="INFO",            # 日志级别
    rotation="1 day",        # 每天轮转
    retention="7 days",      # 最多保留 7 天
    compression="zip",       # 压缩旧文件
    encoding="utf-8",        # 文件编码
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}" # 自定义格式
)
  
```

### 4. Loguru 的强大特性

#### 4.1 优雅的异常捕获

这是 Loguru 的**杀手级功能**。使用 logger.catch() 装饰器，可以非常方便地捕获函数中未处理的异常，并自动记录详细的堆栈跟踪信息。

```py
from loguru import logger

@logger.catch
def divide(a, b):
    return a / b

divide(10, 0)  
```

当运行这段代码时，你不需要写 try...except 块，Loguru 会自动捕获ZeroDivisionError 并输出非常详细的错误报告，包括变量的值，极大地简化了调试过程。

#### 4.2 字符串格式化

Loguru 使用 {} 占位符，就像 Python 的 f-string 一样，非常直观。

```python
user_id = 123
ip_address = "192.168.1.1"

logger.info("用户 {} 从 IP {} 登录成功", user_id, ip_address)
```

#### 4.3 结构化日志 (Structured Logging)

对于需要被日志分析系统（如 ELK, Splunk）处理的场景，结构化日志（通常是 JSON 格式）非常有用。

- **使用 serialize=True**

  ```Python
  import sys
  
  # 将日志序列化为 JSON 格式并输出到 stdout
  logger.add(sys.stdout, serialize=True)
  
  logger.info("这是一条结构化日志")
  # 输出会是一个 JSON 字符串，包含时间、消息、级别、上下文等信息
  ```

- **使用 .bind() 添加额外数据**

   

  ```Python
  log = logger.bind(user_id=123, request_id="abc-xyz")
  log.info("用户请求处理中")
  # 输出的 JSON 中会包含 "user_id" 和 "request_id" 字段
  ```

### 5. 与现有代码和库集成

如果你的项目中使用了其他遵循 Python 内置 logging 模块的库（例如 requests, fastapi），你可以让 Loguru 拦截这些库产生的日志，统一管理。

```py
import logging
import sys
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取日志记录对应的级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 查找调用栈的深度
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# 配置内置 logging，让所有日志都通过我们的拦截器
logging.basicConfig(handlers=[InterceptHandler()], level=0)

# 现在，任何使用标准 logging 模块的库，其日志都会被 loguru 捕获
import requests
try:
    requests.get("http://a.non.existent.domain")
except requests.exceptions.RequestException:
    pass # 错误日志会被 loguru 捕获并打印

logger.info("上面的 requests 错误日志被 Loguru 成功拦截了！")
  
```

### 总结

Loguru 是一个现代、强大且极易使用的 Python 日志库。它的主要优点在于：

- **极简的配置：** 开箱即用，一行代码搞定复杂的文件日志。
- **强大的功能：** 内置日志轮转、压缩、保留，以及优雅的异常捕获。
- **优秀的可读性：** 默认格式清晰，支持颜色，字符串格式化友好。
- **生产环境适用：** 默认线程/多进程安全，支持结构化日志，易于集成。

对于任何新的 Python 项目，或者想要改进现有项目日志系统的开发者来说，Loguru 都是一个非常值得推荐的选择。