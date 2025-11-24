---
title: requests库
author: Alen
published: 2025-10-10
description: "网络请求工具：requests库的介绍"
first_level_category: "Python"
second_level_category: "工程化与工具"
tags: ['python']
draft: false
---

# requests

requests 是 Python 中一个极其流行且功能强大的第三方库，它的口号是 **“HTTP for Humans”**（为人类设计的HTTP）。顾名思义，它旨在使发送 HTTP 请求变得极其简单和直观。

---

## 作用

Python 内置了 urllib 库来处理 HTTP 请求，但它的 API 比较复杂和繁琐。requests 库在此基础上进行了封装，提供了更简洁、更人性化的接口，并处理了很多底层细节，例如连接管理、cookie 持久化、重定向等。

**requests 相比 urllib 的优点：**

- **API 简洁直观：** 代码更少，可读性更高。
- **功能丰富：** 自动处理 JSON、自动解压内容、支持会话（Session）、方便地处理文件上传等。
- **健壮性：** 自动处理连接池，使得性能更优。

---

## 安装

requests 是一个第三方库，需要通过 pip 进行安装。在终端或命令行中运行：

```
pip install requests
```

## 核心用法

requests 库的核心就是发送 HTTP 请求。HTTP 请求有多种方法（Method），最常用的是 GET 和 POST。

### 发送 GET 请求

GET 请求通常用于从服务器获取（或“读取”）数据。

**基本示例：**

```py
import requests

# 发送一个 GET 请求到指定的 URL
response = requests.get('https://api.github.com/events')

# 打印响应对象，可以看到状态码 [200] 表示成功
print(response) 
# 输出: <Response [200]>
```

**处理响应 (Response)：**
当你发送一个请求后，requests 会返回一个 Response 对象，这个对象包含了服务器返回的所有信息。

```py
# 1. 检查状态码
if response.status_code == 200:
    print("请求成功！")
elif response.status_code == 404:
    print("页面未找到！")

# 2. 获取响应内容
# response.text: 以文本（字符串）形式返回响应内容（requests 会自动解码）
print("响应文本内容 (部分):", response.text[:100])

# response.content: 以字节（bytes）形式返回响应内容，适用于图片、视频等二进制文件
# print("响应字节内容 (部分):", response.content[:100])

# response.json(): 如果响应内容是 JSON 格式，可以直接将其解析为 Python 字典或列表
# 这非常方便！
data = response.json()
print("解析后的 JSON 数据 (第一个事件的类型):", data[0]['type'])

# 3. 获取响应头
print("响应头信息:", response.headers)
print("响应内容的类型:", response.headers['Content-Type'])

# 4. 获取编码
print("响应内容的编码:", response.encoding)
```

**带参数的 GET 请求：**
有时你需要向 URL 传递参数，例如 http://httpbin.org/get?key=value。requests 允许你通过 params 参数以字典形式传递。

```py
# 定义参数
payload = {'key1': 'value1', 'key2': 'value2'}

# requests 会自动将参数编码并附加到 URL 后面
response = requests.get('https://httpbin.org/get', params=payload)

# 打印实际发出的 URL
print("实际请求的 URL:", response.url)
# 输出: https://httpbin.org/get?key1=value1&key2=value2
```

---

### 发送 POST 请求

POST 请求通常用于向服务器提交（或“写入”）数据，例如提交表单、上传文件等。

**提交表单数据：**
使用 data 参数，传递一个字典。

```python
payload = {'username': 'john', 'password': 'doe'}
response = requests.post('https://httpbin.org/post', data=payload)

# 查看服务器收到的表单数据
print(response.json()['form'])
# 输出: {'username': 'john', 'password': 'doe'}
```

**提交 JSON 数据：**
现代 API 通常使用 JSON 格式进行通信。使用 json 参数，requests 会自动将字典转换为 JSON 字符串，并设置正确的请求头 Content-Type: application/json。

```python
payload = {'name': 'Alice', 'age': 30}
response = requests.post('https://httpbin.org/post', json=payload)

# 查看服务器收到的 JSON 数据
print(response.json()['json'])
# 输出: {'name': 'Alice', 'age': 30}
```

### 其他 HTTP 方法

requests 也支持其他所有 HTTP 方法：

- requests.put(): 更新资源
- requests.delete(): 删除资源
- requests.head(): 获取响应头
- requests.options(): 获取服务器支持的方法

## 高级用法

### 自定义请求头 (Headers)

有些网站需要特定的请求头才能访问，例如 User-Agent（模拟浏览器）。

你可以通过 headers 参数传递一个字典。

```py
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Authorization': 'Bearer YOUR_ACCESS_TOKEN' # 例如 API 认证
}
response = requests.get('https://httpbin.org/headers', headers=headers)
print(response.json())
```

### 会话对象 (Session)

如果你需要向同一个网站发送多个请求，使用会话对象 (Session) 会带来两个好处：

1. **Cookie 持久化：** 会话会自动记录并发送服务器设置的 Cookie。这对于需要登录才能访问的网站非常有用。
2. **性能提升：** 会话会复用底层的 TCP 连接（连接池），减少了重复建立连接的开销。

```py
# 创建一个会话对象
s = requests.Session()

# 第一次请求，登录并获取 cookie
s.get('https://httpbin.org/cookies/set/sessioncookie/123456789')

# 第二次请求，会自动带上之前获取的 cookie
response = s.get('https://httpbin.org/cookies')

print(response.json())
# 输出: {'cookies': {'sessioncookie': '123456789'}}
```

### 异常处理

网络请求可能会失败（如网络中断、DNS 查询失败等），或者服务器返回错误的状态码。

- **处理网络层面的异常：**
  使用 try...except 块来捕获 requests.exceptions.RequestException 或更具体的异常。

  ```python
  try:
      response = requests.get('http://a.very.nonexistent.domain.com', timeout=5)
  except requests.exceptions.ConnectionError as e:
      print(f"连接错误: {e}")
  ```

- **处理 HTTP 错误状态码：**
  response.raise_for_status() 是一个非常有用的方法。如果响应的状态码是 4xx 或 5xx（客户端或服务器错误），它会抛出一个 HTTPError 异常。

  ```py
  try:
      response = requests.get('http://httpbin.org/status/404')
      response.raise_for_status()  # 如果状态码不是 2xx，则抛出异常
  except requests.exceptions.HTTPError as e:
      print(f"HTTP 错误: {e}")
  ```

### 文件上传与下载

- **文件上传：**
  使用 files 参数。

  ```python
  files = {'file': open('report.xls', 'rb')} # 'rb' 表示以二进制读取
  response = requests.post('https://httpbin.org/post', files=files)
  # print(response.json())
  ```

- **下载大文件（流式下载）：**
  对于大文件（如视频、安装包），为了防止内存耗尽，应该使用流式下载。

  设置 stream=True，然后迭代 response.iter_content()。
  
  ```py
  url = 'https://some-large-file-url.com/file.zip'
  with requests.get(url, stream=True) as r:
      r.raise_for_status()
      with open('downloaded_file.zip', 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192): # 每次写入 8KB
              f.write(chunk)
  print("文件下载完成！")
  ```

### 超时设置 (Timeout)

为了防止请求永远等待下去，设置 timeout 参数是一个好习惯。单位是秒。

```python
# 等待服务器响应最多 5 秒
response = requests.get('https://github.com', timeout=5)
```

### SSL 证书验证

默认情况下，requests 会验证 SSL 证书。如果遇到证书问题（例如在某些内部网络），你可以通过 verify=False 来禁用验证（**注意：这会带来安全风险，请谨慎使用**）。

```py
# 禁用 SSL 验证，并抑制警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
response = requests.get('https://some-site-with-bad-ssl.com', verify=False)
```

---

## 总结

requests 库是 Python 中进行网络编程和 Web 开发（尤其是 API 交互和爬虫）的必备工具。它将复杂的 HTTP 协议简化为了几个简单易懂的函数调用，极大地提高了开发效率。掌握了 GET, POST 请求、处理响应、使用会话和异常处理等核心功能，你就能应对绝大多数的网络请求场景。