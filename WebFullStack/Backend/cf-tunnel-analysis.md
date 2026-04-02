---
title: 'Cloudflare Tunnel 反向代理链路解析与超时问题'
publishDate: 2026-04-01
description: "从域名解析到 CF Tunnel 再到源站，解析完整请求链路及 524 超时的成因与解决方案"
tags: ['cloudflare', 'nginx']
language: 'Chinese'
first_level_category: "Web全栈开发"
second_level_category: "后端技术"
draft: false
---

## 核心问题

- Cloudflare Tunnel 反向代理的完整请求链路
- 524 超时错误的成因及其在链路中的位置
- 流式传输（SSE）如何规避超时
- 非流式长耗时请求的解决方案

## 背景

将一个运行在阿里云服务器 `localhost:7860` 上的 Gemini API 服务通过域名 `gemini.juyaohuang.com` 暴露到公网。由于阿里云服务器绑定未备案域名会被拦截（80/443 端口流量被劫持到备案提示页），因此选择 Cloudflare Tunnel 作为反向代理方案，绕过备案检查。

## 请求链路

### 传统 Nginx 反向代理

在没有备案限制的场景下，一个标准的反向代理链路如下：

```
用户浏览器
    │
    │  DNS 解析：gemini.juyaohuang.com → 服务器公网 IP
    ▼
阿里云服务器（公网 IP:443）
    │
    │  Nginx 反向代理：proxy_pass http://127.0.0.1:7860
    ▼
应用服务（localhost:7860）
    │
    │  应用内部发起 HTTPS 请求
    ▼
Google Gemini API（外部服务）
```

这条链路简洁直接：用户通过 DNS 找到服务器 IP，Nginx 将请求转发给本地应用，应用再去调用 Google API。**但在国内服务器上，80/443 端口的入站流量会被检查域名备案状态**，未备案域名的请求会被劫持。

### Cloudflare Tunnel 链路

为绕过备案拦截，引入 Cloudflare Tunnel（cloudflared）后，链路变为：

```
用户浏览器
    │
    │  DNS 解析：gemini.juyaohuang.com → Cloudflare Edge IP（非源站 IP）
    ▼
Cloudflare Edge 节点（全球 CDN 边缘）  ←── 100 秒超时限制
    │
    │  通过 Tunnel 隧道转发（长连接，由 cloudflared 主动建立）
    ▼
cloudflared 进程（运行在阿里云服务器上）
    │
    │  转发到本地服务
    ▼
应用服务（localhost:7860）
    │
    │  应用内部发起 HTTPS 请求（通过代理访问 Google）
    ▼
Google Gemini API
```

与传统方案的关键区别在于：**用户的请求不再直接到达阿里云服务器**。DNS 解析的结果是 Cloudflare 的 IP，而不是源站 IP。流量先到 CF 的边缘节点，再通过预先建立好的隧道转发到源站。

这就像收快递：传统方式是快递员直接送到你家（备案检查相当于小区门禁不让进）；CF Tunnel 相当于快递先送到小区外面的代收点（CF Edge），然后你自己从里面出来取（cloudflared 主动向外建立连接），门禁管不着你从里面出去。

### 为什么能绕过备案？

阿里云的备案检查机制作用于**入站的 80/443 端口流量**，检查 HTTP 请求中的 Host 头是否已备案。而 CF Tunnel 的工作方式是：

1. `cloudflared` 进程从服务器**主动向外**建立到 Cloudflare 的长连接（出站流量）
2. 用户请求到达 CF Edge 后，通过这条**已有的出站隧道**传回服务器
3. 服务器上不需要监听 80/443 端口接收外部流量

因此，阿里云的入站端口检查机制不会被触发。

## 524 超时问题

### 超时发生在哪？

524 是 Cloudflare 特有的 HTTP 状态码，含义是"CF Edge 等待源站响应超时"。它发生在链路的这个位置：

```
用户 ←── 返回 524 ─── CF Edge    ···超过 100 秒未收到数据···    源站（还在等 Google 返回）
```

CF Edge 节点有一个硬性限制：如果源站在 **100 秒内没有返回任何数据**（free 计划），CF 就主动断开连接并向用户返回 524 错误。这个限制是 Cloudflare 平台级别的，无法通过 `cloudflared` 配置修改。

以图片生成请求为例，`gemini-imagen` 模型生成一张图需要 100 秒以上。在这期间，源站一直在等 Google API 返回结果，没有任何数据可以发给 CF Edge。CF Edge 等了 100 秒，认为连接已死，返回 524。而此时源站可能刚刚收到 Google 返回的图片数据，但前端连接已经断了。

### 为什么流式请求不超时？

流式传输（Server-Sent Events, SSE）的核心机制是：服务端不等所有数据准备好再一次性返回，而是**边生成边发送**。对于聊天接口，模型每生成一个 token（几十毫秒）就立刻通过 SSE 推送一个 chunk：

```
CF Edge 收到 chunk 1（第 0.5 秒）  → 转发给用户
CF Edge 收到 chunk 2（第 1.0 秒）  → 转发给用户
CF Edge 收到 chunk 3（第 1.5 秒）  → 转发给用户
...
CF Edge 收到 chunk N（第 90 秒）   → 转发给用户
CF Edge 收到 [DONE]（第 91 秒）    → 连接正常关闭
```

由于数据**持续流动**，CF Edge 的 100 秒空闲计时器不断被重置，所以即使总耗时超过 100 秒也不会触发 524。

但图片生成是非流式的。模型必须完整生成图片后才能返回，中间没有任何数据输出。这段"沉默期"一旦超过 100 秒，CF Edge 就会断开连接。

### 解决方案对比

| 方案 | 适用场景 | 原理 |
|------|---------|------|
| 使用 `stream: true` | 文本聊天 | 数据持续流动，CF 不会判定超时 |
| 直连服务器 IP | 图片/视频生成 | 绕过 CF，无 100 秒限制 |
| 升级 CF 计划（Enterprise） | 所有场景 | 可自定义超时时间至 600 秒 |
| 放弃 CF Tunnel + 域名备案 | 所有场景 | 使用 Nginx 直接反代，无中间层超时 |

## 总结

Cloudflare Tunnel 本质上是在用户和源站之间插入了一个中间人（CF Edge），代价是引入了一层不可控的超时限制。对于实时性要求高的流式聊天场景，SSE 的持续数据流可以天然规避超时；但对于图片/视频生成等需要长时间等待的非流式请求，CF 的 100 秒硬限制成为无法绕过的瓶颈。

当应用场景同时包含流式和非流式长耗时请求时，要么为不同接口选择不同的链路（流式走 CF、非流式直连），要么从根本上去掉 CF 这一层——备案或换用海外服务器。
