---
title: 配置anyrouter的网络环境
published: 2025-11-24
description: "解决使用anyrouter的claude code报错 502等网络不稳定原因"
first_level_category: "Web全栈开发"
second_level_category: "前端技术"
tags: ['Vercel']
draft: false
---

本文章主要介绍笔者在使用 anyrouter 的 claude code 时出现：

- `Unable to connect to API (ERR_SSL_SSLV3_ALERT_HANDSHAKE_FAILURE)`
- `520 status code (no body)`
- `juayohuang.top | 520: Web server is returning an unknown error`

等问题。最后一条报错的原因后续会介绍。

第一条是证书信任链或 SNI（服务器名称指示）配置不匹配，Node.js 认为连接不安全，直接切断了握手。常出现在 claude-code（客户端）想通过本地代理（7890）去连接 anyrouter（服务端）时

第二条就是经典的网络问题。有可能是：
- 代理不稳定
- 跳数过多：本地 => 代理服务器 => anyrouter CDN => claude code 服务器
- 链接超时

为解决以上问题，这里有几种可选择的方案，可以一起用，也可以单独用，取决于是否能解决您的问题。笔者全都用上了（哭）

## 配置 `settings.json`（最有效）

当你运行指令安装/更新claude code后，目录`C:\Users\Alen\.claude`会有一个`settings.json`文件，这是claude code运行时的默认配置。（环境变量高于此配置文件）

将内容修改为：
```json
{

  "env": {
    "ANTHROPIC_BASE_URL": "anyrouter的base_url或者你自己搭建的流量中转站",
    "ANTHROPIC_API_KEY": "your_api_key",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1 //核心
  },
  "permissions": {
    "allow": [],
    "deny": []
  }
}

```

之后运行 claude code 尝试是否成功。

> 有人建议这么写，我没试过，不确保能用。但以上配置我确认是可以使用的。
>   "ANTHROPIC_BASE_URL": "https://anyrouter.top",
>   "DISABLE_AUTOUPDATER": "1",
>   "DISABLE_ERROR_REPORTING": "1",
>   "DISABLE_TELEMETRY": "1",
>   "HTTPS_PROXY": "http://127.0.0.1:端口",
>   "HTTP_PROXY": "http://127.0.0.1:端口"
> 

## SSL 证书问题

打开终端运行 claude code 后如果出现`ERR_SSL_SSLV3_ALERT_HANDSHAKE_FAILURE`。可在当前终端强制 Node.js 忽略 SSL 证书错误：
- Mac/Linux:
  ```bash
    # 1. 设置允许不安全连接
    export NODE_TLS_REJECT_UNAUTHORIZED=0

    # 2. 确保代理设置正确（可选）
    export HTTPS_PROXY=http://127.0.0.1:7890

    # 3. 运行
    claude
  ```
- Windows:（Powershell）
  ```bash
  $Env:NODE_TLS_REJECT_UNAUTHORIZED="0"
  $Env:HTTPS_PROXY="http://127.0.0.1:7890"（可选）
  claude
  ```

## 配置个人流量中转站

Anyrouter 主站流量过大，有时（经常）网络不稳定会出现 502 等错误，因此可以选择自己搭建一个流量中转站来转发流量。本地使用代理站点的域名，而不是anyrouter的域名进行访问。

当然最主要的原因还是大陆这边网络容易被切断。anyrouter在美国，因此搭建一个海外流量站点，将流量转到香港比较方便国内访问。即 User => 代理站点 => anyrouter => claude code

笔者目前用的就是个人的代理站点来转发流量。搭建步骤如下。

### 申请域名

这一步是申请一个域名，防止站点由于域名污染而影响 Claude code 的使用。域名申请您决定，笔者用的是阿里云的。

### 使用 Vercel 搭建站点

**不要使用 Cloudflare 搭建站点！！！** Anyrouter 使用的就是 cloudflare，如果你的站点也使用的是 cloudflare 搭建（cloudflare 的 Workers），那么会导致**路由无限回环和重定向风暴，报错 1102**。

表现为：`Error 1102 Worker exceeded resource limits` 和访问 Vercle 部署的站点 url ，打开控制台可以看到 your_point.workers.dev 的请求不断"消失->出现"循环。

这不是因为服务器挂了，而是因为你搭建的cloudflare Worker 和 Cloudflare 之间陷入了死循环和重定向陷阱。

因此需要使用除 Cloudflare 以外的工具（如 Vercel）搭建站点。步骤如下。

> Vercel 是 AWS 的

#### **1. 项目准备**

在 GitHub 上新建一个仓库，用来给 vercel 部署项目。仓库链接（供参考）：https://github.com/JuyaoHuang/cloudflare-transpoint

项目结构：
```bash
cloudflare-transpoint
 ┣ api
 ┃ ┗ index.js
 ┗ vercel.json
```

核心文件为 index.js，项目结构一定要和这个完全相同。

```Javascript
// 文件路径: api/index.js

export const config = {
  runtime: 'edge', // 必须开启 Edge Runtime 以支持流式传输
  // regions: ['sin1'], // 可选：指定新加坡节点 (离国内近)，或者不写让它自动选择
};

export default async function handler(request) {
  const url = new URL(request.url);

  // 1. 健康检查：防止浏览器直接访问报错
  if (url.pathname === '/' || url.pathname === '/index.html') {
    return new Response('AnyRouter Proxy on Vercel is Active.', {
      status: 200,
      headers: { 'Content-Type': 'text/plain' },
    });
  }

  // 2. 上游地址配置
  const UPSTREAM_HOST = 'c.cspok.cn';
  const UPSTREAM_URL = `https://${UPSTREAM_HOST}`;
  
  // 3. 构造转发 URL
  const targetUrl = new URL(url.pathname + url.search, UPSTREAM_URL);

  // 4. 请求头处理
  const headers = new Headers(request.headers);
  headers.set('Host', UPSTREAM_HOST);
  headers.set('Referer', `https://${UPSTREAM_HOST}/`);
  
  // 移除 Vercel 标记，防止被上游识别
  headers.delete('x-vercel-id');
  headers.delete('x-vercel-deployment-url');
  headers.delete('x-forwarded-for');
  headers.delete('x-real-ip');

  try {
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: headers,
      body: request.body,
      redirect: 'manual',
    });

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: e.message }), { status: 500 });
  }
}
```

vercel.json:
```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/api/index.js"
    }
  ]
}
```

创建好仓库后登录 Vercel 进行部署。

#### **2. 添加二级域名**

笔者域名使用 cloudflare 配置 DNS。一级域名用于个人博客：juayohuang.top，因此需要添加一个**二级域名**作为站点的域名。

> 如果不知道怎么用 cloudaflare 配置，参考这篇文章：https://www.juayohuang.top/posts/webfullstack/backend/deploy_web_in_vercel
> 或者上网自助。

在 cloudflare 的 DNS 记录里添加一条新纪录：
- 类型：CNAME
- 名称：自定（例如 abcd）
- 内容：cname.vercel-dns.com
- 代理状态 (Proxy status)： 关闭 (变成灰色云朵，DNS Only)。

然后在新部署好的 Vercel 项目里 cloudflare_transpoint => Settings => Domains 添加你新申请的二级域名：abcd.juayohuang.top。

因为你在第一步已经设置了 CNAME 指向 Vercel 且关闭了 CF 代理（灰云），Vercel 应该能很快自动申请到 SSL 证书并显示两个绿色的勾。

浏览器访问 https://abcd.juayohuang.top，应该看到 "AnyRouter Vercel Proxy is Active"。

这样站点就配置好了

### 3.**配置 `settings.json`**

回到第一步的配置 `settings.json`，将 `"ANTHROPIC_BASE_URL": "anyrouter的base_url或者你自己搭建的流量中转站",`即可。
