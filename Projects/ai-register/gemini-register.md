---
title: 'gemini注册机介绍'
publishDate: 2026-03-30
description: "基于business2api的Gemini注册机介绍"
tags: ['gemini', 'ai-registers']
language: 'Chinese'
first_level_category: "项目实践"
second_level_category: "注册机"
draft: false
---

## 项目介绍

使用[businessa2api 项目](https://github.com/yukkcat/gemini-business2api)进行 Gemini business 账号的批量注册，并且构造 api 端点完成 ai 请求的转发。

该项目本质是一个将网页端的请求进行包装，成为特定 api 端点供给给客户端进行请求的工具，并且提供配套的 ui 面板，同时账号是存储在数据库的，因此只要让主分支的数据库配置和账号更新分支的数据库 url 连接的是同一个数据库即可，采用的是 Direct 直接连接模式，**不要使用远程连接**。

例如：
```yaml
# 模式 A：直连数据库 / Mode A: direct database
DATABASE_URL=postgresql://neonxxxx:npg_xxxxx@exxxxxanb8vqsy.c-6.usxxxxxeon.tech/neondb?sslmode=require
#
```

## 部署步骤

使用两个项目进行配置，一个是主分支，用于网关转发；一个是 [refresh-worker 分支](https://github.com/yukkcat/gemini-business2api/tree/refresh-worker#)，用于账号的批量注册和数据库写入。

因为使用主分支时，注册机由于网关/网络 IP 不干净的问题无法成功注册 Gemini business，[具体参考 issue](https://github.com/yukkcat/gemini-business2api/issues/46)。

克隆主分支下来后首先配置主分支的`.env` 文件：
```yaml
ADMIN_KEY=xxxx
# 示例（Neon PostgreSQL）：
DATABASE_URL=postgresql://nxxxxx.tech/neondb?sslmode=require
#
```
配置密码和远程数据库。

远程数据库使用的[neon 数据库，另一个太难注册不用了](https://console.neon.tech)。配置好后启动容器：`docker compose up -d`，然后执行`docker compose logs -f`观察后台日志（也可以不看）。

管理面板的 url 是：`http://localhost:7860`

API 接口：`http://localhost:7860/v1/chat/completions`

推荐将主分支部署到云上，并且使用域名代理 IP，本地跑 refresh-worker，进行 Gemini 账号的更新和同步，而只要后端数据库的账号服务没有问题，那么网关就不会出现问题。

似乎可以使用[render](https://render.com/)进行网关的部署。

## 补充

就像 refresh-worker 介绍的：

Worker 与主服务可分机部署。

Worker 负责账号刷新执行，不负责 API 网关业务。

远程模式下，本地 worker 仍执行浏览器自动化，只是数据从远程管理接口读写。
