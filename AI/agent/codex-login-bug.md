---
title: 'Windows 上 Codex CLI 登录报错 os error 10013 的排查与修复'
publishDate: 2026-05-23
description: "Windows 上 Codex CLI 登录报错 os error 10013 的排查与修复"
tags: ['agent', 'codex']
language: 'Chinese'
first_level_category: "AI"
second_level_category: "agent"
draft: false
---


## 问题描述

在 Windows 11 上使用 `codex login` 登录 ChatGPT Pro 订阅时，报错：

```
Error logging in: 以一种访问权限不允许的方式做了一个访问套接字的尝试。 (os error 10013)
```

无论是 PowerShell 还是 cmd，管理员模式还是普通模式，均无法登录。

## 排查过程

### 1. 代理问题？不是

最初怀疑是网络代理导致。检查了 WinHTTP 代理设置：

```powershell
netsh winhttp show proxy
# 输出：直接访问(没有代理服务器)。
```

尝试设置代理端口 7890，问题依旧。排除代理原因。

### 2. 防火墙？不是

临时关闭 Windows Defender 防火墙，问题依旧。排除防火墙原因。

### 3. 端口占用？不是

检查 1455 端口：

```powershell
netstat -ano | findstr :1455
```

无输出，端口未被任何进程占用。

## 根因

**Windows 的 Hyper-V / WSL2 / Docker 等组件动态保留了一段 TCP 端口范围，Codex 登录回调端口 1455 恰好落在被保留的范围内。**

Codex 的 OAuth 登录流程需要在本地 `127.0.0.1:1455` 启动一个临时 HTTP 服务器接收浏览器回调。即使端口没有被任何进程占用，Windows 也会因为端口保留策略直接拒绝绑定，报出 10013 错误。

验证方法：

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

输出示例：

```
Protocol tcp Port Exclusion Ranges

Start Port    End Port
----------    --------
      1059        1158
      1159        1258
      1359        1458    <-- 1455 在这个范围内
```

## 解决方案

**以管理员身份运行 PowerShell**，执行：

```powershell
net stop winnat
net start winnat
```

然后重新登录：

```powershell
codex login
```

重启 WinNAT 服务后，Windows 会重新分配端口保留范围，1455 大概率会被释放。

如果仍然不行，可以连带重启相关服务：

```powershell
net stop winnat
net stop hns
net stop vmms
net start winnat
net start hns
net start vmms
codex login
```

登录成功后服务自动恢复，不影响 WSL2、Docker 等功能。

## 总结

| 排查项 | 结果 |
|--------|------|
| 代理设置 | 无关 |
| 防火墙 | 无关 |
| 端口占用 | 未占用 |
| Windows 端口保留 | **根因** |

这个 bug 很隐蔽——端口明明没有被占用，却报"访问权限不允许"。核心原因是 Windows 的动态端口保留机制（由 Hyper-V/WSL2/Docker 触发），而非传统的端口冲突或权限问题。
