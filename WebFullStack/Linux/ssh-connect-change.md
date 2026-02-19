---
title: '修改 Linux 服务器 SSH 会话连接时长'
publishDate: 2026-02-19
description: '解决 SSH 空闲自动断连的完整方案：通过配置服务端 sshd_config 中的 ClientAliveInterval 和 ClientAliveCountMax，结合客户端 ~/.ssh/config 双侧保活策略，有效防止模型训练期间连接中断。'
tags: ['linux', 'ssh', 'server']
language: 'Chinese'
first_level_category: "Web全栈开发"
second_level_category: "环境与工具"
draft: false
---

# 修改 Linux 服务器 SSH 会话连接时长

> 远程 SSH 连接在空闲一段时间后自动断开是常见问题，尤其在长时间训练模型、等待任务执行时容易触发。本文介绍如何通过修改 SSH 服务端配置来延长会话保持时长。

## 1. 问题背景

### 1.1 为什么 SSH 会话会断开

SSH 连接断开通常有三个层面的原因：

| 层面 | 触发机制 | 典型表现 |
|---|---|---|
| **SSH 服务端** | `ClientAliveInterval` / `ClientAliveCountMax` 超时 | 服务端主动断开空闲连接 |
| **网络设备** | NAT/防火墙的空闲连接超时（通常 5-15 分钟） | 连接静默死亡，终端卡住无响应 |
| **SSH 客户端** | 客户端侧的 `ServerAliveInterval` 未配置 | 客户端不发送心跳，无法维持连接 |

三者中任何一个触发都会导致断连。要确保长时间连接不断，需要在服务端和客户端两侧都进行配置。

### 1.2 核心参数

SSH 的连接保活机制依赖以下三个关键参数：

- **`TCPKeepAlive`**：是否启用 TCP 层的 keepalive 探测。设为 `yes` 可以检测已死亡的连接并回收资源
- **`ClientAliveInterval`**：服务端每隔多少秒向客户端发送一次心跳请求（通过加密通道）。客户端收到后自动回复，以此维持连接活性
- **`ClientAliveCountMax`**：连续多少次心跳无响应后，服务端判定连接已死并断开

**最大空闲时长 = `ClientAliveInterval` × `ClientAliveCountMax`**

## 2. 服务端配置

### 2.1 修改 sshd_config

编辑 SSH 服务端配置文件 `/etc/ssh/sshd_config`，找到或添加以下三行：

```bash
TCPKeepAlive yes
ClientAliveInterval 60
ClientAliveCountMax 360
```

**参数含义：**

- `TCPKeepAlive yes`：启用 TCP keepalive，确保死连接能被检测到
- `ClientAliveInterval 60`：每 60 秒发送一次心跳
- `ClientAliveCountMax 360`：允许连续 360 次无响应

最大空闲时长：60 × 360 = 21600 秒 = **6 小时**。

### 2.2 参数选择建议

根据使用场景选择不同的配置：

| 场景 | ClientAliveInterval | ClientAliveCountMax | 最大空闲时长 |
|---|---|---|---|
| 日常开发 | 60 | 60 | 1 小时 |
| 模型训练 | 60 | 360 | 6 小时 |
| 长时间挂机 | 60 | 1440 | 24 小时 |

`ClientAliveInterval` 建议保持 60 秒不变——间隔太大会导致 NAT/防火墙提前关闭空闲连接，间隔太小则增加不必要的网络开销。通过调整 `ClientAliveCountMax` 来控制总时长更为合理。

### 2.3 重启 SSH 服务

修改配置后需要重启 sshd 使其生效：

```bash
systemctl restart sshd
```

**注意**：重启 sshd 不会断开当前已建立的 SSH 连接，只影响新连接。但如果配置文件有语法错误导致 sshd 启动失败，你将无法建立新连接。建议在重启前先测试配置：

```bash
sshd -t
```

无输出表示配置正确。

### 2.4 注意 cloud-init 覆盖

云服务器（如华为云、阿里云）通常会在 `/etc/ssh/sshd_config.d/` 目录下放置 cloud-init 生成的覆盖文件（如 `50-cloud-init.conf`）。该目录下的配置会覆盖主配置文件中的同名参数。修改前应检查：

```bash
ls /etc/ssh/sshd_config.d/
cat /etc/ssh/sshd_config.d/*.conf
```

确认覆盖文件中没有与你的修改冲突的参数。

## 3. 客户端配置（可选）

服务端配置只解决了"服务端不主动断开"的问题。如果中间的 NAT/防火墙会丢弃空闲连接，还需要客户端主动发送心跳。

### 3.1 全局配置

编辑客户端的 `~/.ssh/config`（不存在则创建）：

```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 30
```

这样客户端每 60 秒向服务端发送一次心跳，连续 30 次无响应才断开（30 分钟）。与服务端的心跳双向配合，即使中间经过多层 NAT 也能保持连接不断。

### 3.2 单次连接配置

如果不想修改全局配置，可以在连接时通过参数指定：

```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=30 user@server
```

## 4. 辅助方案

### 4.1 使用 tmux 防止意外断连

即使配置了心跳，网络抖动仍可能导致断连。使用 tmux 可以确保断连后任务不中断，重连后恢复会话：

```bash
# 新建会话
tmux new -s train

# 在 tmux 中启动训练任务
python webui.py

# 断连后重新连入
tmux attach -t train
```

### 4.2 使用 autossh 自动重连

对于需要长期保持的连接，`autossh` 可以在断连后自动重新建立 SSH 连接：

```bash
autossh -M 0 -o "ServerAliveInterval=60" -o "ServerAliveCountMax=3" user@server
```

## 5. 配置验证

### 5.1 查看当前生效的配置

```bash
sshd -T | grep -i "clientalive\|tcpkeepalive"
```

输出示例：

```
tcpkeepalive yes
clientaliveinterval 60
clientalivecountmax 360
```

### 5.2 确认服务状态

```bash
systemctl status sshd
```

确保 sshd 处于 `active (running)` 状态。

### 5.3 常见问题

| 问题 | 原因 | 解决方式 |
|---|---|---|
| 修改后仍然断连 | cloud-init 覆盖了配置 | 检查 `/etc/ssh/sshd_config.d/` 下的文件 |
| 连接卡住不断开 | 网络中断但 TCP 未检测到 | 确认 `TCPKeepAlive yes` 已启用 |
| 重启 sshd 后无法连接 | 配置文件语法错误 | 修改前先用 `sshd -t` 测试 |
| 客户端仍然超时 | 客户端未配置心跳 | 添加 `ServerAliveInterval` 到 `~/.ssh/config` |
