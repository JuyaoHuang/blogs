---
title: 'Linux 服务器使用 Mihomo 内核实现流量转发'
publishDate: 2026-02-19
description: '在无图形界面的 Linux 服务器上部署 Mihomo（Clash.Meta），通过配置系统代理环境变量实现 HTTP/HTTPS 流量转发，解决访问 GitHub、HuggingFace 等海外资源的网络问题。'
tags: ['linux', 'proxy', 'networking']
language: 'Chinese'
first_level_category: "Web全栈开发"
second_level_category: "环境与工具"
draft: false
---

# Linux 服务器使用 Mihomo 内核实现流量转发

> Mihomo（原 Clash.Meta）是一个基于 Go 语言的代理内核，支持 Trojan、VMess、VLESS、Shadowsocks 等多种协议，适合在无图形界面的 Linux 服务器上部署透明代理或系统级流量转发。

## 1. Mihomo 简介

### 1.1 什么是 Mihomo

Mihomo 是 Clash.Meta 的继任项目，属于 Clash 生态中的高性能代理内核。与原版 Clash 相比，Mihomo 支持更多协议（如 Trojan、VLESS、Hysteria2 等）、更灵活的规则匹配以及 fake-IP 模式的 DNS 解析。

### 1.2 为什么在服务器上使用

在 GPU 训练服务器上，经常需要从 GitHub、HuggingFace 等海外平台下载模型文件和代码仓库。直接访问可能速度极慢或无法连通，通过 Mihomo 进行流量转发可以解决这一问题。

### 1.3 工作原理

Mihomo 在本地监听一个混合代理端口（默认 7890），同时提供 HTTP 和 SOCKS5 代理服务。通过设置 shell 环境变量，将系统的 HTTP/HTTPS 流量指向该端口，由 Mihomo 根据规则配置决定是直连（DIRECT）还是走代理节点转发。

```
应用程序 → 环境变量指向 127.0.0.1:7890 → Mihomo → 规则匹配 → 直连/代理转发
```

## 2. 安装步骤

### 2.1 下载二进制文件

从 Mihomo 官方 GitHub Release 页面下载对应平台的预编译二进制文件：

```bash
# 下载 linux-amd64 版本（以实际最新版本号为准）
wget https://github.com/MetaCubeX/mihomo/releases/download/<version>/mihomo-linux-amd64-<version>.gz

# 解压
gunzip mihomo-linux-amd64-<version>.gz

# 移动到系统路径并赋予执行权限
mv mihomo-linux-amd64-<version> /usr/bin/mihomo
chmod +x /usr/bin/mihomo
```

### 2.2 创建配置目录

```bash
mkdir -p /etc/mihomo
```

### 2.3 下载 GeoIP 和 GeoSite 数据库

规则匹配依赖地理位置数据库，需要下载到配置目录：

```bash
# GeoIP 数据库
wget -O /etc/mihomo/geoip.metadb https://github.com/MetaCubeX/meta-rules-dat/releases/latest/download/geoip.metadb

# GeoSite 数据库
wget -O /etc/mihomo/GeoSite.dat https://github.com/MetaCubeX/meta-rules-dat/releases/latest/download/GeoSite.dat
```

## 3. 配置文件

### 3.1 获取订阅配置

将代理服务商提供的 Clash 订阅配置文件保存为 `/etc/mihomo/config.yaml`。大多数机场都提供 Clash 格式的订阅链接，可以直接下载：

```bash
wget -O /etc/mihomo/config.yaml "你的订阅链接"
```

### 3.2 关键配置项说明

一个典型的 `config.yaml` 包含以下核心部分：

```yaml
# 代理端口配置
mixed-port: 7890          # HTTP + SOCKS5 混合代理端口
external-controller: 127.0.0.1:9090  # API 控制端口

# 模式设置
mode: rule                # rule（规则匹配）/ global（全局代理）/ direct（全部直连）
allow-lan: false          # 是否允许局域网设备连接

# DNS 配置
dns:
  enable: true
  enhanced-mode: fake-ip  # fake-ip 模式，减少 DNS 污染
  nameserver:
    - 223.5.5.5           # 阿里 DNS
    - 119.29.29.29        # 腾讯 DNS
  fallback:
    - 1.1.1.1             # Cloudflare DNS（用于海外域名解析）
    - 8.8.8.8             # Google DNS
```

### 3.3 规则匹配逻辑

`config.yaml` 中的 `rules` 部分定义了流量路由规则，按顺序从上到下匹配：

```yaml
rules:
  - DOMAIN-SUFFIX,github.com,代理节点组    # GitHub 走代理
  - DOMAIN-SUFFIX,huggingface.co,代理节点组 # HuggingFace 走代理
  - GEOIP,CN,DIRECT                        # 国内 IP 直连
  - MATCH,代理节点组                        # 未匹配的走代理
```

这确保了国内流量不绕远路，海外流量通过代理节点转发。

## 4. 注册为系统服务

### 4.1 创建 systemd 服务文件

创建 `/lib/systemd/system/mihomo.service`：

```ini
[Unit]
Description=mihomo Daemon, Another Clash Kernel.
Documentation=https://wiki.metacubex.one
After=network.target nss-lookup.target network-online.target

[Service]
Type=simple
CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_RAW CAP_NET_BIND_SERVICE CAP_SYS_TIME CAP_SYS_PTRACE CAP_DAC_READ_SEARCH CAP_DAC_OVERRIDE
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_RAW CAP_NET_BIND_SERVICE CAP_SYS_TIME CAP_SYS_PTRACE CAP_DAC_READ_SEARCH CAP_DAC_OVERRIDE
ExecStart=/usr/bin/mihomo -d /etc/mihomo
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=10
LimitNOFILE=infinity

[Install]
WantedBy=multi-user.target
```

**关键参数说明：**

- `ExecStart=/usr/bin/mihomo -d /etc/mihomo`：`-d` 指定配置目录
- `Restart=on-failure`：进程异常退出时自动重启
- `RestartSec=10`：重启间隔 10 秒
- `CapabilityBoundingSet`：授予网络管理、原始套接字等必要权限，无需以 root 身份运行全部功能
- `LimitNOFILE=infinity`：不限制文件描述符数量，避免大量连接时报错

### 4.2 启用并启动服务

```bash
# 重载 systemd 配置
systemctl daemon-reload

# 设置开机自启
systemctl enable mihomo

# 启动服务
systemctl start mihomo

# 检查运行状态
systemctl status mihomo
```

## 5. 配置 Shell 环境变量

Mihomo 启动后仅在本地监听端口，还需要告诉系统的命令行工具通过该端口代理。在 `~/.bashrc`（或 `~/.zshrc`）末尾添加：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891
```

保存后执行 `source ~/.bashrc` 使其生效。之后 `wget`、`curl`、`git`、`pip` 等工具的网络请求都会自动经过 Mihomo 代理。

**临时关闭代理**（如需直连）：

```bash
unset http_proxy https_proxy all_proxy
```

## 6. 验证与排障

### 6.1 确认服务运行

```bash
# 检查进程
ps aux | grep mihomo

# 检查端口监听
ss -tlnp | grep 7890
```

### 6.2 测试代理连通性

```bash
# 测试 HTTP 代理
curl -x http://127.0.0.1:7890 https://www.google.com -I

# 测试环境变量是否生效
curl https://www.google.com -I
```

### 6.3 常见问题

| 问题 | 原因 | 解决方式 |
|---|---|---|
| `curl: (7) Failed to connect` | Mihomo 未运行或端口不对 | `systemctl status mihomo` 检查状态 |
| 能 curl 但 pip/git 不走代理 | 环境变量未生效 | 确认 `source ~/.bashrc` 或重新登录 |
| 国内源变慢 | 国内流量也走了代理 | 检查 `config.yaml` 规则，确保 `GEOIP,CN,DIRECT` |
| 服务启动后立即退出 | 配置文件语法错误 | `mihomo -d /etc/mihomo -t` 测试配置 |

## 7. 节点选择脚本

在无图形界面的服务器上，切换代理节点需要通过 Mihomo 的 RESTful API（默认监听 `127.0.0.1:9090`）来完成。以下是两个实用脚本的编写参考。

### 7.1 API 基础

Mihomo 提供了完整的 RESTful API 用于运行时管理：

| 接口 | 方法 | 用途 |
|---|---|---|
| `/proxies` | GET | 获取所有代理节点信息（名称、类型、延迟历史等） |
| `/proxies/{group}` | PUT | 切换指定代理组的活动节点 |
| `/proxies/{node}/delay` | GET | 测试指定节点的延迟 |

### 7.2 列出所有节点及延迟（list_proxies.py）

该脚本从 API 获取所有代理节点，过滤掉内置的策略组类型（Selector、URLTest、Fallback 等），只保留实际的代理节点，并按延迟排序输出：

```python
import json, sys

data = json.load(sys.stdin)
results = []
for name, info in data['proxies'].items():
    # 过滤掉策略组和内置类型，只保留实际代理节点
    if info.get('type') not in (
        'Selector', 'URLTest', 'Fallback', 'LoadBalance',
        'Direct', 'Reject', 'Compatible', 'Pass'
    ):
        history = info.get('history', [])
        delay = history[-1].get('delay', 0) if history else 0
        results.append((delay, name))
results.sort()
for delay, name in results:
    print(f'{delay:>6}ms  {name}')
```

使用方式：

```bash
curl -s http://127.0.0.1:9090/proxies | python3 list_proxies.py
```

输出示例：

```
   120ms  香港-节点A
   185ms  日本-节点B
   230ms  美国-节点C
     0ms  新加坡-节点D    # 0ms 表示未测速
```

### 7.3 交互式切换节点（switch_proxy.py）

该脚本整合了列出节点、显示当前节点和交互切换三个功能。核心逻辑：

1. **GET `/proxies`** 获取所有节点和当前选中的节点
2. 过滤并按延迟排序后展示列表
3. 用户输入序号后，**PUT `/proxies/{group}`** 切换到目标节点

```python
import json, sys, urllib.request, urllib.parse

API = "http://127.0.0.1:9090"
GROUP = "你的代理组名称"      # 对应 config.yaml 中 proxy-groups 的 name

def get_proxies():
    """获取所有代理信息"""
    with urllib.request.urlopen(f"{API}/proxies") as r:
        return json.load(r)

def list_nodes():
    """列出所有实际代理节点，按延迟排序"""
    data = get_proxies()
    results = []
    for name, info in data['proxies'].items():
        if info.get('type') not in (
            'Selector', 'URLTest', 'Fallback', 'LoadBalance',
            'Direct', 'Reject', 'Compatible', 'Pass', 'REJECT-DROP'
        ):
            history = info.get('history', [])
            delay = history[-1].get('delay', 0) if history else 0
            results.append((delay, name))
    results.sort()
    return results

def get_current():
    """获取当前代理组选中的节点"""
    data = get_proxies()
    group = data['proxies'].get(GROUP, {})
    return group.get('now', '未知')

def switch_node(name):
    """切换代理组的活动节点"""
    encoded = urllib.parse.quote(GROUP)
    url = f"{API}/proxies/{encoded}"
    body = json.dumps({"name": name}).encode()
    req = urllib.request.Request(
        url, data=body, method='PUT',
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return r.status == 204

def main():
    current = get_current()
    print(f"当前节点: {current}\n")

    nodes = list_nodes()
    for i, (delay, name) in enumerate(nodes):
        tag = "(当前)" if name == current else ""
        ms = f"{delay}ms" if delay else "N/A"
        print(f"  {i+1:>3}. [{ms:>7}]  {name} {tag}")

    print("\n输入序号切换节点，直接回车退出：", end=" ")
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not choice:
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(nodes):
            if switch_node(nodes[idx][1]):
                print(f"已切换到: {nodes[idx][1]}")
            else:
                print("切换失败")
        else:
            print("序号超出范围")
    except ValueError:
        print("请输入有效序号")

if __name__ == "__main__":
    main()
```

使用方式：

```bash
python3 switch_proxy.py
```

### 7.4 脚本设计要点

- **只使用标准库**（`json`、`urllib`），无需额外安装依赖，适合在干净的服务器环境中直接运行
- **过滤策略组类型**：API 返回的 `/proxies` 包含策略组（Selector、URLTest 等）和实际节点，脚本通过 `type` 字段过滤，只展示可选择的代理节点
- **GROUP 变量**：必须与 `config.yaml` 中 `proxy-groups` 里类型为 `select` 的组名完全一致（包括 emoji），否则 PUT 请求会返回 404
- **URL 编码**：代理组名称通常包含中文和 emoji，PUT 请求时必须用 `urllib.parse.quote()` 编码
- **延迟为 0 表示未测速**：需要先在 API 上触发测速（`GET /proxies/{node}/delay?url=http://www.gstatic.com/generate_204&timeout=5000`）才能获得真实延迟数据
