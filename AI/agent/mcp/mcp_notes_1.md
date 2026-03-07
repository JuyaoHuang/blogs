---
title: 'MCP 扩展指南：5 个实用插件与配置教程'
publishDate: 2026-03-07
description: '在使用 Maraya 进行开发时，MCP（Model Context Protocol）扩展可以显著增强 AI 的能力边界。本文将介绍几个最实用的 MCP 插件，帮助你理解它们的作用，并提供详细的配置教程。'
tags: ['mcp', 'agent']
language: 'Chinese'
first_level_category: "AI"
second_level_category: "agent"
draft: false
---

## 什么是 MCP？

MCP 是一种让 AI 助手能够访问外部工具和服务的协议。通过连接不同的 MCP 服务，AI 可以：

- 读写本地文件系统
- 操作 Git 仓库
- 抓取网页内容
- 查询最新技术文档
- 执行系统命令

简单来说，MCP 就像是给 AI 装上了"手"和"眼睛"，让它能真正帮你干活。

---

## 核心 MCP 插件介绍

### 1. Filesystem MCP - 文件系统操作

**核心功能**

Filesystem MCP 让 AI 能够访问和操作你指定目录中的文件。

**典型能力**
- 列出目录内容
- 读取文件
- 创建、修改、删除文件
- 重命名和移动文件
- 查看目录结构

**适用场景**
- 批量整理文档和笔记
- 自动生成项目文件
- 修改配置文件
- 代码重构和文件操作

**安全提示**

建议只授权特定目录的访问权限，避免给整个磁盘权限。

**配置教程**

在 Kiro 的 MCP 配置文件中添加：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "D:\\your\\project\\path"
      ]
    }
  }
}
```

将 `D:\\your\\project\\path` 替换为你想授权的目录路径。

---

### 2. Git MCP - 版本控制助手

**核心功能**

Git MCP 让 AI 能够读取和操作 Git 仓库，帮助你进行版本控制相关的工作。

**典型能力**
- 查看仓库状态
- 查看分支信息
- 查看 diff 和改动
- 查看提交历史
- 可能支持提交、切换分支等操作（取决于具体实现）

**适用场景**
- 查看当前改动了哪些文件
- 生成 commit message
- 解释代码变更
- 检查仓库状态
- 辅助代码审查

**重要前提**

Git MCP 要求目标目录必须是一个有效的 Git 仓库（包含 `.git` 目录）。如果目录不是 Git 仓库，需要先执行：

```bash
git init
```

**配置教程**

```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": [
        "mcp-server-git",
        "--repository",
        "D:\\your\\git\\repo\\path"
      ]
    }
  }
}
```

确保路径指向一个有效的 Git 仓库。

---

### 3. Fetch MCP - 网页内容抓取

**核心功能**

Fetch MCP 让 AI 能够直接抓取网页内容、API 返回结果或远程文本资源。

**典型能力**
- 获取网页 HTML 和文本
- 抓取公开 API 数据
- 读取在线文档内容
- 自动化信息收集

**适用场景**
- 抓取网页并总结内容
- 读取在线技术文档
- 获取 API 返回结果
- 监控网站更新

**局限性**
- 不适合需要登录的网站
- 某些网站会限制抓取
- 对复杂动态网页支持有限

**配置教程**

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ]
    }
  }
}
```

---

### 4. Context7 MCP - 技术文档检索

**核心功能**

Context7 是一个专门面向开发者的文档检索 MCP，能够查询最新的技术文档、库文档和 API 文档。

**典型能力**
- 查询框架和库的最新文档
- 提供 API 用法参考
- 避免 AI 依赖过时信息
- 补充官方文档上下文

**适用场景**
- 查询 React、Next.js、Vue 等框架的最新用法
- 根据最新文档生成示例代码
- 排查"这个 API 现在怎么写"
- 学习新技术栈

**为什么需要它**

AI 的训练数据可能过时，Context7 能提供最新的官方文档信息，确保代码示例和建议是最新的。

**配置教程**

Context7 的配置取决于具体实现，通常需要：

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": [
        "-y",
        "context7-mcp-server"
      ],
      "env": {
        "CONTEXT7_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

部分实现可能需要 API key，请查阅对应服务的文档。

---

### 5. URL Fetcher MCP - 专用网页抓取

**核心功能**

URL Fetcher 是 Fetch MCP 的变体，专注于按 URL 抓取网页内容。

**与 Fetch MCP 的区别**

不同实现可能有不同特性：
- 更好的正文提取能力
- 去除网页噪音
- 支持自定义 headers
- 专门优化 Markdown 提取
- 对 API JSON 返回更友好

**是否需要同时安装**

如果你已经安装了 Fetch MCP，通常不需要再安装 URL Fetcher，除非它有你特别需要的功能。

---

## 推荐的 MCP 组合

### 基础开发组合
```
Filesystem + Git + Fetch
```

这是最实用的组合，覆盖了本地文件操作、版本控制和网络资源获取。

### 进阶开发组合
```
Filesystem + Git + Context7 + Fetch
```

加入 Context7 后，可以获得最新的技术文档支持，特别适合学习新技术或使用快速迭代的框架。

---

## 完整配置示例

以下是一个完整的 MCP 配置文件示例：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "D:\\Projects"
      ]
    },
    "git": {
      "command": "uvx",
      "args": [
        "mcp-server-git",
        "--repository",
        "D:\\Projects\\my-repo"
      ]
    },
    "fetch": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ]
    },
    "context7": {
      "command": "npx",
      "args": [
        "-y",
        "context7-mcp-server"
      ]
    }
  }
}
```

---

## 使用建议

### 安全性
1. **限制 Filesystem 访问范围**：只授权必要的目录
2. **谨慎使用 Git 写操作**：建议先只用读取功能
3. **注意 API 限制**：某些 MCP 可能有调用频率限制

### 性能优化
1. **按需连接**：不是所有 MCP 都需要同时启用
2. **定期检查**：确保 MCP 服务正常运行
3. **合理使用**：避免频繁的大量操作

### 故障排查
1. **检查路径**：确保配置的路径存在且有权限
2. **验证环境**：确保 `npx`、`uvx` 等命令可用
3. **查看日志**：MCP 连接失败时查看错误信息

---

## 总结

MCP 扩展让 Kiro AI 从一个"只会说话"的助手，变成了一个"能动手干活"的开发伙伴。通过合理配置这些 MCP 插件，你可以：

- 让 AI 帮你操作文件和代码
- 让 AI 理解你的 Git 仓库状态
- 让 AI 获取最新的技术文档
- 让 AI 抓取网络资源

选择适合你工作流程的 MCP 组合，让开发效率更上一层楼。