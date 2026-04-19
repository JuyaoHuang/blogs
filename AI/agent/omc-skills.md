---
title: "OMC Skills 总览"
publishDate: 2026-04-19
description: "cc配套 oh-my-claude 内置 skills 介绍"
tags: ['claude', 'omc']
language: 'Chinese'
first_level_category: "AI"
second_level_category: "Agent"
draft: false
---

## OMC Skills 总览 (`/omc-*`)

按用途分组，常用度标注：**★★★ 高频** / **★★ 偶用** / **★ 低频**

### 🚀 自主执行类（最核心）

| **指令**                 | **说明**                                 | **适用场景**                       |
| ------------------------ | ---------------------------------------- | ---------------------------------- |
| **`/omc-autopilot` ★★★** | 从想法到可运行代码的端到端自主执行       | "给我做个 XX 功能"——一次说清就走   |
| **`/omc-ralph` ★★★**     | 自指循环，直到 architect 验证通过才停    | 长任务、易漏边缘用例、需要反复打磨 |
| **`/omc-ralph-init` ★★** | 先生成 PRD（产品需求文档），再喂给 ralph | 复杂项目起步，避免 ralph 跑偏      |
| **`/omc-ultrawork` ★★**  | 并行高吞吐任务执行引擎                   | 一堆独立小任务同时推进             |
| **`/omc-ultraqa` ★★**    | 测试→验证→修→再测，循环直到达标          | bug 收敛、回归清零                 |


### 🧠 规划 / 分析类

| **指令**                     | **说明**                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| **`/omc-plan` ★★★**          | 战略规划，可带 interview 问询流                              |
| **`/omc-ralplan` ★★**        | `/omc-plan --consensus` 的别名，多模型共识                   |
| **`/omc-deep-interview` ★★** | 苏格拉底式追问，消除歧义后再执行                             |
| **`/omc-analyze` ★★**        | 深度分析与调研                                               |
| **`/omc-ccg` ★**             | Claude + Codex + Gemini 三模型编排，后端给 Codex、前端给 Gemini |


### 🔍 审查 / 修复类

| **指令**                      | **说明**                   |
| ----------------------------- | -------------------------- |
| **`/omc-code-review` ★★★**    | 综合代码审查               |
| **`/omc-security-review` ★★** | 安全审查                   |
| **`/omc-build-fix` ★★**       | 最小变更修 build / TS 错误 |
| **`/omc-tdd` ★**              | 强制 TDD 流程，先写测试    |


### 👥 多 Agent 协作类

| **指令**                      | **说明**                                          |
| ----------------------------- | ------------------------------------------------- |
| **`/omc-team` ★★**            | N 个协同 agent 共享任务列表（原生 teams）         |
| **`/omc-omc-teams` ★**        | 在 tmux 窗格里拉起 claude/codex/gemini CLI worker |
| **`/omc-sciomc` ★**           | 并行 scientist agent 做全面分析（AUTO 模式）      |
| **`/omc-external-context` ★** | 并行跑 document-specialist 做网搜/查文档          |


### 🧰 工程辅助类

| **指令**                             | **说明**                             |
| ------------------------------------ | ------------------------------------ |
| **`/omc-note` ★★★**                  | 存笔记到 notepad.md，防压缩丢失      |
| **`/omc-trace` ★★**                  | 查看 agent flow 时间线与汇总         |
| **`/omc-deepinit` ★**                | 深度初始化，生成分层 AGENTS.md 文档  |
| **`/omc-project-session-manager` ★** | git worktree + tmux 管理隔离开发环境 |
| **`/omc-learner` ★**                 | 从当前会话提取一个可复用 skill       |


### ⚙️ 配置类（装好后基本不用动）

```bash
/omc-setup`（唯一必学）、`/omc-doctor`、`/omc-omc-help`、`/omc-mcp-setup`、`/omc-configure-notifications`、`/omc-configure-openclaw`、`/omc-hud`、`/omc-skill`、`/omc-release`、`/omc-cancel
```

## 两个明星 skill 详解

**`/omc-autopilot` —— "说完就走"**

- 本质：Opus 级 deep-executor，单次完整执行任务。

- 典型用法：

  > `/omc-autopilot 把 experiment1.py 的三阶段法改成支持 M/G/1 队列，并加对应单测`

- 何时用：你对需求有把握、想一把过，不需要中间停下来对齐。

- 何时不用：需求模糊时——会朝你没想到的方向跑偏。这种情况先 `/omc-plan` 或 `/omc-deep-interview`。

**`/omc-ralph` —— "跑到 architect 说过为止"**

- 本质：自指循环，执行 → architect 审核 → 不通过就再来一轮。

- 典型用法：

  > /omc-ralph-init    # 先生成 PRD
  >
  > /omc-ralph         # 基于 PRD 循环到通过

- 何时用：边缘用例多、容易漏测、一次写不对的任务（如重构、迁移、协议实现）。

- 何时不用：改一行 typo、看个日志——杀鸡用牛刀，而且烧 token。

**两者区别一句话**

autopilot 是"一次性冲线"，ralph 是"反复打磨直到 architect 点头"。

不确定选哪个时：小/中任务用 **autopilot**，大/关键任务用 **ralph**。

## 审查类 skills

### 代码 / 质量审查

| **指令**                         | **定位**                        | **适用**                  |
| -------------------------------- | ------------------------------- | ------------------------- |
| **`/omc-code-review` ★★★**       | 综合代码审查，按严重度分级反馈  | PR 前自查、改完大段代码后 |
| **`/review` ★★**                 | Claude Code 内建——审查一个 PR   | 直接对着远程 PR 用        |
| **`/code-review:code-review` ★** | code-review 插件版，同样是审 PR | 有装该插件时              |


### 安全审查

| **指令**                      | **定位**                     | **适用**                           |
| ----------------------------- | ---------------------------- | ---------------------------------- |
| **`/omc-security-review` ★★** | 综合安全审查                 | 改了认证、鉴权、输入处理等敏感代码 |
| **`/security-review` ★★**     | 内建——审当前分支的待提交改动 | 提交前扫一遍                       |


### 专项验证 / QA

| **指令**                               | **定位**                                                    |
| -------------------------------------- | ----------------------------------------------------------- |
| **`/omc-ultraqa` ★★**                  | 测试→验证→修→循环直到达标（偏执行，不只是审）               |
| **`/requesting-code-review（skill）`** | 在完成任务、合并前，核验工作是否满足要求                    |
| **`/web-design-guidelines（skill）`**  | 按 Web Interface Guidelines 审 UI 代码                      |
| **`/audit-website（skill）`**          | 用 squirrelscan CLI 审线上站点：SEO、性能、安全等 230+ 规则 |


### 对应的 agent（可通过 Agent 工具显式调用）

- **`oh-my-claudecode:code-reviewer`** —— 严重度分级的代码审查专家
- **`oh-my-claudecode:security-reviewer`** —— OWASP Top 10 / 密钥泄露 / 不安全模式
- **`oh-my-claudecode:quality-reviewer`** —— 逻辑缺陷、可维护性、反模式、SOLID
- **`oh-my-claudecode:harsh-critic`** —— 多视角严苛审查（Opus，结构化 gap 分析）
- **`oh-my-claudecode:critic`** —— 工作计划评审与批评（Opus）
- **`oh-my-claudecode:verifier`** —— 验证策略、证据化完成度检查


### 选哪个？

- 改完一段代码想快速过一遍：`/omc-code-review`
- 涉及安全：`/omc-security-review` 或 `/security-review`
- 想要狠人式审视：调用 `harsh-critic` agent
- 方案/计划层面（还没写代码）：`critic` agent

