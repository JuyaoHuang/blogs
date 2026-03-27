---
title: 'Superpowers 技能框架：各 Skills 作用与使用边界'
publishDate: 2026-03-25
description: "面向 Codex 的 Superpowers skills 速查，逐项说明每个 skill 的作用、触发时机与典型用途。"
tags: ['agent', 'superpowers','codex']
language: 'Chinese'
first_level_category: "AI"
second_level_category: "agent"
draft: false
---

## 1. 使用前提

Superpowers 在 Codex 中是“按描述自动匹配”的技能框架，不是手动插件面板。

- 扫描目录：`~/.agents/skills/`
- 当前软限制：只有你明确写出 `Use SuperPowers` 才进入该框架
- 子代理并行类 skill 需要配置：

```toml
[features]
multi_agent = true
```

**重点**：你现在的配置是“默认普通模式，口令切换 Superpowers 模式”。

## 2. Skills 作用总览（逐项）

### 2.1 `using-superpowers`

- 作用：入口控制 skill，决定是否加载整套 Superpowers 工作流。 
- 触发时机：你明确输入 `Use SuperPowers`。  
- 典型用途：需要流程化开发，而不是一次性小改动时。

### 2.2 `brainstorming`

- 作用：在写代码前澄清需求、边界和方案。  
- 触发时机：新功能、较大改动、需求仍模糊。  
- 典型用途：先把“要做什么”说清楚，减少返工。

### 2.3 `writing-plans`

- 作用：把需求拆成可执行任务清单。  

- 触发时机：任务是多步骤、跨文件、可并行。  
- 典型用途：生成“按步骤执行 + 每步可验证”的计划。

### 2.4 `using-git-worktrees`

- 作用：为新任务创建隔离工作区，避免污染当前目录。  

- 触发时机：开始一个独立功能或修复分支。  
- 典型用途：并行开发、隔离试验、降低回滚成本。

### 2.5 `executing-plans`

- 作用：按既定计划分批执行，带阶段检查点。  

- 触发时机：计划已经批准，进入落地阶段。  
- 典型用途：避免边写边漂移，确保执行不偏题。

### 2.6 `subagent-driven-development`

- 作用：按计划把任务分给子代理执行并回收结果。  

- 触发时机：任务块独立、可并行推进。  
- 典型用途：提高吞吐量，缩短整体交付时间。

### 2.7 `dispatching-parallel-agents`

- 作用：并行派工模板，专门处理“多个独立子任务”。  

- 触发时机：至少 2 个任务互不依赖。  
- 典型用途：文档、测试、实现等分线并发处理。

### 2.8 `test-driven-development`

- 作用：强制 RED-GREEN-REFACTOR，先测后写。  

- 触发时机：实现新功能或修 bug。  
- 典型用途：防止“代码先写完再补测试”导致的质量回退。

### 2.9 `systematic-debugging`

- 作用：用固定步骤定位根因，而不是猜测式修复。  

- 触发时机：出现 bug、测试失败、行为异常。  
- 典型用途：先定位根因再动代码，减少误修。

### 2.10 `requesting-code-review`

- 作用：主动发起结构化代码审查。  

- 触发时机：阶段完成、准备合并、关键功能落地后。  
- 典型用途：提前暴露风险，避免把问题带进主分支。

### 2.11 `receiving-code-review`

- 作用：规范处理 review 意见，先验证再修改。  

- 触发时机：收到审查反馈后。  
- 典型用途：避免机械接受建议，保持技术判断。

### 2.12 `verification-before-completion`

- 作用：在“宣布完成”前做强制验证。  

- 触发时机：准备说“已修复/已完成/可合并”之前。  
- 典型用途：把“感觉对”变成“结果对”。

### 2.13 `finishing-a-development-branch`

- 作用：收尾决策流程（merge/PR/保留/清理）。  

- 触发时机：实现和验证都完成后。  
- 典型用途：统一收口动作，避免分支悬空。

### 2.14 `writing-skills`

- 作用：编写或维护自定义 skill。  

- 触发时机：你要扩展这套框架时。  
- 典型用途：把团队最佳实践沉淀为可复用技能。

## 3. 快速使用建议

简单脚本、单文件小改、一次性任务：不输入 `Use SuperPowers`，走普通模式。  跨文件改造、需求不稳定、需要可追溯验证：输入 `Use SuperPowers`，走流程模式。

**核心结论**：Superpowers 不是必须一直开，而是该开时一键开。
