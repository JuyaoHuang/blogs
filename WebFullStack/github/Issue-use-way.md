---
title: "Issue Writing"
published: 2025-12-26
description: "GitHub Issue 的编写格式和要求"
first_level_category: "Web全栈开发"
second_level_category: "GitHub"
tags: ['GitHub']
draft: false
---

一个优秀的 GitHub Issue 应该具备**清晰的上下文、明确的执行目标和可追踪的进度**。它不仅是给别人看的，也是给自己梳理思路的过程。

## 1. Issue 编写的标准通用格式

虽然 Bug Report 和 Feature Request 的模板略有不同，但一个标准、高质量的 Issue 通常包含以下模块：

### 标题 (Title)
*   格式：`[类型] 简短描述`
*   示例：`[Bug] 登录页面在深色模式下显示异常` 或 `[Refactor] 简化用户认证逻辑`

### 正文结构 (Body)

```markdown
## 背景 / 动机 (Motivation)
简要说明为什么要提这个 Issue。
- 如果是重构：说明当前架构的痛点。
- 如果是 Bug：说明预期行为和实际行为的差异。
- 如果是新功能：说明解决了什么用户痛点。

## 详细描述 / 方案 (Description / Proposal)
详细描述你想做什么，或者建议的解决方案。
- 涉及到文件变动的位置。
- 预期的目录结构变化。
- 核心逻辑的调整点。

## 任务清单 (Task List) - *最重要部分*
使用 Markdown 的 Checkbox 语法，方便后续勾选进度。
- [ ] 步骤一
- [ ] 步骤二

## 补充信息 (Additional Context)
- 截图、日志报错。
- 关联的其他 Issue 或 PR（使用 # 引用）。
```

## 2. Issue 标签的选择原则

标签是为了快速筛选和分类。一个混乱的标签系统比没有标签更糟糕。

> [Issue Label 的选择详情查看这篇文章](https://www.juayohuang.top/posts/webfullstack/github/label-create-priciple)

**核心分类原则**：

1.  类型：*必选，通常用不同颜色区分*
    *   `bug` (🔴红): 错误修复
    *   `feat` / `feature` (🟢绿): 新功能
    *   `refactor` (🔵蓝): 代码重构（不改变外部行为）
    *   `chore` / `maintenance` (⚪灰): 杂项（构建过程、依赖更新）
    *   `docs` (🟡黄): 文档修改

2.  优先级 (Priority)：*可选*
    *   `priority: high` (急): 阻碍主流程，需立即处理。
    *   `priority: medium`: 正常排期。
    *   `priority: low`: 有空再做。

3.  状态 (Status)：*可选，用于看板管理*
    *   `wip`: 正在进行中 (Work In Progress)。
    *   `help wanted`: 需要社区帮助。
    *   `good first issue`: 适合新手。

4.  范围 (Scope)：*大型项目才需要*
    `frontend`, `backend`, `ui`, `database`

## 3. 示例

**场景**：你要重构仓库代码结构，将 `linglong/` 目录下的内容搬到根目录，并删除 `nginx` 和 `docker` 部署文件。

**标题**：`[Refactor] 扁平化项目目录结构并清理部署配置`

**内容**：

```markdown
## 背景 (Motivation)
当前项目核心代码被嵌套在 `linglong/` 目录下，导致层级过深，使得 import 路径冗长且不直观。
同时，仓库中现存的 `nginx` 和 `docker` 配置已过时（现在使用 K8s/Vercel 部署），这些文件容易误导新加入的开发者。

## 变更目标 (Goals)
1. 提升项目结构的直观性，符合常规的标准项目结构。
2. 移除死代码和废弃配置，保持仓库整洁。

## 变更详情 (Changes)
1. **目录调整**：将 `linglong/*` 的所有内容移动到项目根目录 `./`，并删除空的 `linglong` 文件夹。
2. **清理配置**：删除根目录下的 `nginx/` 文件夹及相关配置文件。
3. **清理配置**：删除 `docker/` 文件夹及 `Dockerfile`、`docker-compose.yml`。

## 任务清单 (Tasks)
- [ ] 删除 `nginx` 相关配置文件及目录
- [ ] 删除 `Dockerfile` 和 `docker-compose.yml`
- [ ] 将 `linglong/src` 移动至 `./src`
- [ ] 将 `linglong/public` 移动至 `./public`
- [ ] 将 `linglong/package.json` 移动至根目录，并与根目录现有配置（如有）合并
- [ ] 删除空的 `linglong/` 目录
- [ ] **关键**：更新项目中的 import 引用路径（如 `@/linglong/...` -> `@/...`）
- [ ] 更新 `README.md` 中的项目启动文档

## 关联 (Related)
无
```

Labels：
*   refactor
*   chore
*   priority: medium

> 为什么要写这么细：
>
> 哪怕是你自己写给自己看的，写出 “任务清单” 中的 “更新 import 路径” 和 “合并 package.json” 这两点至关重要。因为在移动目录时，往往最容易忘记修复路径引用，导致项目跑不起来。写下来，就是为了防止这些疏漏。

*其实一般自己做的都会忽略任务清单一栏。*
