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

---

## 核心技能详解

### 1. Autopilot - 全自主执行

**定位**：从想法到可运行代码的端到端自主执行

#### 使用场景
- ✅ "给我做个 XX 功能" —— 一次说清就走
- ✅ 需要完整的开发流程（规划 → 实现 → 测试 → 验证）
- ✅ 想要生产就绪的代码（不只是能跑，而是经过验证的）
- ✅ 不想手动迭代修 bug

#### 工作流程（5 阶段）

```
[扩展] → [规划] → [执行] → [QA] → [验证]
  ↓        ↓        ↓       ↓       ↓
analyst  architect  3x     qa-    architect
         + critic  executor tester + security
                                   + reviewer
```

**Phase 1: 扩展（Expansion）** - 0:00-0:30
- analyst agent 自动创建详细规格
- 需求扩展：数据模型、路由、验证规则、测试策略

**Phase 2: 规划（Planning）** - 0:30-1:30
- architect agent 设计系统架构
- critic agent 验证设计
- 创建文件结构：`src/`、`tests/`、`package.json`

**Phase 3: 执行（Execution）** - 1:30-3:30
- 多个 executor agents 并行工作
- 一个处理模型，一个处理路由，一个写测试
- 依赖自动安装

**Phase 4: QA 循环** - 3:30-4:30
- build-fixer 运行 TypeScript 编译
- qa-tester 运行测试套件
- 发现错误 → 自动修正 → 重新运行

**Phase 5: 验证（Validation）** - 4:30-5:00
- architect 验证实现是否符合规格
- security-reviewer 检查漏洞
- code-reviewer 验证代码质量

#### 示例

```bash
# 基本用法
autopilot: build a REST API for a bookstore inventory with CRUD operations for books

# 带约束
autopilot: build a REST API using Go and PostgreSQL

# 预期输出
bookstore-api/
├── package.json
├── tsconfig.json
├── src/
│   ├── models/Book.ts
│   ├── routes/books.ts
│   ├── middleware/validation.ts
│   └── app.ts
├── tests/
│   └── books.test.ts
└── .omc/
    └── plans/autopilot-bookstore-api.md
```

#### 关键特性
- **零手动步骤**：一条命令从想法到可运行代码
- **多阶段工作流**：扩展 → 规划 → 执行 → QA → 验证
- **内嵌并行**：多个 agents 同时工作
- **自我修正**：自动修复错误直到测试通过
- **生产就绪**：不只是"能跑"，而是完全验证过的

---

### 2. Ralph - 持久执行直到完成

**定位**：自指循环，直到 architect 验证通过才停

#### 使用场景
- ✅ 复杂重构（可能遇到边缘情况）
- ✅ 技术栈迁移（数据库、架构）
- ✅ 关键功能（必须工作）
- ✅ 任务有未知障碍
- ❌ 简单的单次任务（直接执行更快）

#### Ralph 循环

```
1. 尝试任务
2. 遇到错误？→ 诊断
3. 应用修复
4. 从步骤 1 重试
5. 成功？→ 请求 architect 验证
6. Architect 批准？→ 完成
7. Architect 拒绝？→ 回到步骤 1
```

#### 工作流程示例

```bash
ralph: refactor auth.js to use TypeScript, JWT tokens, bcrypt password hashing

# 执行过程
[RALPH ITERATION 1]
✗ Error: Missing @types/bcrypt
Ralph: Self-correcting...

[RALPH ITERATION 2]
✗ Error: JWT_SECRET not defined
Ralph: Self-correcting...

[RALPH ITERATION 3]
✓ Refactoring complete
Ralph: Requesting architect verification...

[ARCHITECT VERIFICATION]
✓ TypeScript compilation: No errors
✓ JWT implementation: Correct
✓ Bcrypt hashing: Proper work factor
✓ Security: No plain text passwords
Architect verdict: ✓ APPROVED

[RALPH COMPLETE]
Summary:
  Ralph iterations: 3
  Errors encountered: 2
  Errors auto-fixed: 2
  Total time: 2m 15s
```

#### 关键特性
- **永不放弃**：错误触发自我修正，而非失败
- **自我修正循环**：错误 → 诊断 → 修复 → 重试（自动）
- **Architect 验证**：不经批准不会声称完成
- **迭代跟踪**：显示过程，而非只显示结果
- **复杂任务处理**：完美适用于重构、迁移、多步骤工作

#### Ralph vs Autopilot

| 特性 | Autopilot | Ralph |
|------|-----------|-------|
| **定位** | 完整工作流（想法 → 代码） | 持久层（保证完成） |
| **包含** | 规划 + 执行 + QA | 错误恢复 + 验证循环 |
| **适用** | 新功能开发 | 复杂重构/迁移 |
| **组合** | - | `ralph autopilot` |

---

### 3. Ralplan - 共识规划

**定位**：`/omc-plan --consensus` 的别名，多 agent 共识规划

#### 使用场景
- ✅ 复杂项目起步（避免执行跑偏）
- ✅ 需求不明确（需要先澄清）
- ✅ 高风险工作（认证/安全、迁移、生产事故）
- ✅ 想在执行前审查计划

#### 工作流程

```
0. [可选] 公司上下文调用
1. Planner 创建初始计划 + RALPLAN-DR 摘要
2. [--interactive] 用户反馈草稿
3. Architect 审查架构合理性（必须提供最强反论）
4. Critic 评估质量标准
5. 重审循环（最多 5 次迭代）
6. [--interactive] 用户批准
7. 执行：team（推荐）或 ralph
```

#### RALPLAN-DR 摘要结构

```markdown
- Principles (3-5 条原则)
- Decision Drivers (前 3 个决策驱动因素)
- Viable Options (>=2 个可行选项，带优缺点)
- [Deliberate 模式] Pre-mortem (3 个场景) + 扩展测试计划
```

#### 标志

```bash
# 基本用法
/ralplan "task description"

# 交互模式（在关键决策点提示用户）
/ralplan --interactive "task description"

# 审慎模式（高风险工作）
/ralplan --deliberate "migrate auth to OAuth2"

# 使用 Codex 做 Architect 审查
/ralplan --architect codex "task description"
```

#### 预执行门控

**问题**：执行模式（ralph、autopilot、team）在模糊请求上浪费资源

**解决**：ralplan 拦截未明确的执行请求，重定向到共识规划

**通过门控**（足够具体）：
- `ralph fix src/hooks/bridge.ts:326`
- `autopilot implement issue #42`
- `team add validation to processKeywordDetector`

**被门控**（需要先规划）：
- `ralph fix this`
- `autopilot build the app`
- `team improve performance`

**绕过门控**：
- `force: ralph refactor auth`
- `! autopilot optimize everything`

---

### 4. Team - 协调式团队编排

**定位**：N 个协同 agents 共享任务列表（Claude Code 原生 teams）

#### 使用场景
- ✅ 独立并行任务（多个子任务可独立执行）
- ✅ 需要协调开销的复杂项目
- ✅ 明确的任务分工
- ❌ 简单单线性任务
- ❌ 任务间强依赖（需串行）

#### 团队流水线

```
team-plan → team-prd → team-exec → team-verify → team-fix (循环)
```

**各阶段说明**：
1. **team-plan**：规划任务分解和 agent 分工
2. **team-prd**：生成详细的产品需求文档
3. **team-exec**：各 agent 并行执行任务
4. **team-verify**：验证执行结果
5. **team-fix**：修复问题（有界的修复循环）

#### 使用方式

```bash
# 基本用法：N 个 agent 协同工作
/team 5:executor "fix all TypeScript errors across the project"

# 自动确定 agent 数量
/team "refactor the auth module with security review"

# 结合 Ralph（串行验证）
/team ralph "build a complete REST API for user management"

# 使用 Codex CLI workers
/team 2:codex "review architecture and suggest improvements"

# 使用 Gemini CLI workers
/team 2:gemini "redesign the UI components"
```

#### 架构

```
User: "/team 3:executor fix all TypeScript errors"
              |
              v
      [TEAM ORCHESTRATOR (Lead)]
              |
              +-- TeamCreate("fix-ts-errors")
              +-- 分析 & 分解任务
              +-- TaskCreate x N
              +-- Task(team_name, name) x 3 (spawn teammates)
              +-- 监控循环
              +-- 完成 & 清理
```

#### 阶段 Agent 路由

| 阶段 | 必需 Agents | 可选 Agents | 选择标准 |
|------|------------|-------------|---------|
| **team-plan** | explore (haiku), planner (opus) | analyst, architect | 需求不清用 analyst，复杂边界用 architect |
| **team-prd** | analyst (opus) | critic (opus) | 用 critic 挑战范围 |
| **team-exec** | executor (sonnet) | debugger, designer, writer, test-engineer | 根据子任务类型匹配 agent |
| **team-verify** | verifier (sonnet) | security-reviewer, code-reviewer | 认证/加密改动加 security-reviewer |
| **team-fix** | executor (sonnet) | debugger, executor (opus) | 类型/构建错误用 debugger |

#### Team + Ralph 组合

```bash
/team ralph "build a complete REST API"
```

**提供**：
- **Team 编排**：多 agent 分阶段流水线
- **Ralph 持久**：失败重试，architect 验证，迭代跟踪

**执行流程**：
1. Ralph 外循环开始（迭代 1）
2. Team 流水线运行：plan → prd → exec → verify
3. 如果 verify 通过：Ralph 运行 architect 验证
4. 如果 architect 批准：两种模式完成
5. 如果 verify 失败或 architect 拒绝：进入 team-fix，循环回 exec → verify
6. 如果修复循环超过 max_fix_loops：Ralph 增加迭代，重试完整流水线

---

### 5. Ultrawork - 并行执行引擎

**定位**：高吞吐量并行任务执行

#### 使用场景
- ✅ 多个独立任务可同时运行
- ✅ 用户说 "ulw"、"ultrawork" 或想要并行执行
- ✅ 需要同时委托工作给多个 agents
- ❌ 需要保证完成和验证（用 ralph）
- ❌ 需要完整自主流水线（用 autopilot）
- ❌ 只有一个串行任务（直接委托给 executor）

#### 执行策略

```bash
# 三个独立任务同时触发
Task(subagent_type="oh-my-claudecode:executor", model="haiku", 
     prompt="Add missing type export for Config interface")
Task(subagent_type="oh-my-claudecode:executor", model="sonnet", 
     prompt="Implement the /api/users endpoint with validation")
Task(subagent_type="oh-my-claudecode:executor", model="sonnet", 
     prompt="Add integration tests for the auth middleware")
```

#### 模型路由

| 任务复杂度 | 模型层级 | 适用场景 |
|-----------|---------|---------|
| **简单** | LOW (Haiku) | 简单查找/定义、添加类型导出 |
| **标准** | MEDIUM (Sonnet) | 标准实现、API 端点、测试 |
| **复杂** | HIGH (Opus) | 复杂分析/重构、架构设计 |

#### 与其他模式的关系

```
autopilot (自主执行)
 └── includes: ralph (持久)
     └── includes: ultrawork (并行)
         └── provides: 仅并行执行
```

**Ultrawork 是并行层**。Ralph 添加持久和验证。Autopilot 添加完整生命周期流水线。

---

### 6. UltraQA - QA 循环工作流

**定位**：测试 → 验证 → 修复 → 重复，直到达标

#### 使用场景
- ✅ Bug 收敛、回归清零
- ✅ 需要循环直到所有测试通过
- ✅ 构建/lint/类型检查错误修复

#### 循环工作流

```
Cycle N (最多 5 次)
1. 运行 QA：执行验证
2. 检查结果：通过？→ 退出；失败？→ 继续
3. Architect 诊断：分析失败原因
4. 修复问题：应用 architect 的建议
5. 重复：回到步骤 1
```

#### 使用方式

```bash
# 运行测试直到全部通过
/ultraqa --tests

# 修复构建错误
/ultraqa --build

# 修复 lint 错误
/ultraqa --lint

# 修复 TypeScript 错误
/ultraqa --typecheck

# 自定义模式
/ultraqa --custom "pattern"

# 交互式 CLI/服务测试
/ultraqa --interactive
```

#### 退出条件

| 条件 | 动作 |
|------|------|
| **目标达成** | 退出并成功："ULTRAQA COMPLETE: Goal met after N cycles" |
| **达到第 5 次循环** | 退出并诊断："ULTRAQA STOPPED: Max cycles. Diagnosis: ..." |
| **相同失败 3 次** | 提前退出："ULTRAQA STOPPED: Same failure detected 3 times" |
| **环境错误** | 退出："ULTRAQA ERROR: [tmux/port/dependency issue]" |

---

## 技能组合使用推荐

### 组合 1：Ralph + Ultrawork（速度 + 可靠性）

```bash
ralph ulw: refactor all auth modules to TypeScript
```

**适用**：多个独立模块需要重构，既要并行速度，又要保证完成

**工作方式**：
- Ultrawork 并行执行多个重构任务
- Ralph 确保每个任务都完成并通过验证
- 遇到错误自动修复并重试

---

### 组合 2：Team + Ralph（团队 + 持久）

```bash
/team ralph "build a complete REST API for user management"
```

**适用**：复杂功能开发，需要多 agent 协作 + 保证完成

**工作方式**：
- Team 分阶段流水线：plan → prd → exec → verify → fix
- Ralph 外循环确保直到 architect 验证通过才停
- 失败自动重试，不会半途而废

---

### 组合 3：Ralplan → Team（规划 + 执行）

```bash
# 第一步：共识规划
/ralplan --interactive "build user authentication system"

# 第二步：用户批准后，选择 team 执行
# ralplan 会自动提示选择执行方式
```

**适用**：复杂项目，需要先明确需求和架构，再并行执行

**工作方式**：
- Ralplan 先通过 Planner + Architect + Critic 达成共识
- 生成详细 PRD 和架构设计
- 用户批准后，Team 按计划并行执行

---

### 组合 4：Autopilot（全包）

```bash
autopilot: build a REST API for bookstore inventory with CRUD operations
```

**适用**：想要一条命令搞定一切，从想法到生产就绪代码

**工作方式**：
- 内部包含 Ralph（持久）和 Ultrawork（并行）
- 自动完成：扩展 → 规划 → 执行 → QA → 验证
- 无需手动干预

---

## 实战示例

### 示例 1：修复项目中所有 TypeScript 错误

**场景**：项目有 50+ 个 TypeScript 错误，分布在多个文件

**方案**：

```bash
# 方案 A：快速并行修复（无保证）
/team 5:executor "fix all TypeScript errors across the project"

# 方案 B：并行 + 保证完成（推荐）
/team ralph "fix all TypeScript errors and ensure all pass tsc --noEmit"

# 方案 C：循环修复直到全部通过
/ultraqa --typecheck
```

**对比**：
- **方案 A**：最快，但可能遗漏边缘情况
- **方案 B**：速度 + 可靠性平衡，推荐
- **方案 C**：最可靠，但串行执行较慢

---

### 示例 2：重构遗留认证系统

**场景**：将 JavaScript 认证代码迁移到 TypeScript + JWT + bcrypt

**方案**：

```bash
# 方案 A：持久执行（推荐）
ralph: refactor auth.js to use TypeScript, JWT tokens, bcrypt password hashing, and proper error handling

# 方案 B：先规划再执行（复杂项目）
/ralplan --interactive "migrate auth to TypeScript + JWT + bcrypt"
# 批准后选择 ralph 执行

# 方案 C：全自主（包含规划）
autopilot: migrate auth system to modern TypeScript with JWT and bcrypt
```

**对比**：
- **方案 A**：直接执行，适合需求明确的情况
- **方案 B**：先规划后执行，适合复杂迁移
- **方案 C**：完全自主，适合"一次说清就走"

---

### 示例 3：构建完整的用户管理 REST API

**场景**：从零开始构建用户管理系统（CRUD + 认证 + 测试）

**方案**：

```bash
# 方案 A：全自主（推荐新项目）
autopilot: build a complete REST API for user management with authentication, CRUD operations, validation, and tests

# 方案 B：团队协作 + 持久（复杂项目）
/team ralph "build user management REST API with auth, CRUD, validation, tests"

# 方案 C：先规划再团队执行（高风险项目）
/ralplan --interactive --deliberate "build production-ready user management API"
# 批准后选择 team 执行
```

**对比**：
- **方案 A**：最简单，一条命令搞定
- **方案 B**：并行 + 持久，适合多模块项目
- **方案 C**：最严谨，适合生产环境

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



