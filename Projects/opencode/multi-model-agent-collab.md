---
title: "Claude Code 多模型 Agent 协作实践：OMO、omc、CCG 三方案深度对比与选型"
publishDate: 2026-03-08
description: "深度对比 opencode OMO、Claude Code omc 与 CCG 三种多模型 Agent 协作架构，从模型多样性、编排精细度、通信效率与代理兼容性四个维度拆解各方案优劣取舍，附详细选型建议，助你快速找到最适合的 AI 编码工具组合。"
tags: ['ai-agent','claude-code', 'opencode', 'llm']
language: 'Chinese'
first_level_category: "项目实践"
second_level_category: "Agent 搭建"
heroImage: { src: './3.png'}
draft: false
---


# 多模型 Agent 协作：从 OMO 到 CCG 的范式比较与反思

> 前两篇分别聊了 opencode 的 SDK 绑定问题和 OMO 的角色编排设计。写到最后我一直在想一个问题：OMO 在 opencode 里通过 provider 系统把不同任务路由给不同模型，这和 cc 里用 `Task(model: "opus/sonnet/haiku")` 编排不同角色，不是一回事吗？再看[风佬的 ccg](https://linux.do/t/topic/1581788)，用跨 CLI 的方式实现多模型协作——三种方案解决的其实是同一个问题：**怎么让最合适的模型干最合适的活**。这篇就来把它们摆一起比比。

## 1. 同一个目标，两条路

不管是 OMO、omc 还是 CCG，做的事情本质上都是"按角色选模型"。区别在于怎么选、能选多广。

**OMO 的做法：框架内路由**

OMO 跑在 opencode 上，opencode 的 provider 系统支持同时配置多个模型供应商（Anthropic、OpenAI、Google 等）。OMO 定义了一套 Category 分类，根据任务性质自动把请求路由到不同模型：

```
用户需求 → Sisyphus 分析任务类型 → Category: "visual-engineering"
  → 路由到 Gemini provider → Gemini 执行前端任务
```

整个过程在同一个工具、同一个进程里完成，agent 之间的通信是原生的。

**omc 的做法：家族内路由**

omc 跑在 cc 上，cc 只有一个 provider（Anthropic）。omc 通过 Task tool 的 `model` 参数在 Claude 家族内做选择：

```
用户需求 → planner(model: "opus") 规划
  → executor(model: "sonnet") 执行
    → explore(model: "haiku") 检索
```

模型多样性有限（只有 haiku/sonnet/opus 三档），但通信是原生的，agent 之间可以实时委派、验证、重试。

**CCG 的做法：跨 CLI 路由**

CCG 跑在 cc 上，但通过 shell 调用外部 CLI（Codex CLI、Gemini CLI）来获得多模型能力：

```
用户需求 → Claude Code 编排决策
  → shell 调用 Codex CLI → 后端任务 → 返回 Patch
  → shell 调用 Gemini CLI → 前端任务 → 返回 Patch
  → Claude Code 审核 Patch → apply
```

获得了多模型能力，但通信退化成了文件传递，外部 CLI 只返回 Patch，不能实时对话。

## 2. OMO ≈ 增强版的 cc `/model`

这一点是我折腾完这三个东西之后最大的感受。

你在 cc 里写 `Task(model: "opus")`，让 opus 去做架构决策；OMO 里写 `task(category: "ultrabrain")`，也是让高端模型去做架构决策。机制是一样的，只是 OMO 的"模型池"更大，它能跨模型家族，cc 只能在 Claude 内部选。

| 维度 | cc + omc | OMO on OpenCode |
|---|---|---|
| 选模型的方式 | `Task(model: "opus/sonnet/haiku")` | `task(category: "ultrabrain/deep/quick")` |
| 可选范围 | Claude 家族内（3 档） | 跨模型家族（Claude/GPT/Gemini 等） |
| 路由逻辑 | 手动或 prompt 驱动 | Category 自动分类 + 模型回退链 |
| 通信方式 | 原生 Task tool，实时委派 | 原生，框架内通信 |

为什么 OMO 能做到跨家族？因为 opencode 底下有 Vercel AI SDK 做统一抽象层，不管是 Claude、GPT 还是 Gemini，对上层来说都是一个 `LanguageModel` 接口，调用方式一样。cc 没有这个抽象层，它直接用 `@anthropic-ai/sdk`，天然只能调 Claude。

如果把 opencode 的 Vercel AI SDK 看作一个"多模型适配器"，那 OMO 其实就是"cc 的 model 参数 + 一个多模型适配器"。思路没有本质差异，差的只是底层的模型接入能力。

## 3. 绕过 cc 限制的实用方案：ccg

cc 没有多 provider 抽象层，怎么办？[风佬做的 CCG](https://linux.do/t/topic/1581788) 给出了一个务实的方案：不改 cc 的内核，直接通过 shell 调用其他 CLI。

架构很直观：

```
Claude Code（编排 + 审核）
       │
   ┌───┴───┐
   ↓       ↓
Codex CLI  Gemini CLI
 (后端)     (前端)
   │       │
   └───┬───┘
       ↓
  Unified Patch → Claude 审核后 apply
```

这架构有几个优质的设计：

- **外部模型没有写入权限**：Codex 和 Gemini 只能返回 Patch，不能直接改你的代码。所有改动都经过 Claude 审核后才 apply。这比给外部模型直接写权限安全多了
- **命令粒度清晰**：`/ccg:frontend` 调 Gemini，`/ccg:backend` 调 Codex，`/ccg:review` 做交叉审查。每个命令干什么一目了然
- **支持 Agent Teams 并行**：v1.7.60 之后可以 spawn 多个 Builder 并行写代码，适合能拆成独立模块的任务
- **集成了 OPSX 规范**：用约束集限制 AI 自由发挥，配合"零决策计划"让执行更可控

但也有明显的 trade-off：

- 单向通信：Claude 给 Codex/Gemini 发任务，它们返回结果，没有来回对话。不像 OMO 或 omc 里 agent 之间可以实时交互
- 上下文不共享：每个 CLI 有自己独立的上下文窗口，Codex 不知道 Gemini 在做什么，反过来也是。CCG 的解决方案是每步之间 `/clear`，通过文件传递状态
- 编排精细度受限：没法像 OMO 那样做 6 段委派模板级别的精细控制，也没法让外部 CLI 里的 agent 被 resume 续跑
- 依赖外部 CLI 安装：得装 Codex CLI 和 Gemini CLI，配置也比纯 cc 复杂

## 4. 三种方案的 trade-off

把三种方案放一起看：

| 维度 | opencode  omo | cc omc | cc ccg |
|---|---|---|---|
| **模型多样性** | 跨家族（Claude/GPT/Gemini） | Claude 家族内（3 档） | 跨家族（通过外部 CLI） |
| **编排精细度** | 高（角色链 + 6 段模板 + Category 路由） | 中高（角色链 + skills + prompt 驱动） | 中（命令级，无实时 agent 交互） |
| **通信效率** | 高（框架内原生通信） | 高（Task tool 原生通信） | 低（文件传递，无实时对话） |
| **代理兼容性** | 受限于 Vercel AI SDK（上一篇的坑） | 好（官方 SDK，中转站基本兼容） | 好（各 CLI 用各自的官方 SDK） |
| **安装复杂度** | 中（opencode + OMO 插件） | 低（cc + omc 插件） | 中高（cc + Codex CLI + Gemini CLI） |
| **可定制性** | 中（改 prompt 容易，改路由逻辑要碰源码） | 高（纯 markdown 文件，随便改） | 中（命令是固定的，定制要碰 npm 包） |

没有哪个方案是全面碾压其他的。选哪个取决于你最在意什么：

- **模型多样性 + 精细编排** $\Rightarrow$ OMO，但得接受 Vercel SDK 的代理兼容性问题
- **稳定好用 + 高可定制** $\Rightarrow$ omc，但没有跨模型家族能力
- **多模型 + 不想碰 opencode** $\Rightarrow$ CCG，但得接受通信效率的损失

## 5. 个人看法

折腾了这一圈，我的感受是：多模型协作这个方向绝对是对的，但现在所有方案都是 workaround。

理想的状态应该是一个 AI 编码工具原生支持：
1. 多 provider 抽象层（像 opencode 一样能接 Claude/GPT/Gemini）
2. 精细的角色编排（像 OMO 一样有规划 $\Rightarrow$ 审核 $\Rightarrow$ 调度 $\Rightarrow$ 执行的完整链路）
3. 原生的跨模型通信（agent 之间可以实时对话，不只是传文件）
4. 对第三方代理友好（不被中间 SDK 的请求格式差异卡住）

目前没有哪个工具同时做到了这四点。opencode 有 1 但缺 4，cc 有 2 和部分 3 但缺 1，CCG 有 1 但缺 3。

所以回到现实，我的选择是：

**日常开发：cc + omc**：单模型家族内的编排已经够用了。opus 做决策、sonnet 做执行、haiku 跑检索，配上 omc 的 workflow，效率已经比"一个模型从头干到尾"高了不少。而且 cc 用官方 SDK，中转站兼容性好，不用在配置上浪费时间。

**想体验多模型协作：CCG**： 风佬的方案是目前在 cc 生态里实现多模型协作的优质实践。虽然通信效率有损失，但"Claude 编排 + Codex 后端 + Gemini 前端"这个分工确实有道理，尤其是全栈项目能发挥出优势。

**等待未来**： 如果哪天 cc 原生支持了多 provider（比如能在 Task tool 里写 `provider: "openai"`），或者 opencode 解耦了 Vercel SDK 的绑定，那才是真正的"理想方案"降临的时候。在那之前，该折腾折腾，该蹬蹬。

---

*本文为 AI 编码工具多模型协作系列的第三篇，基于 OMO、omc、CCG 的实际使用和源码分析整理。*

*系列文章：*
1. *[OpenCode 的 Provider 困境：Vercel AI SDK 绑定与第三方 API 代理适配](sdk-opencode.md)*
2. *[Agent 角色编排架构拆解：从 OMO 的设计到 Claude Code 自建](agent-workflow-opencode.md)*
3. *本文*
