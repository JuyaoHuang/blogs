---
title: "oh-my-opencode 多 Agent 流水线架构拆解：角色编排设计与 Claude Code 自建实践"
publishDate: 2026-03-08
description: "深度拆解 oh-my-opencode（OMO）的多 Agent 流水线：解析 Prometheus 规划、Atlas 调度、Sisyphus 执行等角色的职责边界与协作模式，并介绍如何在 Claude Code 中自建同类 Agent 编排体系。"
tags: ['opencode', 'claude-code']
language: 'Chinese'
first_level_category: "项目实践"
second_level_category: "Agent 搭建"
heroImage: { src: './2.png', color: '#D58388' }
draft: false
---


# Agent 角色编排架构拆解：从 OMO 的设计到 Claude Code 自建

> 上一篇聊了 opencode 的 SDK 绑定问题，最后得出结论：对我来说 cc + omc 就够了。但折腾 opencode 的过程中，我顺手翻了一下 oh-my-opencode（OMO）的源码，发现它的 Agent Workflow 编排设计其实挺有意思的，用希腊神话里的角色名来命名不同的 agent，每个 agent 有明确的职责边界和行为约束。这篇就来拆解一下 OMO 的编排思路，然后看看能不能在 Claude Code 里自己搭一套类似的。

## 1. 为什么要研究 OMO 的编排

说实话，一开始我是冲着 opencode 的多模型支持去的（上一篇的故事）。但配了 6 小时代理没配成之后，我开始转向研究 OMO 本身的设计，毕竟它在 opencode 社区里口碑不错，很多人说用了之后"写代码的感觉完全不一样了"。

翻了它的 `dist/index.js`（打包后的源码，大概 9 万行😇），我发现 OMO 最核心的价值不在于"支持多模型"，而在于**它把 AI 编码这件事拆成了一条完整的流水线**：先有人规划，再有人审计划，然后有人分发任务，最后有人执行和验证。每个环节都有专门的 agent 负责，而且有明确的"能做什么"和"不能做什么"。

这个思路其实和人类团队协作是一样的——你不会让产品经理直接写代码，也不会让实习生做架构决策。OMO 把这套分工搬到了 AI agent 上。

## 2. OMO 的角色编排全景

OMO 的完整流水线长这样：

```
Prometheus（规划）
    ↓
  Metis（规划前顾问，可选）
    ↓
  Momus（计划审核，可选）
    ↓
Atlas（执行编排/调度）
    ↓
Sisyphus / Hephaestus（执行）
    ↓
  Explore / Librarian / Oracle（辅助）
```

用大白话讲每个角色在干嘛：

### Prometheus — 规划师

只做规划，不写代码。它的核心任务就是把一个模糊的需求变成一份"决策完备"的计划文件，存到 `.sisyphus/plans/*.md` 里。

> 决策完备：执行者拿到计划之后不需要再做任何判断，该用什么技术栈、文件放哪里、接口怎么设计，全都写清楚了。执行者只需要"照着做"就行。

Prometheus 有三条原则挺值得学：
1. **Decision Complete** — 不给执行者留判断空白
2. **Explore Before Asking** — 能通过读代码搞清楚的事就别问用户
3. **区分事实和偏好** — 技术栈版本之类的能查到的是事实，用 React 还是 Vue 这种是偏好，只有偏好才需要问用户

### Metis — 规划前顾问

在复杂任务里给 Prometheus 打辅助。它是只读的，不改任何东西，就是帮忙找出计划里可能有问题的地方：意图分类不清晰、范围边界模糊、有没有遗漏的风险、验收标准够不够具体。

它还会给 Prometheus 输出一份"MUST/MUST NOT"指令——相当于在规划之前先画好红线。

### Momus — 计划审核员

这个角色的设计我觉得很精妙。它的定位是**阻塞检查器，不是完美主义审稿器**。默认偏向通过（OKAY），只有发现"真阻塞"才会拒绝（REJECT）。

> 真阻塞比如计划里引用了一个不存在的文件路径，或者某个任务的前置依赖没写清楚，或者测试场景根本跑不了。

这个"默认通过"的设计很重要——如果审核太严，流水线会一直卡在审核环节转圈；如果太松，烂计划就会流到执行环节浪费 token。Momus 的平衡点是：**只拦住那些会导致执行必定失败的问题**。

### Atlas — 执行总调度

定位是 Conductor（总指挥），自己不写代码，只负责把计划拆成具体任务分发给执行者，然后验收结果。

Atlas 有两个强制约束：
1. **委派必须用 6 段式提示**（后面会详细讲）
2. **验收必须亲自做** — 不能光看执行者说"我做完了"就信，得自己跑测试、读改动的文件、必要时做实操 QA

### Sisyphus — 主执行者

名字取自那个推石头上山的人，挺形象的。Sisyphus 是默认的执行 agent，但有个有趣的设定：**它的默认行为是委派，不是自己做**。只有超简单的本地改动（比如改个拼写错误）才会自己动手，其他的都往下委派。

它的执行流程是固定的：意图识别 → 探索代码 → 制定方案 → 路由到合适的执行者 → 执行 → 验证 → 失败则重试 → 完成。

### Hephaestus — 深度自治执行者

定位是 Senior Staff Engineer，给它一个任务就闷头干，除非真的卡住了才会来问你。和 Sisyphus 的区别是：Sisyphus 习惯委派，Hephaestus 习惯自己搞定。

OMO 推荐给 Hephaestus 用 GPT-5.3 Codex 模型是因为 Codex 擅长长时间自治推理，和这个角色的定位很匹配（现在有 gpt-5.4 了，还没咋蹬，因为太降智了，juice 才 20，5.3-codex 是正常的）。

### 辅助角色

- Explore：代码库检索专家。先分析你的意图，再并行调用搜索工具，最后给出结构化结果
- Librarian：外部文档检索。去查官方文档、开源实现、版本信息，带可溯源链接
- Oracle：高阶咨询。只在架构权衡、反复失败后的疑难杂症、重大技术决策时才请出来。只读、昂贵（推荐用 Opus）
- Multimodal-Looker：看图说话。解释 PDF、截图、图表这些非纯文本内容

## 3. 几个值得学的设计亮点

### 6 段委派模板

Atlas 和 Sisyphus 每次委派任务的时候，都要遵循一个固定的 6 段格式：

```
1. TASK          — 原子任务描述（不能模糊）
2. EXPECTED OUTCOME — 预期产物、行为、验证命令
3. REQUIRED TOOLS   — 必须用什么工具
4. MUST DO          — 必须遵守的规则
5. MUST NOT DO      — 绝对禁止的事
6. CONTEXT          — 相关路径、参考模式、依赖信息
```

这个模板的好处是**消除歧义**。你直接跟 AI 说"帮我加个登录功能"，它可能从 UI 开始写，也可能从数据库开始建表，你不知道它会走哪条路。但如果你用 6 段模板把预期产物、必须用的工具、禁止做的事都写清楚了，AI 跑偏的概率就小很多。

### Category 模型路由

OMO 内置了 8 个任务类别，每个类别有推荐的模型：

| Category | 语义 | 推荐模型倾向 |
|---|---|---|
| `visual-engineering` | 前端/UI/UX | Gemini |
| `ultrabrain` | 高难逻辑/架构 | Claude Opus |
| `deep` | 深度自治问题求解 | GPT Codex |
| `artistry` | 非常规/创造性解题 | 看任务 |
| `quick` | 单文件小改 | 快速模型 |
| `writing` | 文档与写作 | 通用 |

这个路由机制让 OMO 可以按任务性质自动选择最合适的模型——前端任务丢给 Gemini，深度推理丢给 Codex，架构决策用 Opus。**这也是 OMO 和 omc 的核心区别之一**：omc 只能在 Claude 家族内选（haiku/sonnet/opus），而 OMO 能跨模型家族。

### Decision Complete 规划原则

这个理念我觉得不管用不用 OMO 都值得借鉴。很多时候我们给 AI 的指令太模糊了，然后 AI 自己做了一堆决策，做得好了你觉得它很聪明，做得不好你觉得它在乱搞——但其实问题不在 AI，在于你没把需求想清楚。

> 这一点论坛里很多佬都讲过类似的情况，我也是。
> 有时候搞不清需求说个大概然后用 cc 的 omc 中的 `/deep-interview` 模式和 AI 探讨一下设计方案和实现方式，自己也就清楚了。

Prometheus 的做法是：**在规划阶段就把所有决策做完**，执行者只需要机械性地执行。这听起来前期工作量大了，但实际上省下了后面反复修改和返工的成本。

## 4. 能不能在 Claude Code 里自建一套

答案是：可以的兄弟，可以的。（但有关键差异）

### 可以的，兄弟

Claude Code 原生就支持自定义 agent。在项目的 `.claude/agents/` 目录下放 markdown 文件就行：

```markdown
<!-- .claude/agents/prometheus.md -->
---
name: prometheus
description: "规划专家，只做规划不写代码"
tools:
  - Read
  - Glob
  - Grep
  - WebSearch
---

你是 Prometheus，专职规划师。你的任务是...
（把 Prometheus 的三原则和行为约束写进去）
```

类似地可以创建 `metis.md`、`momus.md`、`atlas.md`、`sisyphus.md` 等。每个 agent 可以限制它能用的工具——比如 Momus 和 Oracle 只给只读工具，Atlas 不给文件编辑工具。

Skills 也能做。在 `.claude/skills/` 下放 markdown 文件，定义触发条件和执行流程：

```markdown
<!-- .claude/skills/workflow.md -->
---
name: workflow
description: "完整的规划→审核→执行流水线"
---

当用户提出一个需求时，按以下步骤执行：
1. 启动 Prometheus agent 进行规划...
2. 启动 Momus agent 审核计划...
3. 启动 Atlas agent 分发执行任务...
```

最终的文件结构：

```
.claude/
  agents/
    prometheus.md     # 规划
    metis.md          # 规划前顾问
    momus.md          # 计划审核
    atlas.md          # 执行编排
    sisyphus.md       # 主执行
    hephaestus.md     # 深度执行
    librarian.md      # 外部检索
    oracle.md         # 高阶咨询
  skills/
    workflow.md       # 流水线触发器
  claude.md           # 全局编排规则 + 6段委派模板
```

全是 markdown 文件，不需要写一行代码。

### 做不到，兄弟

**1. 流水线的严格性**

OMO 的 `Prometheus → Metis → Momus → Atlas → Sisyphus` 是代码级保证的顺序执行。在 Claude Code 里，这个流水线靠 prompt 驱动——你在 skill 里写"先调 Prometheus 再调 Momus"，但 AI 偶尔会跳步或者合并步骤。大部分时候它会遵守，但不是 100%。**当上下文过长时，不稳定性就很高。**（说的就是 A➗提供的 1M context 的 opus）

**2. 模型路由的粒度**

Claude Code 的 Task tool 有 `model` 参数，但只支持三档：`haiku`、`sonnet`、`opus`。你没法像 OMO 那样把前端任务路由到 Gemini、深度推理路由到 GPT Codex。只能在 Claude 家族内做粗粒度的选择。

**3. Category 自动分类**

OMO 会根据任务内容自动判断 category（`visual-engineering`、`ultrabrain` 等）然后路由到对应模型。Claude Code 没有这个机制，你得在 prompt 里自己写分类逻辑，或者手动指定。

## 5. 自建的核心风险

说白了就一个：**prompt 不是合同**。

你在 agent 的 markdown 文件里写了"你只能做规划，不能写代码"，AI 大部分时候会遵守，但它不是被代码强制限制的——它是"被说服"的。遇到某些场景，它可能会觉得"直接改一行代码更高效"然后就动手了，完全无视你的约束。

OMO 也有这个问题，但它在代码层面做了更多防护。比如给只读角色真的只传入只读工具，而不仅仅是在 prompt 里说"你只能读"。Claude Code 的 agent frontmatter 也支持工具限制（`tools:` 字段），这一点倒是能对齐。

另一个风险是**调试成本**。OMO 的角色 prompt 是经过大量迭代优化的（从 `dist/index.js` 的代码量就能看出来），每个角色的行为边界、异常处理、失败回退都考虑得比较周全。自建的话，你得自己踩这些坑——比如 Momus 审核太严导致流水线卡死、Sisyphus 委派时上下文丢失、Atlas 验收不够严格导致 bug 流入等等。这些都是 OMO 踩过的坑，自建大概率会重新踩一遍。

## 6. 结论

值不值得自建？看你的需求：

**如果你用 cc + omc**：omc 已经有一套自己的 workflow 编排（planner、architect、executor、critic 等），虽然角色设计和 OMO 不完全一样，但核心思路是相通的：规划、审核、执行、验证的分工。对于日常开发来说，omc 的 workflow 已经够用了，没必要再自建一套 OMO 风格的。

**如果你想学习 agent 编排的设计思想**：OMO 的设计是很好的参考材料。尤其是"Decision Complete"规划原则、6 段委派模板、Momus 的"默认通过"审核哲学——这些理念不依赖于特定工具，你可以把它们融入到任何 agent workflow 里。

**如果你纯粹想折腾**：那就搞呗（佬加油，搞了记得 @我 来蹬🥵）。写几个 `.claude/agents/*.md` 文件，定义好角色和工具限制，再写个 skill 把它们串起来。整个过程不需要写代码，只是 markdown，改起来也方便。就算最后发现没比 omc 好用多少，至少你对 agent 编排的理解会深一个层次。

OMO 最牛 P 的不是它的代码实现，应该是它的**角色设计哲学**。每个 agent 都有清晰的"不做什么"边界。Prometheus 不写代码，Momus 不追求完美，Atlas 不亲自动手，Oracle 不轻易出山。这种"克制"的设计反而让整个系统更可控。这一点，不管用什么工具做 agent 编排，都值得借鉴。

---

*本文基于 oh-my-opencode `dist/index.js` 源码分析整理，角色 prompt 为提炼版，非原文摘录。*

*系列文章：*
1. *[OpenCode 的 Provider 困境：Vercel AI SDK 绑定与第三方 API 代理适配](sdk-opencode.md)*
2. *本文*
3. *[多模型 Agent 协作：从 OMO 到 CCG 的范式比较与反思](multi-model-agent-collab.md)*