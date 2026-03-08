---
title: "OpenCode 的 Provider 困境：Vercel AI SDK 绑定与第三方 API 代理适配"
publishDate: 2026-03-08
description: "深度分析 OpenCode 架构与 Vercel AI SDK 的绑定关系，对比 @anthropic-ai/sdk 和 @ai-sdk/anthropic 的请求格式差异，并完整记录第三方 Anthropic 代理（anyrouter、wzw）兼容性问题的排查过程与结论。"
tags: ["opencode", "claude-code"]
language: 'Chinese'
first_level_category: "项目实践"
second_level_category: "Agent 搭建"
heroImage: { src: './1.png'}
draft: false
---


# OpenCode 的 Provider 困境：Vercel AI SDK 绑定与第三方 API 代理适配

> 最近折腾 OpenCode 配置第三方 API 代理时，遇到了一连串报错。
> 排查了一圈下来，我发现这些问题的根源不在配置本身，而是 OpenCode 在架构上深度绑定了 Vercel AI SDK——这导致它与 Claude Code CLI 使用的官方 SDK（`@anthropic-ai/sdk`）存在结构性不兼容，而且用户没办法自己改。

## 1. 事情起因

[OpenCode](https://opencode.ai) 虽然没有 openclaw 这么无脑火，但我在使用 cc 的配套项目 oh-my-claude(omc) 的时候刷到过。我看他它支持自定义 provider，理论上只要改几行 JSON 配置就能接入各种 LLM 代理，正好可以使用他推荐的 claude 主控 + codex 执行 + Gemini 设计 ui。

我手上是有着 codex 的 team，但习惯用 claude 后，没咋用 codex。但逛社区的时候发现论坛里说 gpt-5.3 codex 挺强的，反倒是 cc 在抽风？（我用这么久了，没见到抽风过，最近加上 omc 的 workflow 后性能更强了）然后不是新出了 gpt-5.4 嘛，正好看到这个项目，就像体验最好的开发 AI 工作流：CC 编排主控，codex 负责执行，Gemini 设计 UI。感觉这个组合挺完美的，就想试试。

查看 opencode 推荐的配置：

```json
{
  "provider": {
    "my-provider": {
      "npm": "@ai-sdk/anthropic",
      "options": {
        "baseURL": "https://api.example.com/v1",
        "apiKey": "sk-..."
      },
      "models": {
        "claude-sonnet-4-6": { "name": "claude-sonnet-4-6" }
      }
    }
  }
}
```

其中 `npm` 字段指定用哪个 Vercel AI SDK 的 provider 包来发请求。用官方 API 的时候一切正常，但当我尝试接入中转代理时，就出了挺多 bug。当然总结下来就像开头说的，是 opencode 的架构绑定了 vercel 的 SDK，所以用一些中转站（像 anyrouter）就会因为请求格式不对而被拒绝。

## 2. 排查经过：从"配置写错"到"这不是配置的问题"

我的配置里有两个 Anthropic 代理：`wzw`（Wong 佬的公益站） 和 `anyrouter`（代善人）。这俩个要是直接接入 cc 的话没有任何问题，但配置 opencode 时，就报错了。

Wong 佬的中转站使用以下配置就行（不支持 opus）：

```json
"wzw": {
  "models": {
    "claude-sonnet-4-5": {
      "name": "claude-sonnet-4-5"
    }
  },
  "name": "WZW",
  "npm": "@ai-sdk/anthropic",
  "options": {
    "apiKey": "sk-xxxx",
    "baseURL": "https://wzw.pp.ua/v1"
  }
}
```

之后就没报错，但 A 大善人就不行了。

`anyrouter` 不管是走 `betterclau.de` 代理、自搭建的中转平台，还是直连主站都是报错。然后就开始尝试各种 baseURL 的组合......

### 各种 baseURL 的尝试

anyrouter 配置：

```json
    "anyrouter": {
      "models": {
        "claude-opus-4-6": {
          "name": "claude-opus-4-6"
        },
        "claude-sonnet-4-6": {
          "name": "claude-sonnet-4-6"
        }
      },
      "name": "Any Router",
      "npm": "@ai-sdk/anthropic",
      "options": {
        "apiKey": "sk-xxxxxx",
        "baseURL": "https://betterclau.de/claude/anyrouter.top/v1"
      }
    },
```

要理解这些报错，先得知道一个前提：`@ai-sdk/anthropic` 会在你填的 `baseURL` 后面自动追加 `/messages` 来拼出完整的请求地址。知道这个规则后，下面的报错就好理解了：

| 我填的 baseURL | SDK 实际请求的地址 | 请求模型 | 结果 |
|---|---|---|---|
| `.../anyrouter.top`（没加 `/v1`） | `.../anyrouter.top/messages` | claude opus/sonnet | Invalid endpoint. Path must contain v1/messages |
| `.../anyrouter.top/v1` | `.../anyrouter.top/v1/messages` | claude opus | invalid claude code request |
| `.../anyrouter.top/v1` | `.../anyrouter.top/v1/messages` | claude sonnet | 当前模型负载到达上限 |

看到第三行的报错时，我意识到路径是对的：请求确实到了服务端，不过代理那边 sonnet 模型负载过高。到这一步，配置格式的问题其实已经排除了。

### 在 Claude Code 里能用 anyrouter

后面我用 cc 试了一下 A 社 的代理，实际上 opus 4.6 能用，sonnet 4.6 不行（负载过高）。这说明代理本身是没问题的，URL 也是对的，模型名也是对的。

所以关键结论就是：**同一个 anyrouter 代理、同一个 baseURL，在 Claude Code CLI 里用 `claude-opus-4-6` 是完全正常的。**

到这一步我基本确定了：问题不在代理、不在 URL、不在模型名，而应该是两个工具发出去的 HTTP 请求不一样。

### 最后的尝试：换 npm 包

既然 `@ai-sdk/anthropic` 有问题，那换成 `@ai-sdk/openai-compatible`。

结果代理返回：`"Invalid endpoint. Path must contain v1/messages"`。

这说明 anyrouter 只认 Anthropic 格式（`/v1/messages`），而 `@ai-sdk/openai-compatible` 发的是 OpenAI 格式（`/v1/chat/completions`），代理不认。

至此彻底死局：
- `@ai-sdk/anthropic`：路径格式对了，但部分模型被拒
- `@ai-sdk/openai-compatible`：路径格式就不对，代理不认
- `@anthropic-ai/sdk`（Claude Code 用的那个）：什么都对，但 OpenCode **不支持这个包**。

## 3. 问题原因：两套 SDK 到底差在哪

### Claude Code SDK

Claude Code CLI 用的是 Anthropic 官方 SDK `@anthropic-ai/sdk`，直接构造原汁原味的 Anthropic Messages API 请求：

```typescript
import Anthropic from '@anthropic-ai/sdk';
const client = new Anthropic({ baseURL, apiKey });
await client.messages.create({
  model: 'claude-opus-4-6',
  messages: [{ role: 'user', content: 'Hello' }],
  max_tokens: 1024
});
```

### OpenCode SDK

OpenCode 用的是 Vercel AI SDK 封装的 `@ai-sdk/anthropic`，多了一层抽象：

```typescript
import { createAnthropic } from '@ai-sdk/anthropic';
const provider = createAnthropic({ baseURL, apiKey });
await generateText({
  model: provider('claude-opus-4-6'),
  prompt: 'Hello'
});
```

### 差异汇总

虽然两者最终都是往 `/v1/messages` 发 POST 请求，但构造出来的 HTTP 请求并不完全一样：

| 维度 | `@anthropic-ai/sdk`（官方） | `@ai-sdk/anthropic`（Vercel 封装） |
|------|-----|-----|
| Headers | `x-api-key`、`anthropic-version` | 可能附加额外的 headers |
| Request body | 官方 Messages API 格式 | 经过 Vercel AI SDK 转换后的格式 |
| 流式处理 | 原生 SSE | Vercel AI SDK 自己的流式抽象 |
| 客户端校验 | 没有路径校验 | 强制校验 baseURL 是否包含特定路径 |

这些差异通常不影响官方 API 的使用，但第三方代理对请求格式的容忍度各不相同（Wong 佬牛 P）。有些代理能正确处理官方 SDK 的请求，却会拒绝 Vercel 封装版的请求。

## 4. 为什么用户没法自己换 SDK

搞清楚问题出在 npm 包之后，我最直觉的想法就是：把配置里的 `"npm": "@ai-sdk/anthropic"` 改成 `"npm": "@anthropic-ai/sdk"` 不就行了？但实际上这一步是行不通的（笑嘻了🤪，配了 6 小时）。

### 接口契约完全不同

OpenCode 的 provider 系统是围绕 Vercel AI SDK 的 `LanguageModel` 接口设计的。它在运行时动态加载 npm 包，期望拿到一个符合这个接口的 provider 对象，然后传给 `generateText()` 或 `streamText()` 来调用。

> `generateText()` 来自 Vercel AI SDK 的核心包 ai（https://sdk.vercel.ai/docs/ai-sdk-core/generating-text），OpenCode 在内部使用这套接口来调用模型。
> 但 OpenCode 实际的调用代码可能封装了更多层逻辑，不会长得这么简洁。以下是示意代码。

```typescript
// OpenCode 期望的调用链（Vercel AI SDK 接口）
const provider = createAnthropic({ baseURL, apiKey });
const model = provider('claude-opus-4-6');   // 返回 LanguageModel 对象
await generateText({ model, prompt: '...' });

// @anthropic-ai/sdk 的调用方式（完全不同的接口）
const client = new Anthropic({ baseURL, apiKey });
await client.messages.create({ model: '...', messages: [...] });
```

两个包的构造函数签名不一样，返回的对象类型不一样，调用方式也不一样。`@anthropic-ai/sdk` 返回的 `Anthropic` 实例没有实现 `LanguageModel` 接口，塞进 `generateText()` 里只会报类型错误。

### 运行时加载链路是写死的

OpenCode 运行时的加载链路大概是这样：

```
opencode.json 的 npm 字段
  → require("@ai-sdk/anthropic")
    → createAnthropic(options)
      → 返回 LanguageModel provider
        → 传入 generateText() / streamText()
```

这条链路的每一步都假设加载的包遵循 Vercel AI SDK 的接口规范。你没法在中间插入一个接口完全不同的包。

### 改 node_modules 也没用

我还想过一个"野路子"：直接去 `~/.cache/opencode/node_modules/@ai-sdk/anthropic/` 里改源码。但试了一下就放弃了，OpenCode 调用的不是 `@ai-sdk/anthropic` 的 HTTP 发送函数，而是 Vercel AI SDK 的顶层函数 `generateText`。你改了 `@ai-sdk/anthropic` 的内部实现，也影响不到 `generateText` 那一层的逻辑。要改就得把 OpenCode 自身的调用链路也一起改掉，那就不是"改配置"，是"重写核心逻辑"。

### 开源归开源，但有点固化了🤪

OpenCode 确实是开源项目，代码在 GitHub 上。但它是 Go + TypeScript 混合架构（Go 做 TUI 界面，JavaScript 做 provider bridge）。如果要让它支持 `@anthropic-ai/sdk`，大致有两条路：

1. 写一个适配器，把 `@anthropic-ai/sdk` 的 API 包装成 Vercel AI SDK 的 `LanguageModel` 接口
2. 重写 provider 加载逻辑，让它能识别和处理非 Vercel AI SDK 的包

不管走哪条路，都是架构级的改动。对我来说，我就想试试不同模型参与合作，实现 agent team 协作的效果是不是比单用一个模型家族（cc）的 omc 更好🥵。

## 5. 退一步看，这是 trade-off，不是 bug

说到底，OpenCode 选择 Vercel AI SDK 作为唯一的 LLM 抽象层，是一个有得有失的架构决策：

| 角度 | 得 | 失 |
|:----:|:---------:|:---------:|
| 开发者 | 一套接口支持 OpenAI/Anthropic/Google 等多家 provider | 被锁定在 `@ai-sdk/*` 生态里 |
| 用户 | 切换 provider 只改一行 `npm` 字段，非常方便 | 只能用 Vercel 已经封装好的 provider 包 |
| 第三方代理 | 主流代理开箱即用 | 非标准代理可能因请求格式差异而不兼容 |

反过来看 Claude Code CLI，它直接用 `@anthropic-ai/sdk`，没有中间抽象层。好处是发出去的请求与 Anthropic 官方 API 完全一致，第三方代理兼容性更好。代价是它只能跑 Claude 模型，没有 OpenCode 那种切换到 GPT 或 Gemini 的灵活性。

两种选择都有道理，只是面向的用户群不同。但如果你恰好是那种"需要用第三方代理 + 代理又只兼容官方 SDK"的用户，就会像我一样被卡在中间。（有点想魔改一下了，但投产比有点低，cc 的 omc 就满足了🥵）

## 6. 实用建议

折腾了这么一圈，总结几条实际可操作的建议：

1. 选 agent 之前先测兼容性：不是所有第三方代理都能被 vercel 的 SDK `@ai-sdk/anthropic` 正常调用。找一个已经验证过能在 OpenCode 里用的代理，比瞎试 baseURL 高效（不知佬们有没有现成的测试用例）
2. 不兼容代理就用原生 cc 或 codex 了：如果某个代理在 OpenCode 里不行但在 Claude Code 里能用，那就在 Claude Code 里用它，没必要死磕
3. 不要在配置层面钻牛角尖：遇到"Invalid claude code request"这类报错，如果 baseURL 和模型名都确认没错，大概率是 SDK 格式差异造成的。这个问题在配置层面无解
4. 关注社区进展：向 OpenCode 项目提 feature request，建议支持 `@anthropic-ai/sdk` 作为可选 provider 包

## 7. 最后吐槽一下

opencode say goodbye 了😆。

说真的，我对 opencode 的期待挺高的。多 provider 支持、能同时调 Claude/GPT/Gemini，UI 也挺好看——听起来就是那种"终极 AI 编码工具"。但实际用下来的感受就是：它把"支持多模型"这件事做到了表面，底层却被 Vercel AI SDK 固定了。（可以自己魔改，或者等社区贡献匹配器吧）

你说它不能用吧，官方 API 确实能跑。你说它好用吧，稍微用个非标准代理就各种炸。最离谱的是你想自己修都修不了，配置改不动，源码改不起，node_modules 改了也没用。6 个小时的排查换来一个结论："这不是你的问题，是架构的问题。" 好家伙，架构的问题那不就是没得救了么😇

cc 虽然只能跑 Claude 家族的模型，但人家用的是官方 SDK，请求格式就是标准的 Anthropic API，中转站基本都能兼容。再配上 omc 的 workflow，单模型家族内的角色编排其实已经够用了。你用 opus 做架构决策，sonnet 做日常执行，haiku 跑快速检索，效果比折腾多模型协作省心多了。

当然我不是说多模型协作这个方向不对——恰恰相反，我觉得这是未来。让 Claude 做编排、Codex 做深度推理、Gemini 做前端设计，这个分工确实有道理。但 opencode 现在的实现离这个愿景还有点远：**它给了你选模型的自由，却没给你选 SDK 的自由。** 你能填 `@ai-sdk/openai`、`@ai-sdk/anthropic`、`@ai-sdk/google`，但这几个包的请求格式和官方 SDK 都有微妙差异。等于是你以为自己在直接调 Claude API，其实中间隔了一层 Vercel 的翻译器——而这个翻译器有时候会"翻译"出代理听不懂的东西。

当然在这感谢[风佬做的 ccg](https://linux.do/t/topic/1581788)（https://linux.do/t/topic/1581788），虽然不像单 agent cli 一样能做到很精细的工作编排（使用 skills、Claude.md 精细化），通信退化为文件传递，失去实时协作能力，但能编排三个 cli：cc、codex、gemini cli，感觉能实现三个 cli，使用不同家族的模型协作，真挺强的了🌹🌹🌹（已经在蹬了）

所以最后的结论很简单：如果你和我一样主力用 Claude，代理也是 Claude 系的，**老老实实用 cc + omc 就完事了**（cc 贵了点，A➗也有点......但贵有贵的道理）。想体验多模型协作的话，等 opencode 什么时候把 SDK 层解耦了再说吧。或者等哪个大佬写个适配器（我不写，我配了 6 小时已经累了🥵）。

---

*本文基于 OpenCode 实际配置排查过程整理，所有报错信息均为真实遇到。*

*系列文章：*
1. *本文*
2. *[Agent 角色编排架构拆解：从 OMO 的设计到 Claude Code 自建](agent-workflow-opencode.md)*
3. *[多模型 Agent 协作：从 OMO 到 CCG 的范式比较与反思](multi-model-agent-collab.md)*