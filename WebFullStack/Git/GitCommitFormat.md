---
title: "Git commit 编写格式"
publishDate: 2025-12-07
description: "Git commit、PR 和 CR 编写格式要求和介绍"
tags: ['git']
language: 'Chinese'
first_level_category: "Web全栈开发"
second_level_category: "Git与版本控制"
draft: false
---

## Commit Message 编写格式

### 📚 优质 Commit 书写格式指南

目前 GitHub 上最流行的是 **[Conventional Commits](https://www.conventionalcommits.org/)** 规范。

#### 1. 基本结构
```text
<type>(<optional scope>): <description>
<space line>
<body>
<space line>
<footer>
```

#### 2. 逐行解析

**第一行：Header (标题)**
*   **`<type>` (类型)**: 告诉别人你干了什么
    *   `feat`： 新增功能 feature。`introduces a new feature to the codebase`
    *   `fix`： 修补代码库里存在的 Bug
    *   `docs`： 文档改变 (Documentation) 
    *   `style`： 格式调整 (不影响代码运行，如空格、缩进)
    *   `refactor`： 代码重构 (既没加新功能也没修 Bug，优化结构)
    *   `chore`： 构建过程或辅助工具的变动 (如修改 .gitignore)
*   **`(<scope>)` (范围)**： (可选) 影响了哪个模块？例如 `(api)`, `(ui)`, `(prompt)`
*   **`<description>` (主题)**: 简短描述（建议 50 字符以内），**用祈使句**（即 "Add file" 而不是 "Added file"），全部小写，结尾不要加句号

**第三行：Body (正文)**

*   详细描述为什么要改，以及改了什么
*   可以使用列表（如 `- ` 开头）
*   每行尽量不超过 72 个字符（防止在 git log 中换行难看）

**最后一行：Footer (页脚)**

*   (可选) 关联 Issue 或其他信息。例如：`Closes #123`

### 示例

原始 commit：

```bash
Accomplished study of Prompt Engineering in the stage one
```

修改后：

代码提交：

```bash
feat(prompt_test.py): implement stage 1 prompt engineering patterns

Completed the basic study of Prompt Engineering techniques.
This commit includes:
- Role Playing: Implemented system prompts for dynamic persona switching.
- Chain-of-Thought (CoT): Added step-by-step reasoning logic to improve accuracy.
- Structured Output: Enforced JSON formatting for API responses.

Ref: Stage 1 Learning Plan, Issue #1
```

文档提交：

```bash
docs(Prompt_building): add notes for stage 1 prompt engineering

Documented key concepts learned in Stage 1:
- Role Playing (System Prompt configuration)
- Chain-of-Thought (CoT) reasoning
- Structured Output (JSON extraction)
```

---

## Pull Request 编写格式

```bash
<!-- 
PR Title Format: type(scope): short description
Example: feat(prompt): implement Chain-of-Thought for logic tasks
-->

## 📝 Description (背景与简介)

**What does this PR do?**
<!-- 简要描述这个 PR 完成了什么功能，解决了什么问题 -->
This PR implements the core logic for the Chatbot, including multi-turn memory and streaming response handling.

**Why are we doing this?**
<!-- 关联 Issue 或解释动机 -->
- Closes #12 (Link to the issue)
- Necessary for the upcoming Streamlit UI integration.

## 🛠️ Key Changes (主要变更点)

<!-- 列出具体的修改逻辑，方便 Reviewer 快速定位重点 -->
- **Refactor**: Moved API configuration to `config.py`.
- **Feature**: Added `get_completion_stream()` function in `llm_base.py`.
- **Prompt**: Updated the System Prompt to support JSON mode.

## 🧪 How to Test (如何测试/验证)

<!-- 告诉 Reviewer 如何复现你的结果，这是最重要的部分！ -->
1. Run the script: `python tests/test_stream.py`
2. Input a question: "Tell me a joke."
3. Observe the output: Text should appear character by character.

## 📸 Screenshots / Logs (截图/日志)

<!-- 对于前端改动放截图，对于后端/AI 改动放运行日志。无图无真相！ -->
| Before | After |
| ------ | ----- |
| (Optional) | ![Streaming Demo](url_to_gif_or_image) |

## ✅ Checklist (自查清单)

- [ ] My code follows the project style guide.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented hard-to-understand areas.
- [ ] **NO API KEYS** are committed (checked `.env`).
```

### 🧐 深度解析：为什么这么写

#### 1. 标题 ---- 遵循 Conventional Commits

标题是 PR 的门面。一个好的标题应该一眼就能看出改动的类型和范围

*   格式： `type(scope): subject`
*   常用 Type：
    *   `feat`: 新功能 (Feature)
    *   `fix`: 修补 Bug
    *   `docs`: 仅修改文档
    *   `refactor`: 重构代码（不改变功能）
    *   `chore`: 杂活（构建过程、依赖库更新等）
*   示例：
    *   ❌ `Update code` (太模糊)
    *   ✅ `feat(api): add retry logic for OpenAI connection` (清晰)

#### 2. Description  ---- 上下文

不要只说怎么做，要说为什么做

*   如果是修复 Bug，描述一下 Bug 的现象
*   如果是新功能，描述一下预期的效果
*   **关键点**：一定要关联 Issue（如 `Closes #42`），这样 PR 合并后，Issue 会自动关闭，保持项目整洁

#### 3. How to Test  ---- 验证步骤，建立信任

区分新手和资深工程师的分水岭

*   新手直接丢代码，让 Reviewer 自己猜怎么跑
*   资深工程师会给出**复现步骤**。这大大降低了 Reviewer 的心智负担，PR 合并速度会快 3 倍以上
*   针对 LLM 项目：你可以贴一段你精心设计的 Prompt 和 AI 的精彩回复

#### 4. Checklist ---- 自查清单，职业素养

列举一个 ToDO List，确保没有遗漏疏忽的地方和多余的修改。

#### 核心

**Be kind to your reviewer.** (对审查者好一点)

假如审查者是你那个"很忙、很累、且对这块代码不熟悉"的同事，你的描述能让他 1 分钟看懂吗？如果能，这就是一个完美的 PR

---

## Code Review 编写格式

提交代码（Commit）是给机器和历史看的，而 Code Review (CR，代码评审) 是给人看的

一个优质的 Code Review 不仅仅是找 Bug，更是知识共享和团队对齐的过程。以下是一套业界通用的 Code Review 优质格式指南，分为评审者模板和评论标签规范两部分

### 1. 结构化 CR 模板

当你 Review 别人的代码（或者让 AI Review 你的代码）时，建议遵循 "总-分-总" 的结构

#### 评审总结
放在 PR 的最上方，给出一个整体的评价：

> **Summary ( 总结 )**
>
> 这次提交结构很清晰，逻辑没问题。主要实现了 Prompt 的流式输出。
>
> **Highlights ( 亮点 )**
>
> *   用了 `dotenv` 管理环境变量，安全性很好
> *   错误处理（Try-Except）做得不错
>
> **Blockers ( 阻断项 )**:
>
> *   有一个地方硬编码了 Model Name，建议提取出来
> *   缺少了 `requirements.txt` 更新
>
> **Conclusion ( 结论 )**
>
> 总体 LGTM (Looks Good To Me)，修复 Blockers 后即可合并。
>

### 2. 评论标签规范

目前 GitHub 上最流行的格式是行间评论（Inline Comment）。通过在评论开头加标签，明确告知对方这个意见的严重程度。

**常用标签：**

1.  `[Blocking]` / `[Must]` (必改)

    *   含义：代码有逻辑错误、安全漏洞或严重违反规范。不改不能合并
    *   *示例*：`[Blocking] 这里直接把 API Key 打印在日志里了，有安全风险，必须删除。`

2.  `[Suggestion]` / `[Should]` (建议)

    *   含义：代码是对的，但我有更好的写法（如性能更好、更 Pythonic）。不改也可以合并，但建议改
    *   *示例*：`[Suggestion] 这里可以用列表推导式 list comprehension，代码会更简洁。`

3.  `[Question]` / `[Ask]` (疑问)

    *   含义：我不理解这里的逻辑，或者好奇为什么要这么写
    *   *示例*：`[Question] 为什么要在这里设置 sleep(5)？是 API 有速率限制吗？`

4.  `[Nit]` / `[Nitpick]` (吹毛求疵/小点)

    *   含义：微小的细节，如拼写错误、格式缩进。改不改随你，通常是顺手修一下
    *   *示例*：`[Nit] 这里的变量名 'user_nmae' 拼写错了。`

5.  `[Praise]` / `[Nice]` (赞赏)

    *   含义：写得漂亮！这是很多新手忽略的，**正向反馈非常重要**
    *   *示例*：`[Praise] 这个递归逻辑写得很优雅，注释也很清晰！`

### 3. 沟通技巧

优质的 CR 不仅在于格式，还在于沟通。

* ❌ Bad (针对人)：

  "你这里写错了。"

  "你的代码太乱了。"

  "你为什么不加注释？"

*   ✅ Good ( 针对代码，使用"我们" )：

    "这段代码如果在高并发下可能会报错。" (只谈代码)
    
    "我们是否可以把这个函数拆分一下？这样可读性会更好。" (拉进关系)
    
    "这里加一点注释可能会对后续维护更有帮助。" (提出建议而非命令)

### 4. 实战示例

假设你在 Review 队友写的 API 调用代码：

代码片段：

```python
def get_reply(msg):
    k = "sk-123456..." # 硬编码 Key
    res = client.chat.completions.create(
        model="gpt-3.5",
        messages=[{"role":"user","content":msg}]
    )
    return res.choices[0].message.content
```

**优质的 CR 回复：**

> **[Blocking]** 安全隐患：这里硬编码了 `k` (API Key)。请务必改为从 `os.environ` 获取，防止 Key 泄露到 GitHub
>
> **[Suggestion]** 模型名称 `"gpt-3.5"` 建议提取为常量或者配置文件，方便以后升级到 GPT-4
>
> **[Nit]** 变量名 `res` 和 `k` 有点过于简单了，建议改为 `response` 和 `api_key` 以提高可读性
>
> **[Question]** 这里没有做 `try-except` 处理。如果网络超时了，程序会直接崩溃吗？

既然是独立开发者，你可以这样利用这些知识：

1. 自我审查：

   在你提交 Commit 之前，自己假装是另一个人，用上面的标准检查一遍代码。这能极大提升代码质量

2. 让 AI 做 Reviewer：

   你可以把你的代码发给 AI，并使用以下 Prompt：

   > "请作为一个资深的 Python 架构师 Code Review 我的代码。请使用 Conventional Comments 格式（[Blocking], [Suggestion], [Nit]），并重点检查安全性、代码规范和潜在 Bug。"

