---
title: "Git 指令"
published: 2025-11-14
tags: ['git']
first_level_category: "Web全栈开发"
second_level_category: "Git与版本控制"
author: "Alen"
draft: false
---


## 拉取某一仓库并设置为上游仓库

### 情景

对于某个本地仓库，我想要将GitHub上某一仓库作为它的上游仓库并且跟踪它（此时本地仓库并没有远程仓库的任何信息）

### 1. git fetch 拉取历史记录

使用
```bash
git fetch upstream_reposity 
```

会连接远程仓库 upstream_reposity，下载所有本地还没有的数据，并且更新本地仓库的远程跟踪分支

> 远程跟踪分支类似于本地仓库中给远程仓库打上的"标签"，
> 它指向远程仓库的各个分支在上次连接本地仓库时的状态。

但注意：它没有将这些更改应用到当前的工作分支上，代码文件没有任何变化

示例：
```bash
git fetch https://github.com/JuyaoHuang/integrity.git

ouputs:
remote: Enumerating objects: 179, done.
remote: Counting objects: 100% (176/176), done.
remote: Compressing objects: 100% (103/103), done.
remote: Total 179 (delta 99), reused 146 (delta 73), pack-reused 3 (from 1)
Receiving objects: 100% (179/179), 70.07 KiB | 646.00 KiB/s, done.
Resolving deltas: 100% (99/99), completed with 18 local objects.
From https://github.com/JuyaoHuang/integrity
 * branch            HEAD       -> FETCH_HEAD
```

当使用一个**完整的URL**来进行 fetch操作时，Git 会将远程的数据下载下来，但它只会把这些信息保存在一个临时的地方叫做 FETCH_HEAD，不会自动创建或更新一个叫做 origin/main 的本地跟踪分支。

如果 fetch时使用的是仓库的名字，说明本地仓库里存有该仓库的 URL，即已跟踪的上游仓库的URL，假设该远程仓库名字为 origin，那么可直接跳过第2、3步

现在拿到远程仓库的分支和代码后，下一步就是要将它应用到当前分支，假设为 main中来。


### 2. git remote 添加远程仓库别名

1. 检查现有远程仓库的内容
    ```bash
    git remote -v
    ```

2. 为远程仓库的URL设置别名

    ```bash
    git remote add origin https://github.com/JuyaoHuang/integrity.git
    ```

    或者

    ```bash
    git remote add origin git@github.com:JuyaoHuang/integrity.git
    ```

    该命令行对于一个仓库只用运行一次

3. 从远程仓库拉取最新代码和分支

    ```bash
    git fetch origin
    ```

    输出：
    ```bash
    git fetch origin
    From https://github.com/JuyaoHuang/integrity
    * [new branch]      courier         -> origin/courier
    * [new branch]      heimdall-readme -> origin/heimdall-readme
    * [new branch]      main            -> origin/main
    * [new branch]      sentinel-readme -> origin/sentinel-readme
    ```

### 3. git branch设置上游分支

1. 检查远程分支是否存在
   
   ```bash
   git branch -r
   ```
   
   输出：
   ```bash
     origin/HEAD -> origin/main
     origin/courier
     origin/heimdall-readme
     origin/main
     origin/sentinel-readme
   ```

2. 设置上游分支
   
   如果本地仓库当前命令行窗口没有位于 main分支，先执行 `git checkout main`切换到 main分支
   ```bash
   git branch --set-upstream-to=origin/main main
   ```
   其中
   - --set-upstream-to=origin/master: 这是核心参数，用于建立“跟踪”（tracking）关系。
   - origin：本地给上游仓库名称的重命名
   - origin/main的 main：上游仓库的某个分支名
   - main：本地仓库分支名
   

接下来可以直接使用`git pull`拉取内容，或者使用`git merge`

### 4. git merge将远程仓库内容覆盖到当前分支

使用
```bash
git merge <branch_name>
```
将远程仓库的分支 branch_name上的更改合并到当前分支 main上

```bash
git checkout main
git merge origin/main
```

将远程仓库 origin的 main分支代码合并到 本地仓库的 main分支

---

`git pull` 指令实际上就是 `git fetch` 和 `git merge` 1、2这两步的便捷语法：

```bash
git pull origin main = git fetch origin + git merge origin/main
```

因此，如果设置好了上游仓库，就可以使用 `git pull` 快速拉取

---

## 为分支设置上游仓库

### 1. git remote -v 检查远程仓库状态

```bash
git remote -v
```

检查远程仓库状态并获取远程跟踪分支：

```bash
origin  https://github.com/JuyaoHuang/integrity.git (fetch)
origin  https://github.com/JuyaoHuang/integrity.git (push)
```

可知此时 URL=https://github.com/JuyaoHuang/integrity.git的本地标签别名是 origin

### 2. git branch --set-upstream-to设置上游仓库

1. 切换到要跟踪的分支 

   ```bash
   git checkout main
   ```

2. 设置要跟踪的上游仓库

   ```bash
   git branch --set-upstream-to=origin/main main
   ```

   - "origin/main"：上游仓库的分支名
   - "main"：本地仓库分支

### 3. git branch --unset-upstream  取消跟踪上游仓库

```bash
git branch --unset-upstream <branch_name>
```

取消跟踪某一分支的上游仓库

```bash
git branch --unset-upstream origin/main
```

---

## 回退 commit

### 1. 硬回退

```bash
git reset --hard HEAD^
```

该 commit 回退会撤销 commit 的同时，**将 commit 中对仓库的修改一并删除**。

`HEAD^` 表示回退到上一个版本；`--hard` 表示连同文件内容一起恢复原状。

例如：某一 commit 编号为 B，提交的内容有 3 个 files。执行该指令后这三个文件的修改的内容都会被删除。

### 2.软回退

顾名思义，就是撤销 "commit" 这个动作，但保留本地仓库对代码的更改。

```bash
git reset --soft HEAD^
```

### 示例

```bash
commit d95c91b977f7c2a7e151f8ed183fbd875c4107c4 (HEAD -> re_format, origin/main, origin/HEAD, main)
Merge: 83d9834 d833c26
Author: Alen Nelson
Date:   Fri Nov 21 21:05:52 2025 +0800

    Merge pull request #10 from JuyaoHuang/s3

    doc:add session 3 of 信息论
commit d833c268c386e8fa104ae71ebccb052c86ca88fa (origin/s3)
Author: Juyao Huang 
Date:   Fri Nov 21 21:03:42 2025 +0800

    doc:add session 3 of 信息论
```

如果执行 `git reset --hard HEAD^`，那么 hash 为 `d95c91b977f7c2a7e151f8ed183fbd875c4107c4`的 commit 就会被撤回，并且**修改的内容会被删除**。如果两个 commit 之间没有执行过 `git pull`，那么执行 `git reset --hard HEAD^` 后仓库的内容会回退到 hash 为 `d833c268c386e8fa104ae71ebccb052c86ca88fa`的 commit 的状态。

如果执行的是 `git reset --soft HEAD^`，那么 hash 为 `d95c91b977f7c2a7e151f8ed183fbd875c4107c4`的 commit 就会被撤回，但是修改的内容不会被删除。

## 编写长段 Commit

如果有编写一个完整的、长段的 commit，使用 `git commit`指令（不带 `-m' 参数）。

这会打开一个终端文本器：nano 或者 vim。

nano较为常见，此处不多描述。

执行

```bash
git commit
```

进入 vim 的编辑页面： 如果你看到一个光标，但输入字符没反应，那你很可能在 Vim 中。

1. 按下 " i " 键进入插入模式（ insert ），然后开始编写 commit
2. 写完后，按下 "ESC" 键退出插入模式，并输入 `:wq` 然后回车。（`w`: write, `q`: quit）

**commit 规范**：

一个规范的 Commit Message 分为三个部分：

```bash
<类型>(<范围>): <简短描述>   <- 主题行，不超过50个字符
<空一行>                     <- 主题行和正文之间必须有一个空行
<详细描述的正文>              <- 这是你的长段描述。详细解释为什么要做这个修改，
解决了什么问题，带来了什么影响。为了可读性，建议每行不超过72个字符。
<空一行>
<页脚 (Footer)>              <- 可选。通常用于关联 Issue，例如：Fixes #123
```

例如：

```bash
feat(api): Add user authentication endpoint

Implement JWT-based authentication for the /api/users/login route.
This new endpoint accepts a username and password, validates them against
the database, and returns a JSON Web Token if successful.

- Uses passlib for password hashing and verification.
- Token expires in 24 hours.
- Adds new dependencies: python-jose and passlib.

Resolves: #42
```

