---
title: "Git实践"
published: 2025-10-03
tags: ['git']
first_level_category: "Web全栈开发"
second_level_category: "Git与版本控制"
description: "项目开发时遇到的Git交互问题"
draft: false
---

## 1. 分支无共同git历史

### Q1:  分支提交历史不同

​	原仓库没有dev分支，在原仓库创建Pull request只能提交B仓库的main分支，但是修改的代码都在B仓库的dev分支。将B仓库的dev分支pull request到原仓库的main分支会出现上面的错误

核心问题： 正如 GitHub 错误信息所说，fork 的仓库 **origin_fork** 的 dev 分支，和原仓库 **origin** 的 main 分支，**没有任何共同的 Git 历史**

**用一个比喻来解释这个问题**

- 正确的流程（共享历史）：

  ​	你拿到一本书（git clone），翻到第 50 页，在上面做了笔记和修改（git commit），然后把这一页撕下来给原作者，说：“请把我的修改更新到你的书里”。原作者能清晰地看到你是在他第 50 页的基础上做的修改，所以他可以比较和合并。

- 现在的情况（不同历史）：

  ​	你自己从一张白纸开始，写了一本全新的书（git init），只是这本书的内容恰好和原作者的书很像。现在你把你这本书的第 50 页撕下来给原作者，说：“请把我的修改更新到你的书里”。原作者会彻底困惑，因为你这一页的“历史”和他书里的任何一页都对不上。他无法进行比较，因为你们没有一个共同的起点。

这就是 GitHub 报错 entirely different commit histories 的原因

**解决方案：将代码“移植”到拥有正确历史的分支上**

1. 在电脑上创建一新的文件夹 A

2. 克隆原仓库 origin ----- **新的本地仓库叫做 local_repo**

3. **现在这个仓库有了一个正确的 git历史**

4. cd origin 后，创建一个新的分支 dev

   ```bash
   git checkout -b dev
   ```

5. 此时，dev 分支的代码和 **orgin** 仓库的 main 分支代码一样

6. 接下来，从原先 fork的仓库 **origin_fork** 的 dev分支获取已经提交（原本就要提交给 **origin**仓库）的代码

7. 先将 **orgin_fork**作为另一个上游仓库

   ```bash
   git remote add my_fork <你的 origin_fork 仓库的 URL>
   ```

8. 从新的远程地址下载仓库的代码，**但不合并**

   ```bash
   git fetch my_fork
   ```

9. 将当前文件夹里的所有文件，使用 my_fork 这个远程仓库的 dev分支里的文件覆盖

   ```bash
   git checkout my_fork/dev -- .
   ```

   " . "代表当前目录

10. **现在新的本地仓库 local_repo有了完整的git历史和正确的代码**

11. 创建一个新的、干净的提交

    ```bash
    git add .
    git commit -m "your commmit"
    ```

12. 推送到 fork的远程仓库**origin_fokr**

    ```bash
    git push --force my_fork dev
    ```

    - **警告**：这会**永久覆盖** B 仓库 dev 分支之前的历史

13. 建立 Pull request

------

**如果不想覆盖掉原来的远程仓库 origi_fork的 dev分支（保留着你要 pull的代码）**，那么在**第 12.步**：

```bash
git push --force my_fork dev:other_dev
```

这会在 **origin_fork** 仓库新建一个 other_dev分支，将你的代码推送到此处

## 2. 分支合并

### Q1:常规合并

1. 更新要合并的分支，保证分支都是最新的

   ```bash
   git checkout main
   git pull origin main
   
   git checkout dev
   git pull origin dev
   ```

2. 检查 dev分支是否存在潜在冲突

   - 为了确保最终合并到 main 分支时绝对不会有任何冲突，一个专业的做法是**先将 main 分支的最新更改合并到 dev 分支**。

   - 在 dev分支上运行

     ```bash
     git merge main
     ```

   - 分析结果：

     - 如果没有任何输出或提示 "Already up to date."  => dev 分支已经包含了 main 的所有历史，最终合并时将会非常顺畅。
     - 如果出现了合并冲突 (Merge Conflicts)  =>这意味着在 dev 分支这个“工作区”里发现了问题。

3. 如果没有合并冲突，执行

   ```bash
   git checkout main
   git merge dev
   ```

   进行合并

4. 推送到上游仓库（可选）

   ```bash
   git push origin main
   ```

### Q2: 不同历史的合并

执行 `git merge master`时报错:

```bash
fatal: refusing to merge unrelated histories
```

**问题根源：两个“独立”的历史提交点**

​	本地的 master 分支和 dev 分支，在 Git 看来，是两个**完全不相干**的项目。它们没有一个共同的初始提交点（没有共同的祖先）。

通常发生在以下情况：

1. 在一个分支上 git init 之后，又用 git checkout --orphan <新分支> 创建了一个全新的、无历史的孤儿分支
2. 在不同的文件夹里分别初始化了项目，然后尝试将它们合并

**解决方案：统一历史**

如果 dev分支的开发已经完成而且稳定，那么强行让 **master** 分支放弃它自己的历史，完全变成和 **dev** 分支一模一样

即用 dev分支直接覆盖 master分支

1. 切换到 dev分支

   ```
   git checkout dev
   ```

2. 确保 dev与远程仓库同步

   ```bash
   git push origin dev
   ```

3. 切换到 master分支

4. 执行 reset指令

   ```bash
   git reset --hard dev
   ```

   - 忘记 master 分支现在的一切，把 master 的指针，**直接、强行地**移动到 dev 分支当前指向的那个提交上
   - --hard 参数会同时更新工作区的文件，让它们也变得和 dev 分支完全一样。

5. 运行`git log --oneline`检查情况

6. 执行

   ```bash
   git push origin master:master --force
   ```

   强制推送本地 master分支覆盖远程仓库的 master分支



## 3. 清理混乱的提交历史

### 情景

开发分支上有多个零碎的 commit（包括旧的提交、新的提交、合并产生的 Merge 节点），想要在 PR 中只显示**一个**干净的 commit。

### 解决方法（软回退）

**前提**：确保你的 main 分支是正确的，没有做过任何错误的修改

1. 切换到开发分支：`git checkout dev`
2. 连接远程仓库：`git fetch origin`。origin 为本地给远程仓库起的别名，使用 `git remote -v`查看别名
3. **软回退**：`git reset --soft origin/main`。软回退用于确保你的所有提交的代码都能够存在
4. 只提交一个 commit：`git commit -m " "`。此时可使用 `git log` 查看提交记录
5. **强制**推送到远程分支：`git push -f origin dev`

**原理**：实际上就是将分支的历史指针直接重置到 main 的位置，但**保留所有文件的修改在暂存区**，从而允许你将其重新打包成一个新的 commit



## 4. 拉取某一仓库并设置为上游仓库

### 情景

对于某个本地仓库，要将GitHub上某一仓库作为它的上游仓库并且跟踪它（此时本地仓库并没有远程仓库的任何信息）

### 解决方法

#### 1. git fetch 拉取历史记录

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

如果 fetch 时使用的是仓库的名字，说明本地仓库里存有该仓库的 URL，即已跟踪的上游仓库的URL，假设该远程仓库名字为 origin，那么可直接跳过第2、3步

现在拿到远程仓库的分支和代码后，下一步就是要将它应用到当前分支，假设为 main中来。


#### 2. git remote 添加远程仓库别名

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

#### 3. git branch设置上游分支

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

#### 4. git merge将远程仓库内容覆盖到当前分支

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

## 5. 为分支设置上游仓库

### 情景

分支缺少要跟踪的上游仓库，无法使用`git pull` 代替 `git fetch` 和 `git merge` 快速拉取仓库

### 解决方法

#### 1. git remote -v 检查远程仓库状态

```bash
git remote -v
```

检查远程仓库状态并获取远程跟踪分支：

```bash
origin  https://github.com/JuyaoHuang/integrity.git (fetch)
origin  https://github.com/JuyaoHuang/integrity.git (push)
```

可知此时 `URL=https://github.com/JuyaoHuang/integrity.git` 的本地标签别名是 origin

#### 2. git branch --set-upstream-to 设置上游仓库

1. 切换到要跟踪的分支 

   ```bash
   git checkout main
   ```

2. 设置要跟踪的上游仓库

   ```bash
   git branch --set-upstream-to=origin/main main
   ```

   - `origin/main`：上游仓库的分支名
   - `main`：本地仓库分支

#### 3. git branch --unset-upstream  取消跟踪上游仓库

```bash
git branch --unset-upstream <branch_name>
```

取消跟踪某一分支的上游仓库

```bash
git branch --unset-upstream origin/main
```



## 6. 回退 commit

### 1. 硬回退

```bash
git reset --hard HEAD^
```

该 commit 回退会撤销 commit 的同时，**将 commit 中对仓库的修改一并删除**

`HEAD^` 表示回退到上一个版本；`--hard` 表示连同文件内容一起恢复原状

例如：某一 commit 编号为 B，提交的内容有 3 个 files。执行该指令后这三个文件的修改的内容都会被删除

### 2.软回退

顾名思义，就是撤销 "commit" 这个动作，但保留本地仓库对代码的更改

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

如果执行 `git reset --hard HEAD^`，那么 hash 为 `d95c91b977f7c2a7e151f8ed183fbd875c4107c4`的 commit 就会被撤回，并且**修改的内容会被删除**。如果两个 commit 之间没有执行过 `git pull`，那么执行 `git reset --hard HEAD^` 后仓库的内容会回退到 hash 为 `d833c268c386e8fa104ae71ebccb052c86ca88fa`的 commit 的状态

如果执行的是 `git reset --soft HEAD^`，那么 hash 为 `d95c91b977f7c2a7e151f8ed183fbd875c4107c4`的 commit 就会被撤回，但是修改的内容不会被删除。

## 7. 编写长段 Commit

如果有编写一个完整的、长段的 commit，使用 `git commit`指令（不带 `-m' 参数）

这会打开一个终端文本器：nano 或者 vim

nano较为常见，此处不多描述

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

## 8. 回退已合并的代码

### 情景

PR 已经合并到 main 分支，但发现有问题，需要撤销。

### 解决方法 revert/reset

**1. 团队合作**：

团队合作的情况下，**千万不要擅自使用 reset 回退**。如果错误合并后，无法及时通知其他成员，需要使用更安全的 revert 进行回退。

因为 revert 会保留错误的 commit 和 PR 记录，方便其他成员知道错误在哪，也避免了 reset 时有其他成员在你错误的 PR 后的历史上 commit 的失踪。 

**操作**：在网页端 GitHub，你提交的 PR 下方有一个 "revert" 按钮，点击它，编写 Revert 的 commit。或者使用 `git revert -m l <commit_hash>`，生成一个新的 commit，指示指针回退到原先的修改，保留历史记录。

**2. 个人/清理历史**：

因为个人不需要考虑 "reset 时有其他成员在你错误的 PR 后的历史上 commit 的失踪" 的问题，可以大胆地用 `reset` 进行**抹除式回退**。

使用 reset 有一个**好处**：可以清理掉以前提交的错误的 commit。

例如

![1](./1.jpg)

（血的教训！）网页端 GitHub 存储了对该分支的一切 commit 记录。红框部分是笔者错误地提交 PR 后，又进行 revert 的记录。

但在仓库`.git`中：

1. 在 main 分支 执行 `git pull`，将最新的代码，包括提交记录一并拉取下来。此时提交历史里就有红框这两个我不希望它存在的 PR commit。

2. 执行 `git log` 获取到我希望回退到的 commit 的代码状态的哈希值

   ```bash
   commit c32506a95add12f3696a1d153ba8229c80c6eac9
   Author: Juyao Huang <tinghua5226@gamil.com>
   Date:   Thu Nov 27 18:46:21 2025 +0800
   
       doc:add "torchvision.transforms" doc.
   ```

3. **执行 `git reset --hard c32506a95add12f3696a1d153ba8229c80c6eac9` 硬回退指令**，将仓库的内容强行回退到该 commit 时的状态。**之前的 commit 的代码全部丢失** 。

   > 在这步之前应执行 `git checkout -b main_bak`保存一下当前的代码，避免丢失关键代码

**总结**

如果是团队合作，使用 revert 即可，毕竟要照顾团队成员。如果是个人项目，**在确保你回退之前的 commit 的确没有新增的代码的前提下**，并且希望删除不必要的提交 commits，再执行`git reset --hard <commit_hash>`

## 8. 需要上传大文件到 GitHub (GLFS)

### 情景

上传大文件到 GitHub 时，GitHub 拒绝你的访问。当使用 `git commit` 提交有关大型文件（大于100M）到 GitHub 上时，会被 GitHub 默认拒绝。需要使用 `Git-LFS` 工具创建指针，上传大文件。

```bash
remote: warning: File experience_two/codes/best_food_cnn.pth is 99.76 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB
remote: error: Trace: 9cf673a5c08a9ccb205aba61f85967171eba83989fd6227c8e502f2cc351b044
remote: error: See https://gh.io/lfs for more information.
remote: error: File experience_two/codes/best_vgg16.pth is 512.35 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.  
To https://github.com/JuyaoHuang/BUPTDeepLearning.git
 ! [remote rejected] main -> main (pre-receive hook declined)
```

### 解决方法   Git-LFS

1. **安装 Git LFS**

   选择一条路径完成：

   1. 前往[Git-LFS官网](https://git-lfs.com/)下载 LFS
   2. 安装最新版本的 Git-for-Windows。

   在仓库下执行：`git lfs install`

   ```bash
   (base) PS D:\Learing_Resourses> git lfs install
   Updated Git hooks.
   Git LFS initialized.
   ```

2. **处理已经提交的大文件**

   即使现在配置了 Git LFS，因为**这个大文件已经存在于本地 commit 记录中了**，直接 push 依然会失败。你需要先清洗你的本地提交记录，把大文件转交给 LFS 管理。

   因为已经对这些大文件进行了 git commit ，普通的 git lfs track 对**过去的**提交无效。需要使用 `migrate` 命令把历史记录里的 .pth 文件从普通 Git 对象转换成 LFS 对象。

   ```bash
   git lfs migrate import --include="*.pth" --everything
   ```

3. **继续推送**

   ```bash
   git status
   git push origin main
   ```

**如果没有对大文件进行过提交**：

1. 使用 `git lfs track ".lfs"` 指令提示 git 将 `.lfs` 文件使用 LFS管理

   ```bash
   git lfs track "*.pth"
   ```

2. 确保`.gitattributes` 被添加进暂存区

   ```bash
   git add .gitattributes
   ```

   > 需要先暂存 ` .gitattributes` 管理文件，再暂存大文件！

3. 重新添加大文件

   ```bash
   git add .
   ```

4. 提交 commit 并进行推送

   ```bash
   git commit -m "your message."
   git push 
   ```

