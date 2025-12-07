---
title: "vim"
published: 2025-12-07
tags: ['git']
first_level_category: "Web全栈开发"
second_level_category: "运维与Linux"
description: "编辑器 vim 的使用教程"
draft: false
---

## vim 在 git commit 下的使用

### 1. 进入编辑模式

当输入指令 git commit 后，会默认进入 vim 的命令模式页面。此时需要按下键盘的 `i` 键，即可进入编辑模式。（屏幕左下角会显示 -- INSERT -- (或 -- 插入 --) 字样）

### 2. 退出编辑模式

按下键盘的 `Esc` 键即可退出，回到命令模式。

### 3. 保存并退出

依次输入： `:wq`，然后回车。

- w：write 写模式
- q：quit 退出

### 4. 不保存并退出（可选）

如果不小心写错了，并想放弃提交：

1. 按下 `Esc`
2. 输入 `:q!` 并按下回车键

提交终止。