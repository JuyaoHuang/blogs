---
title: "GPG 签名"
publishDate: 2026-01-18
description: "GPG Signature 的介绍"
tags: ['git']
language: 'Chinese'
first_level_category: "Web全栈开发"
second_level_category: "Git与版本控制"
draft: false
---

本篇文章介绍 GPG 签名，以及如何使用 GPG 签名进行验证。

## GPG 签名

GPG Signature 的核心作用就是给提交的代码盖上防伪印章。

在开源协作和软件开发的世界里，Git 是最常用的工具。但 Git 有一个设计上的“漏洞”：它允许任何人使用任何名字和邮箱提交代码。这意味着：有人可以在提交记录中冒充开发者，甚至悄悄修改代码内容，而系统默认无法分辨真假。

GPG 签名就是为了解决这个问题而生的。它相当于在数字世界里，给每一次代码提交盖上了一个独一无二且无法被伪造的防伪印章。

### 核心原理

GPG 签名的工作机制依赖于**非对称加密**技术，也就是每个人都拥有一对钥匙：

- 私钥：开发者私有，严格保密。
- 公钥：对外公开，任何人都可以获取到。

### 签名的产生过程

当开发者完成一段代码并执行 Commit 时，如果开启了 GPG 签名，计算机会使用**私钥**对这段代码的内容进行加密运算。这个运算的结果就是签名。

就像开发者在信封封口处，用自己的私章盖了一个火漆印。

### GPG 验证

当代码被推送到 GitHub 或 GitLab 等平台时，平台会拿出开发者预先上传的**公钥**，对这个火漆印进行核验。

核验的过程会确认两件事：
1. 这个印章确实是该名开发者用私钥盖上去的，不是别人冒充。也就是推送者的身份真实正确。
2. 信封在传输过程中没有被拆开过，里面的代码连一个标点符号都没有被篡改。如果代码被修改，这个数字印章就会立即失效，验证失败。

### 重要性

在没有 GPG 签名的情况下，Git 记录中的作者信息就像是用铅笔写的名字，任何人都能擦掉重写。而使用了 GPG 签名后，代码提交就拥有了法律级别的可追溯性。

对个人开发者来说，在 GitHub 上，经过验证的提交会显示一个绿色的 `[Verified]` 标签，代表这是本人操作。

**而对于团队来说**，它防止了恶意攻击者伪造核心维护者的身份混入恶意代码，确保了供应链的安全。

GPG 签名并非复杂的黑科技，它本质上就是一种数字身份认证机制。它确保了在虚拟的代码世界里，大家看到的“谁写了什么”，就是真实的“谁写了什么”。

## GPG 实现

以下是如何使用 GPG 进行从生成密钥到签名文件，再到验证签名的完整流程指南。

### 1. 安装 GPG 工具

在开始之前，确保系统中已安装 GnuPG（GPG）工具。
- Windows：安装 `Gpg4win`
- macOS：运行 `brew install gnupg`
- Linux：`sudo apt install gnupg` (Debian/Ubuntu) 或 `sudo yum install gnupg` (CentOS/RHEL)

### 2. 生成与发布密钥

在使用签名之前，需要拥有一对密钥（公钥和私钥），并将公钥发布到公共服务器，以便他人获取。

#### **2.1. 生成密钥对**

在终端（Terminal 或 PowerShell）中输入以下命令：
```bash
gpg --full-generate-key
```
*   加密算法：默认 `RSA and RSA`
*   密钥长度：推荐选择 `4096` 位
*   有效期：可以选择 `0`（永不过期）或设置具体的年限（如 `2y`）
*   用户信息：输入真实的姓名和邮箱（需与 Git 或其他用途匹配）
*   密码：设置一个强密码来保护私钥

![2](2.jpg)

#### **2.2. 获取密钥 ID**

生成完成后，查看密钥 ID：

```bash
gpg --list-keys
```

```powershell
(base) PS C:\Users\Alen> gpg --list-keys
[keyboxd]
---------
pub   rsa4096 2026-01-16 [SC]
      A1B2C3D4E5F6 ... 123456789ABCDEF0  <- 这一长串是“指纹”
uid           [ultimate] your_host <your_mail@example.com>
sub   rsa4096 2026-01-16 [E]
      E4CC56666DDDDD6DG62G191GF7706AEF4
```
其中 `123456789ABCDEF0`（指纹的后16位或8位）即为 Key ID。

> 对一个密钥添加多个邮箱：
>
> 在 GPG 中，一个 GPG 密钥对可以绑定多个用户标识。
>
> 在终端输入：`gpg --edit-key [Key-ID]`
>
> 终端会进入一个交互式界面，提示符变为 `gpg>`
>
> 在交互提示符下输入：`gpg> adduid`
>
> 系统会依次提示输入：
> 1.  Real name：建议与之前的保持一致
> 2.  Email address：输入新的邮箱
> 3.  Comment：可选，如 "Work Email"
>
> 输入完成后，确认信息并输入 `O`（Okay）。然后执行 `gpg> save` 保存。
>
> **注意**：在本地修改了密钥（添加了新邮箱）后，公钥服务器上的旧版本并不会自动更新。如果不上传新版本，别人在验证新邮箱的签名时会报错

#### **3. 将公钥上传到服务器**

为了让别人能验证你的签名，需要将公钥上传到全球通用的 OpenPGP 密钥服务器（如 keys.openpgp.org 或 keyserver.ubuntu.com）。

```bash
# 替换 [Key-ID] 为你实际的 ID
gpg --keyserver hkps://keys.openpgp.org --send-keys [Key-ID]
```

上传之后别人就可以在相关的服务器上验证你的 GPG 密钥。

![3](./3.jpg)

### 3. 上传公钥到 GitHub

GitHub 需要先拥有公钥副本，才能在收到代码时进行核对。

#### **3.1. 导出公钥副本**

在本地终端执行以下命令，导出 ASCII 格式的公钥：

```bash
# 替换 [Key-ID] 为你的密钥 ID
gpg --armor --export [Key-ID]
```

复制整段内容，包括头尾的 `BEGIN PGP PUBLIC KEY BLOCK` 和 `END ...`。

#### **3.2. 上传至 GitHub**

1.  登录 GitHub，点击右上角头像 -> Settings
2.  在左侧菜单找到 SSH and GPG keys
3.  点击 New GPG key 按钮
4.  在输入框中粘贴刚才复制的公钥内容
5.  点击 Add GPG key

### 4. 配置本地 Git 启用签名

你需要告诉本地的 Git 工具：“使用这把钥匙给我的每一次提交盖章”。

#### **4.1. 关联密钥 ID**

```bash
# 告诉 Git 使用哪个 Key ID 进行签名
git config --global user.signingkey [Key-ID]
```

#### **4.2. 开启自动签名**

```bash
# 告诉 Git 每次 commit 时自动进行 GPG 签名
git config --global commit.gpgsign true
```

可验证关联的 GPG Key：

```bash
git config --global user.signingkey
```

