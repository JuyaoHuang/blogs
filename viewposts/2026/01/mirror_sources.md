---
title: "常用镜像源"
publishDate: 2026-01-21
description: "常用镜像源服务介绍"
tags: ['news']
language: 'Chinese'
first_level_category: "资讯与生活"
second_level_category: "前沿快讯"
draft: false
---

## NPM 镜像源

### 使用方式

临时使用：
```bash
npm install <package-name> --registry=<镜像源地址>
```

全局设置：
```bash
npm config set registry <镜像源地址>
```

### 1. 淘宝/阿里云镜像源

```bash
https://registry.npmmirror.com/
```

特点：由阿里云维护，原淘宝 npm 源升级版。更新速度快，几乎实时同步官方。支持 npm、yarn、pnpm 等全生态。

适用场景：国内服务器 / CI 环境。日常开发环境。

```bash
npm install <package> --registry=https://registry.npmmirror.com/
```

### 2. 腾讯云镜像

```bash
https://mirrors.cloud.tencent.com/npm/
```

特点：腾讯云开源镜像站，稳定可靠。适合部署在腾讯云的环境中。

### 3. 华为云镜像

```bash
https://mirrors.huaweicloud.com/repository/npm/
```

特点：华为云开源镜像站，国内节点多。提供更高的带宽和稳定性。

### 4. 官方源

```bash
https://registry.npmjs.org/
```

特点：npm 官方唯一源，最权威，保证第一时间更新。海外访问较慢，国内常遇到超时问题。

## PIP 镜像源

### 使用方式

临时使用：
```bash
pip install <package-name> -i <镜像源地址>  
```

全局设置：
```bash
pip config set global.index-url <镜像源地址>
```

### 1. 清华大学镜像源

```bash
https://pypi.tuna.tsinghua.edu.cn/simple/
```

特点：高校公益镜像，稳定可靠，学术氛围浓厚。同步频率快，稳定性高，适合科研和学习用途。

适用场景：校园网/教育网环境，作为首选或备用源，科研项目。

```bash
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. 中科大镜像源

```bash
https://pypi.mirrors.ustc.edu.cn/simple/
```

特点：高校镜像，免费稳定，教育网环境下访问速度快。更新速度略慢于商业源，但稳定性好。

### 3. 腾讯云镜像源

```bash
https://mirrors.cloud.tencent.com/pypi/simple/
```

特点：腾讯云开源镜像站，稳定可靠，适合部署在腾讯云的环境中。同步频率快，稳定性高。

### 4. 华为云镜像源

```bash
https://mirrors.huaweicloud.com/repository/pypi/simple/
```

特点：华为云开源镜像站，国内节点多，提供更高的带宽和稳定性。适合企业级应用。

### 5. 阿里云镜像源

```bash
https://mirrors.aliyun.com/pypi/simple/
```

### 推荐顺序

根据 稳定性、更新频率、速度 排序推荐：

1. 清华 TUNA → https://pypi.tuna.tsinghua.edu.cn/simple
2. 阿里云 → https://mirrors.aliyun.com/pypi/simple/
3. 腾讯云 / 华为云
4. 中科大 → https://pypi.mirrors.ustc.edu.cn/simple/

