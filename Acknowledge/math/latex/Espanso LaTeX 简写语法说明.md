---
title: 'Espanso LaTeX 简写语法说明'
publishDate: 2026-04-10
description: "使用 Espanso 文本展开器在 Typora 中快速输入 LaTeX 数学公式"
tags: ['latex', 'espanso', 'typora']
language: 'Chinese'
first_level_category: "知识库"
second_level_category: "数学工具"
draft: false
---

## Espanso LaTeX 简写语法说明

### 什么是 Espanso

[Espanso](https://espanso.org/) 是一个开源、跨平台的**系统级文本展开器**。它可以在任何应用程序中（包括 Typora）监听你输入的缩写，并自动替换为完整的文本。

利用 Espanso，我们可以用极短的缩写快速输入 LaTeX 数学公式，避免手打 `\frac{}{}`、`\sum_{i=1}^{n}` 等冗长命令。

### 安装与配置

1. 从 [espanso.org](https://espanso.org/install/) 下载安装
2. 配置文件位于 `C:\Users\<用户名>\AppData\Roaming\espanso\`
3. 简写规则文件：`match\latex-math.yml`
4. 修改配置后执行 `espanso restart` 生效

### 使用规则

- 所有触发词以 **分号 `;`** 开头，避免与正常输入冲突
- 输入触发词后，Espanso 会自动将其替换为对应的 LaTeX 命令
- 在 Typora 的 `$...$` 或 `$$...$$` 数学环境内使用

### 速查表

#### 1. 基本算术与关系运算符

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;ti` | `\times` | 乘号 $\times$ |
| `;cd` | `\cdot` | 点乘 $\cdot$ |
| `;div` | `\div` | 除号 $\div$ |
| `;pm` | `\pm` | 加减号 $\pm$ |
| `;mp` | `\mp` | 减加号 $\mp$ |
| `;ap` | `\approx` | 约等于 $\approx$ |
| `;ne` | `\neq` | 不等于 $\neq$ |
| `;eq` | `\equiv` | 恒等于 $\equiv$ |
| `;le` | `\leq` | 小于等于 $\leq$ |
| `;ge` | `\geq` | 大于等于 $\geq$ |
| `;ll` | `\ll` | 远小于 $\ll$ |
| `;gg` | `\gg` | 远大于 $\gg$ |
| `;sim` | `\sim` | 相似/渐近 $\sim$ |
| `;prop` | `\propto` | 正比于 $\propto$ |

#### 2. 分数、根号

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;ff` | `\frac{}{}` | 分数 $\frac{a}{b}$ |
| `;sq` | `\sqrt{}` | 根号 $\sqrt{x}$ |
| `;nsq` | `\sqrt[]{}` | n 次方根 $\sqrt[3]{x}$ |

#### 3. 积分、求和、极限、连乘

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;int` | `\int_{}^{}` | 定积分 $\int_{a}^{b}$ |
| `;iint` | `\iint` | 双重积分 $\iint$ |
| `;iiint` | `\iiint` | 三重积分 $\iiint$ |
| `;oint` | `\oint` | 线积分 $\oint$ |
| `;inf` | `\infty` | 无穷大 $\infty$ |
| `;sum` | `\sum_{i=1}^{n}` | 求和 $\sum_{i=1}^{n}$ |
| `;prod` | `\prod_{i=1}^{n}` | 连乘 $\prod_{i=1}^{n}$ |
| `;lim` | `\lim_{x \to }` | 极限 $\lim_{x \to}$ |
| `;liminf` | `\lim_{x \to \infty}` | 趋于无穷极限 $\lim_{x \to \infty}$ |

#### 4. 括号（自适应大小）

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;lr(` | `\left( \right)` | 自适应圆括号 |
| `;lr[` | `\left[ \right]` | 自适应方括号 |
| `;lr{` | `\left\{ \right\}` | 自适应花括号 |
| `;lr\|` | `\left\| \right\|` | 自适应绝对值 |

#### 5. 微分与导数

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;dydx` | `\frac{dy}{dx}` | 普通导数 $\frac{dy}{dx}$ |
| `;ddx` | `\frac{d}{dx}` | 微分算子 $\frac{d}{dx}$ |
| `;par` | `\partial` | 偏导符号 $\partial$ |
| `;pyx` | `\frac{\partial y}{\partial x}` | 偏导数 $\frac{\partial y}{\partial x}$ |
| `;nab` | `\nabla` | 梯度 $\nabla$ |
| `;dd` | `\mathrm{d}` | 正体微分 d |

#### 6. 希腊字母（小写）

| 触发词 | 展开结果 | 触发词 | 展开结果 |
|--------|----------|--------|----------|
| `;al` | `\alpha` | `;mu` | `\mu` |
| `;be` | `\beta` | `;nu` | `\nu` |
| `;ga` | `\gamma` | `;xi` | `\xi` |
| `;de` | `\delta` | `;pi` | `\pi` |
| `;ep` | `\epsilon` | `;rh` | `\rho` |
| `;ze` | `\zeta` | `;si` | `\sigma` |
| `;et` | `\eta` | `;ta` | `\tau` |
| `;th` | `\theta` | `;up` | `\upsilon` |
| `;io` | `\iota` | `;ph` | `\phi` |
| `;ka` | `\kappa` | `;vp` | `\varphi` |
| `;la` | `\lambda` | `;ch` | `\chi` |
| | | `;ps` | `\psi` |
| | | `;om` | `\omega` |

#### 7. 希腊字母（大写）

| 触发词 | 展开结果 | 触发词 | 展开结果 |
|--------|----------|--------|----------|
| `;Ga` | `\Gamma` | `;Si` | `\Sigma` |
| `;De` | `\Delta` | `;Ph` | `\Phi` |
| `;Th` | `\Theta` | `;Ps` | `\Psi` |
| `;La` | `\Lambda` | `;Om` | `\Omega` |
| `;Xi` | `\Xi` | `;Pi` | `\Pi` |

#### 8. 集合与逻辑符号

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;in` | `\in` | 属于 $\in$ |
| `;ni` | `\notin` | 不属于 $\notin$ |
| `;sub` | `\subset` | 子集 $\subset$ |
| `;sube` | `\subseteq` | 子集等于 $\subseteq$ |
| `;sup` | `\supset` | 超集 $\supset$ |
| `;supe` | `\supseteq` | 超集等于 $\supseteq$ |
| `;empty` | `\varnothing` | 空集 $\varnothing$ |
| `;cup` | `\cup` | 并集 $\cup$ |
| `;cap` | `\cap` | 交集 $\cap$ |
| `;fa` | `\forall` | 任意 $\forall$ |
| `;ex` | `\exists` | 存在 $\exists$ |
| `;nex` | `\nexists` | 不存在 $\nexists$ |
| `;imp` | `\implies` | 蕴含 $\implies$ |
| `;Ra` | `\Rightarrow` | 推出 $\Rightarrow$ |
| `;La` | `\Leftarrow` | 左推出 $\Leftarrow$ |
| `;iff` | `\iff` | 当且仅当 $\iff$ |
| `;Lra` | `\Leftrightarrow` | 等价 $\Leftrightarrow$ |
| `;bc` | `\because` | 因为 $\because$ |
| `;tf` | `\therefore` | 所以 $\therefore$ |
| `;land` | `\land` | 逻辑与 $\land$ |
| `;lor` | `\lor` | 逻辑或 $\lor$ |
| `;neg` | `\neg` | 逻辑非 $\neg$ |
| `;oplus` | `\oplus` | 异或 $\oplus$ |
| `;odot` | `\odot` | 同或/按元素乘 $\odot$ |

#### 9. 箭头

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;to` | `\to` | 趋向 $\to$ |
| `;ra` | `\rightarrow` | 右箭头 $\rightarrow$ |
| `;lar` | `\leftarrow` | 左箭头 $\leftarrow$ |
| `;lra` | `\leftrightarrow` | 双向箭头 $\leftrightarrow$ |
| `;map` | `\mapsto` | 映射 $\mapsto$ |

#### 10. 字母修饰符号

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;hat` | `\hat{}` | 帽子（估计值）$\hat{y}$ |
| `;bar` | `\bar{}` | 横线（均值）$\bar{x}$ |
| `;tilde` | `\tilde{}` | 波浪号 $\tilde{y}$ |
| `;wtilde` | `\widetilde{}` | 宽波浪号 $\widetilde{abc}$ |
| `;dot` | `\dot{}` | 一阶导数点 $\dot{y}$ |
| `;ddot` | `\ddot{}` | 二阶导数点 $\ddot{y}$ |
| `;vec` | `\vec{}` | 向量 $\vec{a}$ |
| `;ol` | `\overline{}` | 上划线 $\overline{A}$ |
| `;ul` | `\underline{}` | 下划线 $\underline{x}$ |

#### 11. 字体样式与常用数集

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;bb` | `\mathbb{}` | 黑板粗体 |
| `;bf` | `\mathbf{}` | 粗体 |
| `;cal` | `\mathcal{}` | 花体 |
| `;rm` | `\mathrm{}` | 正体 |
| `;txt` | `\text{}` | 文本 |
| `;RR` | `\mathbb{R}` | 实数集 $\mathbb{R}$ |
| `;NN` | `\mathbb{N}` | 自然数集 $\mathbb{N}$ |
| `;ZZ` | `\mathbb{Z}` | 整数集 $\mathbb{Z}$ |
| `;QQ` | `\mathbb{Q}` | 有理数集 $\mathbb{Q}$ |
| `;CC` | `\mathbb{C}` | 复数集 $\mathbb{C}$ |

#### 12. 线性代数

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;otimes` | `\otimes` | 张量积/克罗内克积 $\otimes$ |
| `;perp` | `\perp` | 垂直 $\perp$ |
| `;ang` | `\langle  \rangle` | 内积尖括号 $\langle x, y \rangle$ |
| `;det` | `\det` | 行列式 $\det$ |
| `;tr` | `\text{Tr}` | 矩阵的迹 $\text{Tr}$ |

#### 13. 概率论与统计

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;EE` | `\mathbb{E}` | 期望 $\mathbb{E}$ |
| `;var` | `\mathrm{Var}` | 方差 $\mathrm{Var}$ |
| `;cov` | `\mathrm{Cov}` | 协方差 $\mathrm{Cov}$ |
| `;binom` | `\binom{}{}` | 组合数 $\binom{n}{k}$ |
| `;indep` | `\perp\!\!\!\perp` | 独立 $\perp\!\!\!\perp$ |

#### 14. 大型结构

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;pmat` | `\begin{pmatrix}...\end{pmatrix}` | 圆括号矩阵 |
| `;bmat` | `\begin{bmatrix}...\end{bmatrix}` | 方括号矩阵 |
| `;vmat` | `\begin{vmatrix}...\end{vmatrix}` | 行列式 |
| `;case` | `\begin{cases}...\end{cases}` | 分段函数 |
| `;align` | `\begin{aligned}...\end{aligned}` | 多行对齐 |

#### 15. 三角函数与常用函数

| 触发词 | 展开结果 | 触发词 | 展开结果 |
|--------|----------|--------|----------|
| `;sin` | `\sin` | `;ln` | `\ln` |
| `;cos` | `\cos` | `;exp` | `\exp` |
| `;tan` | `\tan` | `;max` | `\max` |
| `;log` | `\log` | `;min` | `\min` |

#### 16. 省略号与其他

| 触发词 | 展开结果 | 含义 |
|--------|----------|------|
| `;cd3` | `\cdots` | 居中省略号 $\cdots$ |
| `;ld3` | `\ldots` | 底部省略号 $\ldots$ |
| `;vd3` | `\vdots` | 竖直省略号 $\vdots$ |
| `;dd3` | `\ddots` | 对角省略号 $\ddots$ |
| `;und` | `\underbrace{}_{}` | 下括号标注 |
| `;ovb` | `\overbrace{}^{}` | 上括号标注 |

### 使用示例

在 Typora 中输入 `$` 进入行内公式模式后：

**示例 1：分数**

```
输入：;ff
展开：\frac{}{}
填入：\frac{1}{2}
效果：½
```

**示例 2：求和公式**

```
输入：;sum
展开：\sum_{i=1}^{n}
补充：\sum_{i=1}^{n} x_i
效果：∑ᵢ₌₁ⁿ xᵢ
```

**示例 3：偏导数**
```
输入：;pyx
展开：\frac{\partial y}{\partial x}
效果：∂y/∂x
```

**示例 4：组合使用**
```
输入：;int ;ff ;dd x
展开：\int_{}^{} \frac{}{} \mathrm{d} x
填入：\int_{0}^{1} \frac{1}{x^2+1} \mathrm{d} x
```

### 自定义扩展

编辑 `C:\Users\Alen\AppData\Roaming\espanso\match\latex-math.yml`，按以下格式添加：

```yaml
  - trigger: ";你的缩写"
    replace: "展开后的LaTeX代码"
    word: true
```

修改后执行 `espanso restart` 生效。
