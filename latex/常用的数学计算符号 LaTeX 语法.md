---
title: Latex数学符号语法
author: Alen
published: 2025-10-10
description: "常用的数学计算符号 LaTeX 语法"
first_level_category: "知识库"
second_level_category: "数学工具"
tags: ['latex']
draft: false
---


## 常用的数学计算符号 LaTeX 语法

在 Markdown 中，当编辑器支持 LaTeX 渲染时（例如 Typora, Obsidian, VS Code + Markdown Preview Enhanced），可以直接使用 LaTeX 的数学模式来输入公式。

### 1. 基本算术运算符

| 描述   | LaTeX 命令 | 效果              | 示例 Markdown         |
| ------ | ---------- | ----------------- | --------------------- |
| 加号   | +          | $ + $             | $...+$                |
| 减号   | -          | $ - $             | $...-$                |
| 乘号   | \times     | $ \times $        | $...{\times}$         |
| 除号   | / 或 \div  | $ / $ 或 $ \div $ | $.../$ 或 $...{\div}$ |
| 加减号 | \pm        | $ \pm $           | $...{\pm}$            |
| 约等于 | \approx    | $ \approx $       | $...{\approx}$        |
| 等于   | =          | $ = $             | $...=$                |
| 不等于 | \neq       | $ \neq $          | $...{\neq}$           |

### 2. 乘方与下标

| 描述          | LaTeX 命令   | 效果          | 示例 Markdown  |
| ------------- | ------------ | ------------- | -------------- |
| 上标          | ^            | $ x^2 $       | $...x^2$       |
| 上标 (多字符) | ^{...}       | $ x^{2y} $    | $...x^{2y}$    |
| 下标          | _            | $ x_2 $       | $...x_2$       |
| 下标 (多字符) | _{...}       | $ x_{2y} $    | $...x_{2y}$    |
| 上下标        | ^{...}_{...} | $ a^{n}_{i} $ | $...a^{n}_{i}$ |

### 3. 分数与根号

| 描述    | LaTeX 命令        | 效果            | 示例 Markdown      |
| ------- | ----------------- | --------------- | ------------------ |
| 分数    | \frac{分子}{分母} | $ \frac{a}{b} $ | $...{\frac{a}{b}}$ |
| 根号    | \sqrt{...}        | $ \sqrt{x} $    | $...{\sqrt{x}}$    |
| n次方根 | \sqrt[n]{...}     | $ \sqrt[3]{x} $ | $...{\sqrt[3]{x}}$ |

### 4. 积分与无穷大

| 描述         | LaTeX 命令             | 效果                  | 示例 Markdown            |
| ------------ | ---------------------- | --------------------- | ------------------------ |
| 积分         | \int                   | $ \int $              | $...{\int}$              |
| 定积分       | \int_{下限}^{上限}     | $ \int_{a}^{b} $      | $...{\int_{a}^{b}}$      |
| 双重积分     | \iint                  | $ \iint $             | $...{\iint}$             |
| 多重积分     | \int\int\int 或 \iiint | $ \iiint $            | $...{\iiint}$            |
| 无穷大       | \infty                 | $ \infty $            | $...{\infty}$            |
| 上下限的积分 | \int_{0}^{\infty}      | $ \int_{0}^{\infty} $ | $...{\int_{0}^{\infty}}$ |

### 5. 括号

| 描述                    | LaTeX 命令            | 效果                           | 示例 Markdown                       |
| ----------------------- | --------------------- | ------------------------------ | ----------------------------------- |
| 小括号                  | ()                    | `()()`                         | $...()$                             |
| 中括号                  | [] 或 \lbrack \rbrack | `[][]` 或 `[][]`               | $...[]$ 或 $...{\lbrack \rbrack}$   |
| 大圆括号 (自动调整高度) | \left( ... \right)    | $ \left( \frac{a}{b} \right) $ | $...{\left( \frac{a}{b} \right)}$   |
| 大中括号 (自动调整高度) | \left[ ... \right]    | $ \left[ \frac{a}{b} \right] $ | $...{\left[ \frac{a}{b} \right]}$   |
| 大花括号 (自动调整高度) | \left\{ ... \right\}  | $ \left\{ \frac{a}{b} \right\} $  | $...{\left\{ \frac{a}{b} \right\}}$ |
| 绝对值                  | `                     | x                              | 或\left                             |

### 6. 微分方程与导数

| 描述                   | LaTeX 命令                        | 效果                                  | 示例 Markdown                            |
| ---------------------- | --------------------------------- | ------------------------------------- | ---------------------------------------- |
| 普通导数 (y关于x)      | \frac{dy}{dx}                     | $ \frac{dy}{dx} $                     | $...{\frac{dy}{dx}}$                     |
| 高阶导数               | \frac{d^2y}{dx^2}                 | $ \frac{d^2y}{dx^2} $                 | $...{\frac{d^2y}{dx^2}}$                 |
| 偏导数                 | \frac{\partial y}{\partial x}     | $ \frac{\partial y}{\partial x} $     | $...{\frac{\partial y}{\partial x}}$     |
| 高阶偏导数             | \frac{\partial^2 y}{\partial x^2} | $ \frac{\partial^2 y}{\partial x^2} $ | $...{\frac{\partial^2 y}{\partial x^2}}$ |
| 向量（常用来表示梯度） | \nabla                            | $ \nabla $                            | $...{\nabla}$                            |
| 梯度                   | \nabla f                          | $ \nabla f $                          | $...{\nabla f}$                          |

### 7. 希腊字母 (常用小写)

| 希腊字母 | LaTeX 命令 | 效果         | 示例 Markdown   |
| -------- | ---------- | ------------ | --------------- |
| Alpha    | \alpha     | $ \alpha $   | $...{\alpha}$   |
| Beta     | \beta      | $ \beta $    | $...{\beta}$    |
| Gamma    | \gamma     | $ \gamma $   | $...{\gamma}$   |
| Delta    | \delta     | $ \delta $   | $...{\delta}$   |
| Epsilon  | \epsilon   | $ \epsilon $ | $...{\epsilon}$ |
| Zeta     | \zeta      | $ \zeta $    | $...{\zeta}$    |
| Eta      | \eta       | $ \eta $     | $...{\eta}$     |
| Theta    | \theta     | $ \theta $   | $...{\theta}$   |
| Iota     | \iota      | $ \iota $    | $...{\iota}$    |
| Kappa    | \kappa     | $ \kappa $   | $...{\kappa}$   |
| Lambda   | \lambda    | $ \lambda $  | $...{\lambda}$  |
| Mu       | \mu        | $ \mu $      | $...{\mu}$      |
| Nu       | \nu        | $ \nu $      | $...{\nu}$      |
| Xi       | \xi        | $ \xi $      | $...{\xi}$      |
| Pi       | \pi        | $ \pi $      | $...{\pi}$      |
| Rho      | \rho       | $ \rho $     | $...{\rho}$     |
| Sigma    | \sigma     | $ \sigma $   | $...{\sigma}$   |
| Tau      | \tau       | $ \tau $     | $...{\tau}$     |
| Upsilon  | \upsilon   | $ \upsilon $ | $...{\upsilon}$ |
| Phi      | \phi       | $ \phi $     | $...{\phi}$     |
| Chi      | \chi       | $ \chi $     | $...{\chi}$     |
| Psi      | \psi       | $ \psi $     | $...{\psi}$     |
| Omega    | \omega     | $ \omega $   | $...{\omega}$   |

$$
\alpha~~\beta~~\gamma~~\delta~~\epsilon~~\zeta~~\eta~~\theta~~\iota~~\kappa~~\lambda~~\mu~~\nu~~\xi~~\pi~~\rho~~\sigma~~\tau~~\upsilon~~\phi~~\chi~~\psi~~\omega
$$



**希腊字母 (常用大写)**

| 希腊字母 | LaTeX 命令 | 效果        | 示例 Markdown  |
| -------- | ---------- | ----------- | -------------- |
| Delta    | \Delta     | $ \Delta $  | $...{\Delta}$  |
| Gamma    | \Gamma     | $ \Gamma $  | $...{\Gamma}$  |
| Lambda   | \Lambda    | $ \Lambda $ | $...{\Lambda}$ |
| Sigma    | \Sigma     | $ \Sigma $  | $...{\Sigma}$  |
| Omega    | \Omega     | $ \Omega $  | $...{\Omega}$  |

$$
\Delta ~~\Gamma ~~\Lambda ~~\Sigma ~~\Omega
$$



### 8. 集合与逻辑符号

| 描述     | LaTeX 命令               | 效果                             | 示例 Markdown                          |
| -------- | ------------------------ | -------------------------------- | -------------------------------------- |
| 属于     | \in                      | $ \in $                          | $...{\in}$                             |
| 不属于   | \notin                   | $ \notin $                       | $...{\notin}$                          |
| 子集     | \subset 或 \subseteq     | $ \subset $ 或 $ \subseteq $     | $...{\subset}$ 或 $...{\subseteq}$     |
| 真子集   | \subset                  | $ \subset $                      | $...{\subset}$                         |
| 超集     | \supset 或 \supseteq     | $ \supset $ 或 $ \supseteq $     | $...{\supset}$ 或 $...{\supseteq}$     |
| 非空集合 | \emptyset 或 \varnothing | $ \emptyset $ 或 $ \varnothing $ | $...{\emptyset}$ 或 $...{\varnothing}$ |
| 并集     | \cup                     | $ \cup $                         | $...{\cup}$                            |
| 交集     | \cap                     | $ \cap $                         | $...{\cap}$                            |
| 属于     | \forall                  | $ \forall $                      | $...{\forall}$                         |
| 存在     | \exists                  | $ \exists $                      | $...{\exists}$                         |
| 蕴含     | \implies 或 \Rightarrow  | $ \implies $ 或 $ \Rightarrow $  | $...{\implies}$ 或 $...{\Rightarrow}$  |
| 当且仅当 | \iff 或 \Leftrightarrow  | $ \iff $ 或 $ \Leftrightarrow $  | $...{\iff}$ 或 $...{\Leftrightarrow}$  |



### 9.积分与无穷大 LaTeX 语法

| 描述                | LaTeX 命令                             | 效果                             | 示例 Markdown                       |
| ------------------- | -------------------------------------- | -------------------------------- | ----------------------------------- |
| 积分符号            | \int                                   | $ \int $                         | $...{\int}$                         |
| 上下标的积分        | \int_{下限}^{上限}                     | $ \int_{a}^{b} $                 | $...{\int_{a}^{b}}$                 |
| 反常积分            | \int_{0}^{\infty}                      | $ \int_{0}^{\infty} $            | $...{\int_{0}^{\infty}}$            |
| 无穷大符号          | \infty                                 | $ \infty $                       | $...{\infty}$                       |
| 双重积分            | \iint                                  | $ \iint $                        | $...{\iint}$                        |
| 三重积分            | \iiint                                 | $ \iiint $                       | $...{\iiint}$                       |
| n重积分             | \idotsint (不常用，通常用 \int...\int) | $ \idotsint $                    | $...{\idotsint}$                    |
| 线积分 (标量场)     | \oint                                  | $ \oint $                        | $...{\oint}$                        |
| 线积分 (向量场)     | \int \vec{F} \cdot d\vec{r}            | $ \int \vec{F} \cdot d\vec{r} $  | $...{\int \vec{F} \cdot d\vec{r}}$  |
| 面积分 (标量场)     | \iint                                  | $ \iint $                        | $...{\iint}$                        |
| 面积分 (向量场)     | \iint \vec{F} \cdot d\vec{S}           | $ \iint \vec{F} \cdot d\vec{S} $ | $...{\iint \vec{F} \cdot d\vec{S}}$ |
| 体积分              | \iiint                                 | $ \iiint $                       | $...{\iiint}$                       |
| 微分算子            | \, (小空格) + d (或 dx, dy 等)         | $ f(x) dx $                      | $...{f(x) \, dx}$                   |
| 连乘 (求积)         | \prod                                  | $ \prod $                        | $...{\prod}$                        |
| 上下标的连乘        | \prod_{i=1}^{n}                        | $ \prod_{i=1}^{n} $              | $...{\prod_{i=1}^{n}}$              |
| 求和                | \sum                                   | $ \sum $                         | $...{\sum}$                         |
| 上下标的求和        | \sum_{i=1}^{n}                         | $ \sum_{i=1}^{n} $               | $...{\sum_{i=1}^{n}}$               |
| 极限                | \lim                                   | $ \lim $                         | $...{\lim}$                         |
| 下标的极限          | \lim_{x \to \infty}                    | $ \lim_{x \to \infty} $          | $...{\lim_{x \to \infty}}$          |
| 下标的极限 (左极限) | \lim_{x \to a^-}                       | $ \lim_{x \to a^-} $             | $...{\lim_{x \to a^-}}$             |
| 下标的极限 (右极限) | \lim_{x \to a^+}                       | $ \lim_{x \to a^+} $             | $...{\lim_{x \to a^+}}$             |

### 10.字母扩展符号（如预测值）

|      描述      |    Latex指令    |     效果示例      |
| :------------: | :-------------: | :---------------: |
|     波浪号     |    \tilde{y}    |    $\tilde{y}$    |
| 估计值（帽子） |     \hat{y}     |     $\hat{y}$     |
|      横线      |     \bar{y}     |     $\bar{y}$     |
|       点       |     \dot{y}     |     $\dot{y}$     |
|    宽波浪号    | \widetilde{abc} | $\widetilde{abc}$ |

### 11.线性代数

|      描述      |    Latex指令    |     效果示例      |
| :------------: | :-------------: | :---------------: |
|     矩阵的按元素乘法     |    \odot{y}    | $\odot$ |
|  |          |         |
|            |          |       |
|             |          |      |
|        |  | |