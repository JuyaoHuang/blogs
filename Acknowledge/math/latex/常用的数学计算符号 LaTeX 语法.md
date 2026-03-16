---
title: Latex数学符号语法
publishDate: 2025-10-10
description: "常用的数学计算符号 LaTeX 语法"
tags: ['latex']
language: 'Chinese'
first_level_category: "知识库"
second_level_category: "数学工具"
draft: false
---

## 常用的数学计算符号 LaTeX 语法

在 Markdown 中，当编辑器支持 LaTeX 渲染时（例如 Typora, Obsidian, VS Code + Markdown Preview Enhanced），可以直接使用 LaTeX 的数学模式来输入公式。

### 1. 基本算术与关系运算符

| 描述     | LaTeX 命令 | 效果              | 示例 Markdown |
| -------- | ---------- | ----------------- | --------------------- |
| 加号     | +          | $ + $             | $...+$                |
| 减号     | -          | $ - $             | $...-$                |
| 乘号     | \times     | $ \times $        | $...{\times}$         |
| 点乘     | \cdot      | $ \cdot $         | $...{\cdot}$          |
| 除号     | / 或 \div  | $ / $ 或 $ \div $ | $.../$ 或 $...{\div}$ |
| 加减号   | \pm        | $ \pm $           | $...{\pm}$            |
| 减加号   | \mp        | $ \mp $           | $...{\mp}$            |
| 约等于   | \approx    | $ \approx $       | $...{\approx}$        |
| 等于     | =          | $ = $             | $...=$                |
| 不等于   | \neq       | $ \neq $          | $...{\neq}$           |
| 恒等于   | \equiv     | $ \equiv $        | $...{\equiv}$         |
| 小于     | <          | $ < $             | $...<$                |
| 大于     | >          | $ > $             | $...>$                |
| 小于等于 | \leq 或 \le | $ \leq $          | $...{\leq}$           |
| 大于等于 | \geq 或 \ge | $ \geq $          | $...{\geq}$           |
| 远小于   | \ll        | $ \ll $           | $...{\ll}$            |
| 远大于   | \gg        | $ \gg $           | $...{\gg}$            |
| 相似/渐近| \sim       | $ \sim $          | $...{\sim}$           |
| 正比于   | \propto    | $ \propto $       | $...{\propto}$        |

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
| n 次方根 | \sqrt[n]{...}     | $ \sqrt[3]{x} $ | $...{\sqrt[3]{x}}$ |

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
| 普通导数 (y 关于 x)      | \frac{dy}{dx}                     | $ \frac{dy}{dx} $                     | $...{\frac{dy}{dx}}$                     |
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
| 真子集   | \subsetneq               | $ \subsetneq $                   | $...{\subsetneq}$                      |
| 超集     | \supset 或 \supseteq     | $ \supset $ 或 $ \supseteq $     | $...{\supset}$ 或 $...{\supseteq}$     |
| 非空集合 | \emptyset 或 \varnothing | $ \emptyset $ 或 $ \varnothing $ | $...{\emptyset}$ 或 $...{\varnothing}$ |
| 并集     | \cup                     | $ \cup $                         | $...{\cup}$                            |
| 交集     | \cap                     | $ \cap $                         | $...{\cap}$                            |
| 任意/所有| \forall                  | $ \forall $                      | $...{\forall}$                         |
| 存在     | \exists                  | $ \exists $                      | $...{\exists}$                         |
| 不存在   | \nexists                 | $ \nexists $                     | $...{\nexists}$                        |
| 蕴含(推导) | \implies 或 \Rightarrow  | $ \implies $ 或 $ \Rightarrow $  | $...{\implies}$ 或 $...{\Rightarrow}$  |
| 推出(左) | \Leftarrow               | $ \Leftarrow $                   | $...{\Leftarrow}$                      |
| 当且仅当 | \iff 或 \Leftrightarrow  | $ \iff $ 或 $ \Leftrightarrow $  | $...{\iff}$ 或 $...{\Leftrightarrow}$  |
| 因为     | \because                 | $ \because $                     | $...{\because}$                        |
| 所以     | \therefore               | $ \therefore $                   | $...{\therefore}$                      |
| 逻辑与   | \land 或 \wedge          | $ \land $ 或 $ \wedge $          | $...{\land}$ 或 $...{\wedge}$          |
| 逻辑或   | \lor 或 \vee             | $ \lor $ 或 $ \vee $             | $...{\lor}$ 或 $...{\vee}$             |
| 逻辑非   | \lnot 或 \neg            | $ \lnot $ 或 $ \neg $            | $...{\lnot}$ 或 $...{\neg}$            |
| 异或     | \oplus                   | $ \oplus $                       | $...{\oplus}$                          |
| 同或     | \odot                    | $ \odot $                        | $...{\odot}$                           |
| 逻辑非 (上横线) | \overline{A}      | $ \overline{A} $                 | $...{\overline{A}}$                    |
| 逻辑与 (集合表示) | \cap            | $ \cap $                         | $...{\cap}$                            |
| 逻辑或 (集合表示) | \cup            | $ \cup $                         | $...{\cup}$                            |



### 9.积分与无穷大 LaTeX 语法

| 描述                | LaTeX 命令                             | 效果                             | 示例 Markdown                       |
| ------------------- | -------------------------------------- | -------------------------------- | ----------------------------------- |
| 积分符号            | \int                                   | $ \int $                         | $...{\int}$                         |
| 上下标的积分        | \int_{下限}^{上限}                     | $ \int_{a}^{b} $                 | $...{\int_{a}^{b}}$                 |
| 反常积分            | \int_{0}^{\infty}                      | $ \int_{0}^{\infty} $            | $...{\int_{0}^{\infty}}$            |
| 无穷大符号          | \infty                                 | $ \infty $                       | $...{\infty}$                       |
| 双重积分            | \iint                                  | $ \iint $                        | $...{\iint}$                        |
| 三重积分            | \iiint                                 | $ \iiint $                       | $...{\iiint}$                       |
| n 重积分             | \idotsint (不常用，通常用 \int...\int) | $ \idotsint $                    | $...{\idotsint}$                    |
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

|      描述      |    Latex 指令    |     效果示例      |
| :------------: | :-------------: | :---------------: |
|     波浪号     |    \tilde{y}    |    $\tilde{y}$    |
| 估计值（帽子） |     \hat{y}     |     $\hat{y}$     |
|      横线      |     \bar{y}     |     $\bar{y}$     |
|       点       |     \dot{y}     |     $\dot{y}$     |
|    宽波浪号    | \widetilde{abc} | $\widetilde{abc}$ |

### 11.线性代数与其他

|      描述      |    Latex 指令    |     效果示例      | 示例 Markdown |
| :------------: | :-------------: | :---------------: | :-------------: |
| 矩阵按元素乘法 |    \odot        |     $\odot$       | $...{\odot}$    |
| 张量积/克罗内克积 |   \otimes       |     $\otimes$     | $...{\otimes}$  |
| 正交/垂直         |    \perp        |     $\perp$       | $...{\perp}$    |
| 范数双竖线        |   \|x\|           |    $\|x\|$      | $...{\|x\|}$   |
| 内积大括号        |  \langle x, y \rangle | $\langle x, y \rangle$ | $...{\langle x, y \rangle}$ |
| 行列式            | \det(A)         |    $\det(A)$      | $...{\det(A)}$  |
| 迹                | \text{Tr}(A)    |    $\text{Tr}(A)$ | $...{\text{Tr}(A)}$ |

### 12. 概率论与数理统计

| 描述               | LaTeX 命令                          | 效果                          | 示例 Markdown                     |
| ------------------ | ----------------------------------- | ----------------------------- | --------------------------------- |
| 事件概率           | P(A)                                | $ P(A) $                      | $...{P(A)}$                       |
| 条件概率           | P(A\|B)                             | $ P(A\|B) $                   | $...{P(A\|B)}$                    |
| 期望               | E(X) 或 \mathbb{E}(X)               | $ E(X) $ 或 $ \mathbb{E}(X) $ | $...{E(X)}$ 或 $...{\mathbb{E}(X)}$|
| 方差               | D(X) 或 \mathrm{Var}(X)             | $ D(X) $ 或 $ \mathrm{Var}(X) $| $...{D(X)}$ 或 $...{\mathrm{Var}(X)}$ |
| 协方差             | \mathrm{Cov}(X, Y)                  | $ \mathrm{Cov}(X, Y) $        | $...{\mathrm{Cov}(X, Y)}$         |
| 分布：正态分布     | X \sim N(\mu, \sigma^2)             | $ X \sim N(\mu, \sigma^2) $   | $...{X \sim N(\mu, \sigma^2)}$    |
| 分布：二项分布     | X \sim B(n, p)                      | $ X \sim B(n, p) $            | $...{X \sim B(n, p)}$             |
| 分布：泊松分布     | X \sim P(\lambda)                   | $ X \sim P(\lambda) $         | $...{X \sim P(\lambda)}$          |
| 分布：均匀分布     | X \sim U(a, b)                      | $ X \sim U(a, b) $            | $...{X \sim U(a, b)}$             |
| 独立 (常用于事件)  | \perp\!\!\!\perp                    | $ \perp\!\!\!\perp $          | $...{\perp\!\!\!\perp}$           |
| 排列               | A_n^k 或 P_n^k                      | $ A_n^k $                     | $...{A_n^k}$                      |
| 组合               | C_n^k 或 \binom{n}{k}               | $ C_n^k $ 或 $ \binom{n}{k} $ | $...{C_n^k}$ 或 $...{\binom{n}{k}}$ |

### 13. 高等数学补充符号集

| 描述               | LaTeX 命令                          | 效果                              | 示例 Markdown                     |
| ------------------ | ----------------------------------- | --------------------------------- | --------------------------------- |
| 偏导数 (符号)      | \partial                            | $ \partial $                      | $...{\partial}$                   |
| 二阶偏导数混合     | \frac{\partial^2 z}{\partial x \partial y} | $ \frac{\partial^2 z}{\partial x \partial y} $ | $...{\frac{\partial^2 z}{\partial x \partial y}}$ |
| 全微分             | \mathrm{d} y = f'(x)\mathrm{d} x    | $ \mathrm{d} y = f'(x)\mathrm{d} x $| $...{\mathrm{d} z = ...}$         |
| 多元函数极限       | \lim_{(x,y) \to (0,0)}              | $ \lim_{(x,y) \to (0,0)} $        | $...{\lim_{(x,y) \to (0,0)}}$     |
| 等价无穷小         | \sim                                | $ \sin x \sim x $                 | $...{\sin x \sim x}$              |
| 泰勒/麦克劳林展开  | \sum_{n=0}^{\infty} \frac{x^n}{n!}  | $ \sum_{n=0}^{\infty} \frac{x^n}{n!} $| $...{\sum_{n=0}^{\infty} ...}$    |
| 自然对数底数 $e$   | \mathrm{e} 或 e                     | $ \mathrm{e}^x $                  | $...{\mathrm{e}^x}$               |
| 复合函数求导法     | \frac{\mathrm{d} y}{\mathrm{d} x} = \frac{\mathrm{d} y}{\mathrm{d} u} \cdot \frac{\mathrm{d} u}{\mathrm{d} x} | $ \frac{\mathrm{d} y}{\mathrm{d} x} = \frac{\mathrm{d} y}{\mathrm{d} u} \frac{\mathrm{d} u}{\mathrm{d} x} $ | $...{\frac{\mathrm{d} y}{\mathrm{d} x}}$ |

### 14. 常用大型结构（分段函数与矩阵）

#### 分段函数 (cases 环境)

使用 `\begin{cases}` 和 `\end{cases}`，用 `&` 分隔表达式和条件，`\\` 表示换行。
示例代码：
```latex
f(x) =
\begin{cases}
1, & x > 0 \\
0, & x = 0 \\
-1, & x < 0
\end{cases}
```
**效果：**
$$
f(x) =
\begin{cases}
1, & x > 0 \\
0, & x = 0 \\
-1, & x < 0
\end{cases}
$$

#### 各种矩阵环境

Markdown 常用的矩阵格式环境不需要 `\left[` 来包围，直接使用特定标签。如 `pmatrix` (圆括号)、`bmatrix` (方括号)、`vmatrix` (单竖线行列式)、`Vmatrix` (双竖线)。

示例代码 (方括号矩阵 `bmatrix`)：
```latex
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
```
**效果：**
$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

示例代码 (行列式单竖线 `vmatrix`)：
```latex
|A| = \begin{vmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{vmatrix} = a_{11}a_{22} - a_{12}a_{21}
```
**效果：**
$$
|A| = \begin{vmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{vmatrix} = a_{11}a_{22} - a_{12}a_{21}
$$

#### 多行公式对齐 (aligned 环境)

在推导过程（如计算不定积分或矩阵变换）中，使用 `aligned` 进行对齐，默认对齐符为 `&`：

示例代码：
```latex
\begin{aligned}
\int_0^1 x^2 \, dx &= \left[ \frac{1}{3} x^3 \right]_0^1 \\
&= \frac{1}{3} - 0 \\
&= \frac{1}{3}
\end{aligned}
```
**效果：**
$$
\begin{aligned}
\int_0^1 x^2 \, dx &= \left[ \frac{1}{3} x^3 \right]_0^1 \\
&= \frac{1}{3} - 0 \\
&= \frac{1}{3}
\end{aligned}
$$