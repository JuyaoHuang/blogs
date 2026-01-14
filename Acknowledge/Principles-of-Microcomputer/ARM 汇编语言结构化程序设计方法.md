---
title: 汇编语言
publishDate: 2025-10-11
description: "微机原理第四章-汇编语言程序设计"
tags: ['微机原理']
language: 'Chinese'
first_level_category: "知识库"
second_level_category: "微机原理"
draft: false
---

# ARM 汇编语言结构化程序设计方法

结构化程序设计是一种编程范式，旨在通过使用定义良好的控制结构（顺序、选择、循环）和子程序来提高程序的清晰度、质量和开发时间。在汇编语言这种底层语言中，遵循结构化设计的原则尤为重要，它可以帮助我们编写出更易于理解、调试和维护的代码。

## 4.3.1 顺序结构设计

顺序结构是最基本的程序执行流程，即指令按照它们在代码中出现的先后顺序逐条执行。在 ARM 汇编中，除非遇到跳转指令，否则处理器会自动按顺序执行。

**设计要点：**
*   指令逐条排列。
*   依赖于处理器 PC (Program Counter) 寄存器的自动递增。

**示例：**
假设我们要计算 `R2 = R0 + R1`，然后将结果存储到某个内存地址。

```arm
; 假设 R0 和 R1 已有初值
; 假设 R3 指向目标内存地址

ADD R2, R0, R1   ; R2 = R0 + R1 (顺序执行1)
STR R2, [R3]     ; 将 R2 的内容存储到 R3 指向的内存地址 (顺序执行2)
; ... 后续指令 (顺序执行3)
    
```

在这个例子中，ADD 指令执行完毕后，STR 指令会接着执行。

## 4.3.2 选择结构设计

选择结构允许程序根据特定条件执行不同的代码路径。这通常对应于高级语言中的 if-then-else 或 switch-case 语句。

**设计要点：**

1. **条件判断：** 使用比较指令（如 CMP, CMN, TST, TEQ）设置 CPSR (Current Program Status Register) 中的条件标志位 (N, Z, C, V)。
2. **条件跳转：** 使用条件分支指令（如 BEQ, BNE, BGT, BLT, BHI, BLS 等）根据条件标志位的状态来改变程序的执行流程。
3. **无条件跳转：** 使用 B (Branch) 指令跳转到代码的其他部分或跳过某些代码块。

**示例 1: if-then 结构**
如果 R0 等于 R1，则将 R2 设置为 1。

```
CMP R0, R1      ; 比较 R0 和 R1，设置条件标志位
BNE skip_then   ; 如果 R0 != R1 (Not Equal)，则跳转到 skip_then
; then 代码块
MOV R2, #1      ; 如果 R0 == R1，执行此句
skip_then:
; ... 后续代码  
```

**示例 2: if-then-else 结构**
如果 R0 大于 R1，则 R2 = R0；否则 R2 = R1。

```
    CMP R0, R1      ; 比较 R0 和 R1
    BLE else_block  ; 如果 R0 <= R1 (Branch if Less than or Equal)，跳转到 else_block
    ; then 代码块 (R0 > R1)
    MOV R2, R0
    B end_if        ; 跳转到选择结构结束
else_block:
    ; else 代码块 (R0 <= R1)
    MOV R2, R1
end_if:
    ; ... 后续代码
    
```

**注意：** 在 Thumb-2 指令集 (Cortex-M 系列常用) 中，还可以使用 IT (If-Then) 或 ITE (If-Then-Else) 指令来构造更紧凑的条件执行块，但其使用有一定限制。上述示例为更通用的 ARM 汇编实现。

## 4.3.3 循环结构设计

循环结构允许程序重复执行一段代码，直到满足某个退出条件。这对应于高级语言中的 for, while, do-while 循环。

**设计要点：**

1. **循环初始化：** 设置循环计数器或循环条件的初始状态。
2. **循环体：** 需要重复执行的指令序列。
3. **循环条件判断：** 在循环的开始或结束处检查是否继续循环。通常使用 CMP 和条件分支指令。
4. **循环变量更新：** 修改循环计数器或影响循环条件的变量。
5. **循环跳转：** 使用分支指令跳回循环体的开始。

**示例 1: for 循环 (计数循环)**
将内存中从 src_addr 开始的 10 个字 (word) 累加到 R0 (初始为0)。假设 R1 指向 src_addr。

```
    MOV R0, #0      ; R0 = 累加和，初始化为 0
    MOV R2, #10     ; R2 = 循环计数器
    ; R1 假设已加载源地址

loop_start:
    LDR R3, [R1], #4 ; 从 R1 指向的地址加载一个字到 R3，然后 R1 = R1 + 4
    ADD R0, R0, R3   ; R0 = R0 + R3
    SUBS R2, R2, #1  ; R2 = R2 - 1，并设置条件标志位 (S后缀)
    BNE loop_start   ; 如果 R2 != 0 (Not Equal)，则跳转回 loop_start
loop_end:
    ; ... R0 中是累加结果
    
```

**示例 2: while 循环 (条件循环)**
当 R1 指向的内存单元中的值不为 0 时，将该值累加到 R0，并将 R1 指向下一个字。

```
    MOV R0, #0      ; R0 = 累加和，初始化为 0
    ; R1 假设已加载源地址

while_check:
    LDR R2, [R1]    ; 加载 R1 指向的值到 R2
    CMP R2, #0      ; 比较 R2 和 0
    BEQ while_end   ; 如果 R2 == 0 (Equal)，则结束循环

    ; 循环体
    ADD R0, R0, R2  ; 累加
    ADD R1, R1, #4  ; R1 指向下一个字
    B while_check   ; 返回循环条件检查

while_end:
    ; ... R0 中是累加结果
    
```

## 4.3.4 子程序结构设计

子程序（也称为函数、过程或方法）是将一段具有特定功能的代码封装起来，以便在程序的不同地方重复调用。这有助于代码模块化、重用和简化。

**设计要点 (遵循 AAPCS - ARM Architecture Procedure Call Standard)：**

1. **调用：** 使用 BL (Branch with Link) 指令调用子程序。BL 指令会将下一条指令的地址（返回地址）保存到 LR (Link Register, R14)，然后跳转到子程序的目标地址。
2. **参数传递：**
   - 通常前 4 个参数通过寄存器 R0-R3 传递。
   - 更多的参数通过栈传递。
3. **返回值：**
   - 通常通过 R0 (有时也用 R1) 返回。
4. **寄存器保存与恢复：**
   - **调用者保存寄存器 (Caller-saved)：** R0-R3, R12。如果调用者在调用子程序后仍需要这些寄存器的值，调用者负责在调用前保存它们。
   - **被调用者保存寄存器 (Callee-saved)：** R4-R11, LR。如果子程序需要修改这些寄存器，它必须在修改前将它们保存到栈上，并在返回前恢复它们。LR 尤其重要，因为它存储了返回地址。
5. **返回：**
   - 通常使用 BX LR (Branch and Exchange) 指令返回。它会跳转到 LR 中的地址，并根据 LR 最低位的状态切换处理器状态（ARM/Thumb，在现代 Cortex-M 中通常不需要考虑状态切换，直接使用 BX LR 或 POP {PC}）。
   - 如果 LR 已被压栈，可以通过 POP {PC} (将栈顶值弹出到 PC) 的方式返回，同时可以恢复其他被调用者保存的寄存器。
6. **栈操作：** 使用 PUSH 和 POP 指令来保存和恢复寄存器，以及传递栈参数和分配局部变量。栈指针通常是 SP (Stack Pointer, R13)。

**示例：一个简单的加法子程序**
子程序 add_numbers 将 R0 和 R1 中的数相加，结果存回 R0。

```
    ; 主程序调用部分
    MOV R0, #5          ; 第一个参数
    MOV R1, #10         ; 第二个参数
    BL add_numbers      ; 调用子程序
    ; 此时 R0 中应为 15
    ; ... 后续代码

; 子程序定义
add_numbers:
    PUSH {LR}           ; 保存返回地址 (LR 是被调用者保存的，但这里子程序简单，
                        ; 且没有再次调用其他子程序，也可以不 PUSH LR，直接 BX LR。
                        ; 但 PUSH {LR} 是好习惯，尤其是如果子程序内部还会调用其他子程序)
                        ; 如果子程序使用了 R4-R11，也应在此 PUSH
    
    ADD R0, R0, R1      ; 执行加法，结果在 R0
    
    POP {PC}            ; 从栈中弹出 LR 的值到 PC，实现返回
                        ; 同时恢复其他 PUSH 的寄存器（如果 POP {..., PC}）
    
```

如果子程序内部修改了 R4-R11，则：

```
    my_complex_subroutine:
    PUSH {R4-R7, LR}  ; 保存被调用者保存的寄存器和返回地址
    ; ...
    ; 使用 R0-R3 (参数), R4-R7 (临时)
    ; ...
    POP {R4-R7, PC}   ; 恢复寄存器并返回
```

## 4.3.5 ARM可执行映像文件的构成及各个段在存储器中的位置

当 ARM 汇编（或 C/C++）程序被编译和链接后，会生成一个可执行映像文件（如 ELF, AXF, BIN 格式）。这个文件包含了程序的代码和数据，并定义了它们在目标系统内存中的布局。

**主要段 (Sections)：**

1. **.text (或 CODE, RO-CODE) 段：**
   - 包含程序的**可执行指令**。
   - 通常是**只读 (Read-Only)** 的。
   - 在嵌入式系统中，这部分通常存放在 **Flash 或 ROM** 中。
   - 也包含只读数据，如常量字符串 (有时会放在 .rodata 段)。
2. **.rodata (或 CONST, RO-DATA) 段：**
   - 包含**只读数据 (Read-Only Data)**，例如程序中定义的常量。
   - 通常存放在 **Flash 或 ROM** 中。
3. **.data (或 RW-DATA) 段：**
   - 包含程序中**已初始化的全局变量和静态变量**。
   - 这部分数据是**可读写 (Read-Write)** 的。
   - 在嵌入式系统中：
     - 初始值存储在 **Flash/ROM** 中（加载区 Load Region）。
     - 在程序启动时，启动代码 (startup code) 会将这部分数据从 Flash/ROM **复制到 RAM** 中（执行区 Execution Region），因为程序运行时需要修改它们。
4. **.bss (Block Started by Symbol) 段 (或 ZI-DATA)：**
   - 包含程序中**未初始化或初始化为零的全局变量和静态变量**。
   - 这部分数据是**可读写 (Read-Write)** 的。
   - 在映像文件中，.bss 段**不占用实际空间**（只记录大小）。
   - 在程序启动时，启动代码会在 RAM 中为 .bss 段分配空间，并将其**清零**。
5. **堆 (Heap)：**
   - 用于**动态内存分配**（例如，C 中的 malloc()，C++ 中的 new）。
   - 位于 RAM 中，通常从 .bss 段的末尾向上增长（或由链接器指定）。
   - 大小可以在链接器脚本中配置。
6. **栈 (Stack)：**
   - 用于存储**函数调用的返回地址、传递参数、局部变量以及保存寄存器**。
   - 位于 RAM 中，通常从 RAM 的高端向下增长。
   - 大小可以在链接器脚本或启动代码中配置。
   - 每个任务或线程通常有自己的栈。ARM 处理器有主栈指针 (MSP) 和进程栈指针 (PSP)，用于不同模式下的栈操作。

**链接器脚本 (Linker Script / Scatter File)：**
链接器脚本（如 GCC 的 .ld 文件，ARM/Keil 的 .sct 文件）定义了这些段如何组织以及它们在内存中的具体地址。开发者可以修改链接器脚本来定制内存布局，以适应特定的硬件配置。

**启动代码 (Startup Code)：**
在 main() 函数执行之前，会运行一段启动代码 (通常用汇编编写)。其主要任务包括：

- 初始化硬件（如时钟、内存控制器）。
- 将 .data 段从 Flash/ROM 复制到 RAM。
- 将 .bss 段清零。
- 初始化栈指针。
- 调用 C/C++ 库的初始化函数 (如果需要)。
- 最后跳转到 main() 函数。

理解映像文件的构成和内存布局对于调试、优化内存使用以及处理底层硬件交互至关重要。

## 4.3.6 调用其他源文件中的符号

在大型项目中，通常会将代码分散到多个源文件中以提高模块化和可维护性。汇编语言也支持这种做法。要在一个源文件中调用另一个源文件中定义的符号（如子程序标签或变量地址），需要使用特定的汇编伪指令。

**主要汇编伪指令：**

1. **GLOBAL (或 EXPORT)：**
   - 用于将当前源文件中的一个符号（通常是标签）声明为全局可见的。
   - 这样，其他源文件就可以引用这个符号。
   - **用法：** GLOBAL symbol_name 或 EXPORT symbol_name
2. **EXTERN (或 IMPORT)：**
   - 用于声明当前源文件将引用一个在其他源文件中定义的全局符号。
   - 告诉汇编器这个符号是外部定义的，链接器会在链接阶段解析这个引用。
   - **用法：** EXTERN symbol_name 或 IMPORT symbol_name

**工作流程：**

1. **定义方 (Definition File)：** 在包含符号定义的源文件中，使用 GLOBAL (或 EXPORT) 伪指令将该符号导出。

   ```
   ; file1.s
   AREA MyCode, CODE, READONLY
   
   GLOBAL myFunction      ; 导出 myFunction 标签
   myFunction
   ; ... 子程序代码 ...
   BX LR                 ; 返回
   myFunction
   ; ... 子程序代码 ...
   BX LR                 ; 返回
   ```

1. **引用方 (Reference File)：** 在需要调用该符号的源文件中，使用 EXTERN (或 IMPORT) 伪指令声明该符号是外部的。

   ```
   ; file2.s
   AREA MyOtherCode, CODE, READONLY
   
   EXTERN myFunction      ; 声明 myFunction 是外部定义的
   D
   start_here
   ; ...
   BL myFunction         ; 调用在 file1.s 中定义的 myFunction
   ; ...
     END
   ```

1. **链接 (Linking)：**
   - 汇编器分别汇编 file1.s 和 file2.s 生成目标文件 (.o 文件)。
   - 链接器将这些目标文件以及可能需要的库文件链接在一起，生成最终的可执行映像文件。
   - 在链接过程中，链接器会解析 file2.o 中对 myFunction 的引用，将其指向 file1.o 中 myFunction 的实际地址。

**符号类型：**

- **代码符号：** 通常是子程序的入口标签。
- **数据符号：** 可以是变量的标签，表示变量的地址。

这种机制是实现模块化编程和代码库（如标准库、驱动程序库）的基础。



## ARM 汇编中的条件跳转与循环退出条件详解

在 ARM 汇编中，无论是实现选择结构 (if-then-else) 还是循环结构 (for, while, do-while)，其核心都依赖于以下两个步骤：

1.  **设置条件码 (Condition Codes)：** 通过执行特定指令（主要是比较指令或带有 `S` 后缀的数据处理指令）来更新 ARM 处理器状态寄存器 (CPSR) 中的条件码标志位。
2.  **条件执行或条件跳转：** 根据 CPSR 中条件码标志位的状态，决定是否执行后续的一条或多条指令（如 Thumb-2 的 `IT` 指令），或者执行一个条件分支指令（如 `B<cond>`）来改变程序的执行流程。

### 一、选择结构中的条件跳转指令

选择结构的核心是根据一个或多个条件的真假来执行不同的代码块。

#### 1. CPSR 中的条件码标志位

CPSR (Current Program Status Register) 中有四个关键的条件码标志位，它们在执行比较指令或带 `S` 后缀的数据处理指令后会被更新：

*   **N (Negative flag):** 如果结果为负数（即结果的最高位为1），则 N=1；否则 N=0。
*   **Z (Zero flag):** 如果结果为零，则 Z=1；否则 Z=0。
*   **C (Carry flag):**
    *   对于加法运算 (包括 `CMN`)：如果运算产生无符号进位，则 C=1；否则 C=0。
    *   对于减法运算 (包括 `CMP`)：如果运算产生无符号借位，则 C=0；否则 C=1 (可以理解为“无借位”时C=1)。
    *   对于移位操作：C位保存最后移出的位。
*   **V (Overflow flag):** 如果有符号运算的结果超出了目标寄存器能表示的范围（发生溢出），则 V=1；否则 V=0。

#### 2. 设置条件码的指令

主要有以下几类指令可以设置条件码：

*   **比较指令 (不改变操作数，只改变标志位)：**
    *   `CMP Rn, Operand2`: 计算 `Rn - Operand2`，根据结果更新 N, Z, C, V 标志位。
        *   若 `Rn == Operand2`，则 Z=1。
        *   若 `Rn < Operand2` (无符号)，则 C=0。
        *   若 `Rn >= Operand2` (无符号)，则 C=1。
        *   若 `Rn < Operand2` (有符号)，则 N!=V。
        *   若 `Rn >= Operand2` (有符号)，则 N==V。
    *   `CMN Rn, Operand2`: 计算 `Rn + Operand2`，根据结果更新 N, Z, C, V 标志位。主要用于比较一个数与一个负数。
    *   `TST Rn, Operand2`: 计算 `Rn AND Operand2`，根据结果更新 N, Z 标志位，C 标志位不变（或根据移位结果设定，取决于指令），V 标志位清零。常用于测试一个数中的特定位是否为1。
    *   `TEQ Rn, Operand2`: 计算 `Rn EOR Operand2`，根据结果更新 N, Z 标志位，C 标志位不变（或根据移位结果设定），V 标志位清零。常用于测试两个数是否相等，或特定位是否相同。

*   **带 `S` 后缀的数据处理指令：**
    几乎所有的算术逻辑运算指令（如 `ADD`, `SUB`, `MOV`, `AND`, `ORR`, `EOR`, `LSL`, `LSR` 等）都可以加上 `S` 后缀 (如 `ADDS`, `SUBS`, `MOVS`)，使得它们在执行操作的同时也会更新 N, Z, C, V 标志位。
    例如：`SUBS R0, R0, #1`  不仅执行 R0 = R0 - 1，还会根据结果设置条件标志位。

#### 3. 条件分支指令 `B<cond>`

一旦条件码被设置，就可以使用条件分支指令来改变执行流程。其基本格式为 `B<cond> label`，其中 `<cond>` 是一个条件码助记符。

**常用的条件码助记符及其含义：**

| 助记符         | 含义 (英文)                | 标志位状态   | 描述 (中文)                    |
| :------------- | :------------------------- | :----------- | :----------------------------- |
| `EQ`           | Equal                      | Z=1          | 相等                           |
| `NE`           | Not Equal                  | Z=0          | 不相等                         |
| **有符号比较** |                            |              |                                |
| `GT`           | Greater Than               | Z=0 AND N=V  | 大于 (有符号)                  |
| `LT`           | Less Than                  | N!=V         | 小于 (有符号)                  |
| `GE`           | Greater or Equal           | N=V          | 大于等于 (有符号)              |
| `LE`           | Less or Equal              | Z=1 OR N!=V  | 小于等于 (有符号)              |
| **无符号比较** |                            |              |                                |
| `HI`           | Higher                     | C=1 AND Z=0  | 高于 (无符号)                  |
| `LS`           | Lower or Same              | C=0 OR Z=1   | 低于或相同 (无符号)            |
| `HS`/`CS`      | Higher or Same / Carry Set | C=1          | 高于或相同 / 进位设置 (无符号) |
| `LO`/`CC`      | Lower / Carry Clear        | C=0          | 低于 / 进位清除 (无符号)       |
| **其他**       |                            |              |                                |
| `MI`           | Minus / Negative           | N=1          | 负数                           |
| `PL`           | Plus / Positive or Zero    | N=0          | 正数或零                       |
| `VS`           | Overflow Set               | V=1          | 溢出发生                       |
| `VC`           | Overflow Clear             | V=0          | 无溢出                         |
| `AL`           | Always                     | (忽略标志位) | 总是执行 (等同于 `B` 指令)     |

**示例：`if (R0 > R1) then R2 = 1 else R2 = 0` (有符号比较)**

```arm
    CMP R0, R1      ; 比较 R0 和 R1 (有符号)
                    ; 假设 R0=5, R1=3. R0-R1=2. N=0, Z=0, C=1, V=0. (N=V)
                    ; 假设 R0=3, R1=5. R0-R1=-2. N=1, Z=0, C=0, V=0. (N!=V)
                    ; 假设 R0=5, R1=5. R0-R1=0. N=0, Z=1, C=1, V=0. (N=V, Z=1)

    BLE else_branch ; 如果 R0 <= R1 (有符号)，跳转到 else_branch
                    ; BLE (Less or Equal) 条件是 Z=1 OR N!=V

    ; then 代码块 (R0 > R1)
    MOV R2, #1
    B end_if_else   ; 无条件跳转到 if-else 结构结束

else_branch:
    ; else 代码块 (R0 <= R1)
    MOV R2, #0

end_if_else:
    ; ... 后续代码
    
```

#### 4. Thumb-2 中的 IT / ITE 指令 (Cortex-M 系列)

Cortex-M 系列处理器主要使用 Thumb-2 指令集，其中 IT (If-Then) 和 ITE (If-Then-Else) 指令允许最多四条后续指令根据条件码有条件地执行，而无需分支。这可以使代码更紧凑，执行效率更高。

- IT <cond>: If-Then. 下一条指令根据 <cond> 条件执行。
- ITE <cond>: If-Then-Else. 下一条指令根据 <cond> 执行 (Then)，再下一条指令根据 NOT <cond> 执行 (Else)。
- 还可以扩展到 ITT, ITTT, ITTE, ITTEE, ITEEE 等。

**示例：if (R0 == 0) R1 = R1 + 1;**

```
      CMP R0, #0
    IT EQ           ; If R0 is Equal to 0, Then...
    ADDEQ R1, R1, #1  ; ...execute this instruction
    
```

**示例：if (R0 != R1) R2 = 1; else R2 = 0;**

```
      CMP R0, R1
    ITE NE          ; If Not Equal, Then, Else
    MOVNE R2, #1    ; Then: R2 = 1 if R0 != R1
    MOVEQ R2, #0    ; Else: R2 = 0 if R0 == R1
    
```

注意：使用 IT 块时，块内的条件指令必须带有相应的条件后缀（如 ADDEQ, MOVNE）。

### 二、循环结构中的退出循环条件判断

循环结构的核心是在满足特定条件时重复执行一段代码，并在条件不再满足时退出循环。其条件判断机制与选择结构类似，也是通过设置条件码并进行条件跳转。

#### 1. 循环控制变量与条件设置

- **计数器：** 对于 for 类型的循环，通常会有一个计数器。每次循环迭代后，计数器会递增或递减。使用带 S 后缀的指令（如 SUBS 或 ADDS）来更新计数器并设置条件码。
- **状态变量/标志：** 对于 while 或 do-while 类型的循环，循环的持续依赖于某个变量的状态或特定条件是否满足。通常在循环的开始或结束处使用 CMP, TST 等指令检查这个状态，并设置条件码。

#### 2. 退出循环的条件跳转

根据设置的条件码，使用 `B<cond>` 指令来决定是继续循环（跳转回循环体开始）还是退出循环（跳转到循环体之后的代码）。

**关键点在于分支的方向：**

- **继续循环：** 如果条件满足（或不满足，取决于循环逻辑），则分支到循环体的起始标签。
- **退出循环：** 如果退出条件满足，则分支到循环体结束后的标签，或者不进行分支而顺序执行到循环体之后的代码。

**示例 1: for 循环 (递减计数器，直到为0)**

```
      ; 假设循环10次，R0 作为计数器
    MOV R0, #10         ; 初始化计数器 R0 = 10

loop_start_for:
    ; ... 循环体代码 ...

    SUBS R0, R0, #1     ; R0 = R0 - 1, 并设置标志位
                        ; 当 R0 减到 0 时，Z 标志位会被置 1
    BNE loop_start_for  ; 如果 R0 != 0 (Not Equal, Z=0)，则继续循环

loop_end_for:
    ; ... 循环结束后的代码 ...
    
```

**退出条件：** 当 SUBS R0, R0, #1 使得 R0 变为 0 时，Z 标志位置 1。此时 BNE 指令的条件 (Z=0) 不满足，因此不会跳转，程序顺序执行到 loop_end_for。

**示例 2: while (R0 != 0) 循环**

```
      ; 假设 R0 初始有一个值，循环直到 R0 为 0
; R1 用于在循环体中操作

while_condition_check:
    CMP R0, #0          ; 比较 R0 和 0
    BEQ while_end       ; 如果 R0 == 0 (Equal, Z=1)，则退出循环

    ; ... 循环体代码 ...
    ; 例如，在循环体中可能会递减 R0 或根据 R1 的某些操作改变 R0
    ; SUBS R0, R0, #1 ; 假设循环体中会改变 R0

    B while_condition_check ; 无条件跳转回条件检查

while_end:
    ; ... 循环结束后的代码 ...
    
```

**退出条件：** 在 while_condition_check 处，当 CMP R0, #0 后 Z 标志位置 1（即 R0 等于 0），BEQ while_end 指令就会执行，跳转到 while_end 标签，从而退出循环。

**示例 3: do-while (R0 > 0) 循环 (至少执行一次)**

```
      ; 假设 R0 初始有一个值，循环体至少执行一次，直到 R0 <= 0

do_while_body_start:
    ; ... 循环体代码 ...
    ; 例如，在循环体中递减 R0
    SUBS R0, R0, #1     ; R0 = R0 - 1, 并设置标志位

    ; 条件判断在循环体之后
    CMP R0, #0          ; 比较 R0 和 0 (有符号)
    BGT do_while_body_start ; 如果 R0 > 0 (Greater Than, Z=0 AND N=V)，则继续循环

do_while_end:
    ; ... 循环结束后的代码 ...
    
```

**退出条件：** 在循环体执行完毕后，CMP R0, #0 设置标志位。如果 R0 不再大于 0（即 R0 <= 0），则 BGT 指令的条件不满足，程序顺序执行到 do_while_end，退出循环。



总结来说，无论是选择结构还是循环结构，ARM 汇编都依赖于**“设置条件码 -> 条件跳转”**这一核心机制。熟练掌握各种比较指令、带S后缀的指令以及所有条件码助记符的含义，是编写正确且高效的 ARM 汇编程序的关键

------



## 4.4 C语言程序与汇编程序的相互调用

在ARM程序设计中，虽然C语言因其高效性和可移植性成为主流，但在某些特定场景下，直接使用汇编语言仍然是必要的。这些场景包括：
*   **极致性能优化：** 对时间要求极为苛刻的代码段，如中断服务程序（ISR）的特定部分、DSP算法核心等。
*   **硬件直接访问：** 操作特定的协处理器、控制特殊寄存器（如CPSR、SPSR）、执行底层硬件初始化等。
*   **利用特定指令：** 使用C语言无法直接表达的ARM/Thumb指令，如 `SWP` (ARMv5/v6, 已不推荐)、`CLZ` (Count Leading Zeros) 等（虽然现在很多有对应的内部函数）。
*   **引导代码 (Bootloader)：** 系统启动初期的硬件初始化、堆栈设置等。
*   **与现有汇编代码集成：** 调用已有的汇编库或模块。

为了使C语言和汇编语言能够有效地协同工作，必须遵循一套标准的规则，即过程调用标准。

### 4.4.1 AAPCS 标准 (ARM Architecture Procedure Call Standard)

AAPCS 是ARM架构的过程调用标准，它定义了函数调用时：
*   **寄存器的使用规则：**
    *   **参数传递：** 函数的前4个整型/指针参数通常通过寄存器 `R0-R3` 传递。如果参数超过4个，或者参数较大（如结构体），则通过栈传递。浮点参数通常通过 VFP/NEON 寄存器 (`S0-S15` / `D0-D7` 用于传递，`S0-S3`/`D0-D1` 用于返回)传递。
    *   **返回值：** 32位整型/指针返回值通常通过 `R0` 返回。64位整型返回值通过 `R0` 和 `R1` 返回。浮点返回值通过 `S0` (单精度) 或 `D0` (双精度) 返回。
    *   **调用者保存寄存器 (Caller-saved / Scratch registers)：** `R0-R3`, `R12 (IP)`, VFP/NEON `S0-S15`, `D0-D7` (以及 `Q0-Q3`)。如果调用者在函数调用后还需要这些寄存器的值，则调用者必须在调用前保存它们。被调用的函数可以自由使用这些寄存器而无需恢复它们。
    *   **被调用者保存寄存器 (Callee-saved / Variable registers)：** `R4-R11`, `SP (R13)`, `LR (R14)`, VFP/NEON `S16-S31`, `D8-D15` (以及 `Q4-Q7`)。如果被调用的函数需要使用这些寄存器，它必须在函数开始时将它们保存到栈上，并在函数返回前从栈上恢复它们。
    *   **特殊寄存器：**
        *   `SP (R13)`: 栈指针。必须始终保持8字节对齐（在公共接口处）。
        *   `LR (R14)`: 链接寄存器。存储函数调用的返回地址。
        *   `PC (R15)`: 程序计数器。
*   **栈的使用规则：**
    *   栈是满递减栈 (Full Descending Stack)，即 `SP` 指向栈顶元素，向低地址方向生长。
    *   栈帧 (Stack Frame) 的结构。
    *   栈必须在公共接口处保持8字节对齐。某些情况下（如使用NEON）可能需要更严格的对齐。
*   **数据类型的对齐和格式。**

**重要性：** 无论是C调用汇编，还是汇编调用C，都必须严格遵守AAPCS，否则会导致参数传递错误、返回值错误、寄存器内容破坏、栈混乱等严重问题。

### 4.4.2 在汇编程序中调用 C 函数

从汇编代码中调用C函数，需要遵循以下步骤：

1.  **声明C函数：** 在汇编文件中，使用 `IMPORT` (或 `EXTERN`) 伪指令声明要调用的C函数名。C编译器通常不会改变函数名（除非是C++的name mangling，此时C函数应声明为 `extern "C"`）。
    ```arm
    IMPORT c_function_name
    ```
2.  **保存调用者保存寄存器：** 如果汇编代码在调用C函数后仍需要 `R0-R3` 或 `R12` 中的值，应在调用前将它们压栈。
3.  **传递参数：**
    *   将前4个参数放入 `R0-R3`。
    *   如果参数多于4个，将第5个及以后的参数按照AAPCS的规定压入栈中（从右到左的顺序压栈，或者说，第一个栈参数在 `SP`，第二个在 `SP+4`，以此类推）。
4.  **调用函数：** 使用 `BL c_function_name` 指令调用C函数。`BL` 会将返回地址存入 `LR`。
5.  **获取返回值：** C函数返回后，返回值在 `R0` (或 `R0-R1` / `S0` / `D0`)中。
6.  **恢复栈指针：** 如果通过栈传递了参数，调用后需要调整 `SP` 以清除这些参数。
7.  **恢复调用者保存寄存器：** 如果之前压栈了，此时应出栈恢复。

**示例：**
假设有一个C函数：
```c
// c_module.c
int calculate_sum(int a, int b, int c) {
    return a + b + c;
}
```

在汇编中调用它：

```c
; asm_module.s
    AREA MyAsmCode, CODE, READONLY
    IMPORT calculate_sum  ; 声明外部C函数

call_c_function
    PUSH {LR}             ; 保存LR (虽然BL会更新LR, 但如果本汇编函数是被调用的，需要保存)
                          ; 如果R0-R3有重要数据且调用后仍需使用，也应PUSH
MOV R0, #10           ; 参数 a = 10
MOV R1, #20           ; 参数 b = 20
MOV R2, #30           ; 参数 c = 30
BL calculate_sum      ; 调用C函数

; 返回值在 R0 (此时 R0 = 60)
; 可以使用R0中的结果

POP {LR}              ; 恢复LR
BX LR                 ; 返回 (如果本函数是被调用的)

END
```

### 4.4.3 在 C 语言程序中调用汇编函数

从C代码中调用汇编函数，汇编函数必须像一个标准的C函数一样行事：

1. **声明汇编函数：** 在C文件中，使用 extern 关键字声明汇编函数的原型。

   ```
   extern int asm_function_name(int param1, char param2);
   ```

2. **导出汇编函数符号：** 在汇编文件中，使用 GLOBAL (或 EXPORT) 伪指令使汇编函数名对链接器可见。

   ```
   GLOBAL asm_function_name
   EXPORT asm_function_name ; 等效
   ```

3. **编写汇编函数 (严格遵循AAPCS)：**

   - 函数标签即为函数名 (asm_function_name)。
   - **保存被调用者保存寄存器：** 如果函数内修改了 R4-R11, LR (例如汇编函数内部又调用了其他函数)，必须在函数开始时将它们压栈。
   - **获取参数：** 从 R0-R3 和栈上获取参数。
   - **执行功能。**
   - **设置返回值：** 将结果放入 R0 (或 R0-R1 / S0 / D0)。
   - **恢复被调用者保存寄存器：** 从栈上弹出之前保存的寄存器。
   - **返回：** 使用 BX LR 指令返回到调用者。如果 LR 已被压栈，可以使用 POP {..., PC} 的方式恢复并返回。

**示例：**
一个简单的汇编函数，将两个参数相加：

```
; asm_add.s
    AREA AsmCode, CODE, READONLY
    GLOBAL asm_add      ; 导出符号
    EXPORT asm_add

asm_add
    ; 参数 R0, R1 已由C代码传入
    ; R4-R11, LR 未被修改，无需 PUSH/POP (对于这个简单例子)
    ADD R0, R0, R1      ; R0 = R0 + R1 (结果在R0)
    BX LR               ; 返回，返回值在R0
    END
```

在C语言中调用它：

```
// main.c
#include <stdio.h>

// 声明汇编函数
extern int asm_add(int a, int b);

int main() {
    int x = 5, y = 7;
    int result = asm_add(x, y);
    printf("Result from asm_add: %d\n", result); // 应输出 12
    return 0;
}
```

**编译链接 (GCC示例)：**

```
arm-none-eabi-as -o asm_add.o asm_add.s
arm-none-eabi-gcc -c -o main.o main.c
arm-none-eabi-gcc -o program.elf main.o asm_add.o
```

### 4.4.4 嵌入汇编 (Embedded Assembly - 指使用独立的汇编文件)

“嵌入汇编”这个术语有时会有歧义。在此目录结构中，它更可能指的是**将汇编代码编写在独立的 .s 或 .S 文件中，然后与C代码一起编译链接**。这正是上面 **4.4.2** 和 **4.4.3** 所描述的方式。

**优点：**

- **清晰分离：** C代码和汇编代码在不同文件中，逻辑清晰。
- **完整控制：** 可以使用汇编器的所有功能和伪指令。
- **大型模块：** 适合编写较长或较复杂的汇编例程。
- **可重用性：** 汇编模块可以被多个C文件或其他汇编模块调用。

**缺点：**

- **函数调用开销：** 每次调用都有函数调用的开销（参数传递、寄存器保存/恢复、跳转）。
- **上下文切换：** 编译器对C代码的优化可能因外部函数调用而受限。

这种方式是最规范和推荐的混合编程方式，尤其对于功能明确、相对独立的汇编模块。

### 4.4.5 内联汇编 (Inline Assembly - asm 关键字)

内联汇编允许将汇编指令直接嵌入到C/C++代码中。这避免了函数调用的开销，并允许对局部代码段进行精细控制。不同的编译器有不同的内联汇编语法。

**1. GCC (GNU Compiler Collection) 内联汇编语法 (也适用于 Clang)：**
这是最常用和功能最强大的内联汇编形式。
基本格式：

```
asm volatile (
    "assembler template"
    : output operands  /* 可选 */
    : input operands   /* 可选 */
    : clobber list     /* 可选 */
);
```

- asm (或 __asm__)：关键字。
- volatile (或 __volatile__)：可选。告诉编译器不要优化掉这段汇编代码，也不要随意移动它。对于访问硬件或有副作用的汇编，几乎总是需要 volatile。
- **"assembler template"：** 包含汇编指令的字符串。指令间用 \n\t 分隔。可以使用占位符（如 %0, %1）引用C语言变量。%% 表示一个 % 字符。
- **output operands：** 描述汇编代码如何修改C变量。格式为 "[constraint]"(variable)。
  - =r: 变量将被写入一个通用寄存器。
  - +r: 变量将被读写一个通用寄存器。
  - =m: 变量在内存中，汇编代码直接操作内存。
  - 还有许多其他约束，如特定寄存器 (=w for VFP/NEON)。
- **input operands：** 描述汇编代码如何读取C变量。格式为 "[constraint]"(variable)。
  - r: 变量从一个通用寄存器读。
  - m: 变量从内存读。
  - i: 立即数。
- **clobber list：** 告知编译器哪些寄存器或资源（如 "memory", "cc" for condition codes）被汇编代码修改了，但没有在输出操作数中列出。这非常重要，编译器需要这些信息来正确生成代码和管理寄存器。
  - 例如："r0", "r1", "lr", "memory", "cc"。

**GCC 内联汇编示例：**

```
// 禁用中断 (Cortex-M)
void disable_interrupts(void) {
    asm volatile ("cpsid i" : : : "memory");
}

// 加法操作
int add_inline(int a, int b) {
    int sum;
    asm volatile (
        "ADD %0, %1, %2"   // %0 = %1 + %2
        : "=r" (sum)       // 输出操作数：sum (分配到某个寄存器)
        : "r" (a), "r" (b) // 输入操作数：a, b (分配到某个寄存器)
        : "cc"             // Clobber: 条件码寄存器被ADD指令修改
    );
    return sum;
}

// 访问特殊寄存器 (读取 CONTROL 寄存器)
unsigned int read_control_reg(void) {
    unsigned int ctrl_val;
    asm volatile ("MRS %0, CONTROL" : "=r" (ctrl_val));
    return ctrl_val;
}
```

**2. ARM Compiler (ArmCC / Keil MDK) 内联汇编语法：**
ARM 编译器（如 Keil MDK 中的 armcc 或 armclang）支持另一种形式的内联汇编。

- **__asm 关键字 (C/C++) 或 ASMARM / ASMTUMB (旧版 C)：**

  ```
  __asm {
      instruction1
      instruction2 [operand1, operand2, ...]
      // ...
  }
  ```

- **访问 C 变量：** 通常可以直接在 __asm 块中使用作用域内的 C 变量名，编译器会处理它们的加载和存储。

- **限制：**

  - 不能直接指定寄存器分配。
  - 编译器有更多自由度来优化。
  - 不如 GCC 风格灵活。

**ARM Compiler 内联汇编示例：**

```
// 禁用中断 (Cortex-M)
__forceinline void disable_interrupts_keil(void) {
    __asm { CPSID I }
}

// 加法操作
int add_inline_keil(int a, int b) {
    int sum;
    __asm {
        ADD sum, a, b  // 编译器处理 a, b, sum 的寄存器分配
    }
    return sum;
}

// 读取 CONTROL 寄存器
unsigned int read_control_reg_keil(void) {
    unsigned int ctrl_val;
    __asm { MRS ctrl_val, CONTROL }
    return ctrl_val;
}
```

**__forceinline** 提示编译器尽可能内联该函数。

**内联汇编的优点：**

- **无函数调用开销：** 指令直接插入，非常高效。
- **与C代码紧密集成：** 方便访问C变量。
- **精细控制：** 对特定代码片段进行底层操作。

**内联汇编的缺点：**

- **编译器依赖：** 语法和行为因编译器而异，可移植性差。
- **复杂性：** 编写和调试困难，容易出错（尤其是GCC风格的约束和破坏列表）。
- **可读性差：** 降低代码整体的可读性。
- **可能干扰优化：** 如果使用不当，可能阻止编译器进行某些优化。

**何时使用内联汇编：**

- 当需要执行一两条无法用C表达或效率极低的指令时。
- 当函数调用开销相对于汇编代码本身过大时。
- 对于非常小且对性能极致要求的片段。

### 4.4.6 内部函数 (Intrinsic Functions / Compiler Intrinsics)

内部函数是由编译器提供的一些特殊函数，它们在C代码中看起来像普通函数调用，但在编译时会被替换成一条或几条特定的、通常是高效的汇编指令。

**目的：**

- 提供一种可移植性比内联汇编更好的方式来访问处理器特定功能。
- 让编译器能够理解这些操作的语义，从而进行更好的优化。
- 简化对常用特殊指令的调用。

**ARM 相关的内部函数 (部分示例，具体查阅编译器文档和 ARM C Language Extensions - ACLE)：**

- **饱和运算指令：** __ssat() (有符号饱和), __usat() (无符号饱和)
- **SIMD/NEON 指令：** 大量的 v... 系列函数，如 vadd_s16() (向量加法)。
- **DSP 指令 (Cortex-M4/M7/M33/M35P/M55/M85)：**
  - __SMLABB (Signed Multiply Accumulate Bottom Bottom)
  - __SMUAD (Signed Dual Multiply Add)
- **状态寄存器访问：**
  - __get_CPSR() (ARM Compiler, 获取CPSR)
  - __set_CPSR() (ARM Compiler, 设置CPSR)
  - __get_CONTROL() (获取CONTROL寄存器)
  - __set_CONTROL() (设置CONTROL寄存器)
- **中断控制：**
  - __enable_irq()
  - __disable_irq()
  - __enable_fault_irq()
  - __disable_fault_irq()
- **特殊指令：**
  - __NOP() (空操作)
  - __WFI() (Wait For Interrupt)
  - __WFE() (Wait For Event)
  - __ISB() (Instruction Synchronization Barrier)
  - __DSB() (Data Synchronization Barrier)
  - __DMB() (Data Memory Barrier)
  - __CLZ() (Count Leading Zeros)
  - __REV() (Reverse byte order)
- **原子操作 (ARMv7-M, ARMv8-M)：** __LDREX, __STREX 系列用于实现独占访问。

**示例 (使用GCC/Clang兼容的ACLE内部函数)：**

```
#include <arm_acle.h> // 可能需要包含特定的头文件

void wait_for_event_example(void) {
    __wfe(); // 调用WFE指令
}

int count_leading_zeros_example(unsigned int val) {
    return __clz(val); // 调用CLZ指令
}

void data_sync_barrier_example(void) {
    __dsb(0xF); // Full system DSB
}
```

**内部函数的优点：**

- **可移植性较高：** 相较于内联汇编，只要编译器支持相同的内部函数集 (如ACLE)，代码就可以在不同编译器间移植。
- **易用性：** 调用方式与普通C函数类似，更易读写。
- **编译器友好：** 编译器知道内部函数的语义，可以更好地进行优化。
- **类型检查：** 编译器可以对内部函数的参数进行类型检查。

**内部函数的缺点：**

- **功能受限：** 只能使用编译器提供的内部函数，无法执行任意汇编指令。
- **仍然依赖编译器：** 不同编译器支持的内部函数集可能不完全相同，尽管有ACLE这样的标准。

**总结 C 与汇编互调用的选择：**

| 方法                     | 优点                                     | 缺点                                           | 适用场景                                                   |
| ------------------------ | ---------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| **独立汇编文件**         | 清晰分离, 完整控制, 大型模块, 可重用     | 函数调用开销, 可能影响C优化                    | 独立功能模块, 复杂汇编例程, 驱动程序, Bootloader           |
| **GCC内联汇编**          | 无调用开销, 与C紧密集成, 精细控制        | 编译器依赖, 复杂难调试, 可读性差, 可能干扰优化 | 短小、性能极致要求的代码片段, 直接硬件访问, 特殊指令调用   |
| **ARM Compiler内联汇编** | 无调用开销, 语法相对简单                 | 编译器依赖, 不如GCC灵活, 可能干扰优化          | 同GCC内联汇编, 尤其在Keil MDK等ARM Compiler环境中          |
| **内部函数**             | 可移植性较高, 易用, 编译器友好, 类型检查 | 功能受限, 仍依赖编译器支持                     | 常用特殊指令(DSP, SIMD, 同步), 简单硬件控制(中断, WFI/WFE) |

在实际开发中，应优先考虑使用C语言。如果确实需要汇编：

1. 首先查找是否有**内部函数**可以满足需求。
2. 如果内部函数不足以满足，且代码片段短小、对性能要求极致，可以考虑**内联汇编**。
3. 如果需要编写较复杂的汇编模块或需要更好的代码组织，则使用**独立的汇编文件**。

始终确保遵循 AAPCS 标准，这是C与汇编成功交互的基石。