---
title: 总线技术
publishDate: 2025-10-11
description: "微机原理第六章-总线技术"
tags: ['微机原理']
language: 'Chinese'
first_level_category: "知识库"
second_level_category: "微机原理"
draft: false
---

# ARM/Thumb-1/2 指令集详解

## 一、 引言：ARM架构与RISC理念

ARM (Advanced RISC Machines) 架构是典型的**精简指令集计算机 (RISC)** 架构。其核心设计理念包括：

*   **Load/Store 架构：** 只有加载 (`LDR`) 和存储 (`STR`) 指令可以直接访问内存。所有其他算术逻辑运算指令（如 `ADD`, `SUB`）都只操作寄存器或立即数。
*   **大量通用寄存器：** 通常提供 R0-R12 作为通用寄存器，R13 作为堆栈指针 (SP)，R14 作为链接寄存器 (LR)，R15 作为程序计数器 (PC)。
*   **指令长度：**
    *   **ARM 状态：** 指令为 32 位定长。
    *   **Thumb 状态 (Thumb-1)：** 指令为 16 位定长，以提高代码密度，但功能相对精简。
    *   **Thumb-2 技术：** ARMv6T2 及之后架构引入，是 Thumb 指令集的扩展。它混合使用 16 位和 32 位指令，旨在同时获得 Thumb 的代码密度和接近 ARM 指令集的性能。Cortex-M 系列处理器（如 Cortex-M4）主要使用 Thumb-2 指令集（不支持 ARM 状态）。
*   **流水线效率：** 相对规整的指令格式和简单的操作有利于实现高效的指令流水线。
*   **条件执行：** 许多 ARM 指令可以根据 APSR (应用层序状态寄存器) 中的条件标志位来决定是否执行。Thumb-1 主要支持条件分支，Thumb-2 通过 `IT` (If-Then) 指令块扩展了条件执行能力。
*   **S 后缀：** 许多数据处理指令可以添加 `S` 后缀 (如 `ADDS`, `MOVS`)，表示该指令执行后会更新 APSR 中的条件标志位 (N, Z, C, V)。

### 指令基本格式 (通用概念)

​	MNEMONIC {S}  {cond}  Rd, Rn, Operand2

*   `MNEMONIC`: 指令助记符 (如 `ADD`, `MOV`)。
*   `{S}`: 可选后缀，若存在，则指令执行结果会影响 APSR 中的条件标志位。
*   `{cond}`: 可选的条件码 (如 `EQ`, `NE`, `GT` 等)。
*   `Rd`: 目标寄存器。
*   `Rn`: 第一个操作数寄存器 (通常作为基址或源操作数)。
*   `Operand2`: 第二个操作数，其形式多样，是寻址方式的体现。可以是：
    *   立即数: `#<immediate>`
    *   寄存器: `Rm`
    *   寄存器移位: `Rm, <shift_op> #<shift_imm>` 或 `Rm, <shift_op> Rs`

## 二、ARM 指令的寻址方式 (3.2)

寻址方式决定了指令如何获取其操作数。

### 3.2.1 立即寻址 (Immediate Addressing)

操作数直接包含在指令中。

*   **描述：** 操作数是一个常量值，编码在指令的机器码中。
*   **使用方法：**指令中的立即数要以"#"为前缀。用十六进制表示，需要在 # 后面加`0x`或`&`;二进制表示使用`0b`;十进制为`0b`或者不写。
*   **示例：**
    
    *   `MOV R0, #10`       ; R0 = 10
    *   `ADD R1, R2, #0xFF` ; R1 = R2 + 255
    *   `ADD R5, R1, #3`    ; R5 = R1 + 3
    *   `MOV R0,R1,#0b0010`; R0 = R1 + 2
    
    *注意：ARM 指令对立即数的编码有特定规则 (如一个8位常数循环右移偶数位得到)，Thumb-1 指令的立即数范围较小，Thumb-2 的32位指令可以支持更广泛的立即数。*

### 3.2.2 寄存器直接寻址 (Register Direct Addressing)

操作数在通用寄存器中。

*   **描述：** 指令直接指定包含操作数的寄存器。
*   **示例：**
    *   `MOV R0, R1`       ; R0 = R1
    *   `ADD R2, R3, R4`   ; R2 = R3 + R4
    *   `SUB R0, R1, R2`   ; R0 = R1 - R2

### 3.2.3 寄存器移位寻址 (Register Shift Addressing)

操作数是寄存器的内容经过移位操作后的结果。这是 ARM 数据处理指令 `Operand2` 的一种强大形式。

*   **描述：** 第二个操作数寄存器 `Rm` 的内容在被使用前，先进行移位操作。移位位数可以是立即数或另一个寄存器 `Rs` 的低8位。
*   **移位类型：**
    *   `LSL`: 逻辑左移 (Logical Shift Left)，低位补0。逻辑左移一位等价于**无符号数**乘以2(不考虑最高位符号数)-->**快速运算**！ 例如：0x0000 0440 LSL#2  ==> 0x0000 1100(2*4=8，进位)
    *   `LSR`: 逻辑右移 (Logical Shift Right)，高位补0。
    *   `ASR`: 算术右移 (Arithmetic Shift Right)，高位用**原符号位**填充。`Rm`中的值向右移动，左端用`Rm`第31位的值填充。算术右移一位等价于**有符号数**除以2
    *   `ROR`: 循环右移 (Rotate Right)，**移出的低位填充到高位**。
    *   `RRX`: 带扩展的循环右移一位 (Rotate Right with Extend)，原 C 标志位移入最高位，移出的最低位成为新的 C 标志位。(四个标志位: N V Z C 详见转移指令一章)
*   **示例：**
    *   `MOV R0, R1, LSL #2`    ; R0 = R1 << 2
    *   `ADD R0, R1, R2, LSR #4` ; R0 = R1 + (R2 >> 4) (逻辑右移) 
    *   `SUB R0, R1, R2, ASR R3` ; R0 = R1 - (R2 算术右移 R3 位)
    *   `MOV R0, R1, ROR #3`; R1 = `0xFFFF FF31`(FF FF FF 0011 0001)，则`ROR`后为`0x3FFF FFE6` (0011FFF FF 1110 0110 )
    *   `MOV R0,R1,ASR #3`; R0 = R1算术右移3位后的值，假设R1 = `0xFFFF FF31`,则算术右移后是`0xFFFF FFE6`。

### 3.2.4 寄存器间接寻址 (Register Indirect Addressing)

操作数在内存中，其地址由寄存器给出。主要用于 `LDR/STR` 等访存指令。

* **描述：** 第二个操作数寄存器 `Rm` 中存放的是操作数在内存中的地址。

* **解释**： 寄存器(设为`Rm`)中存放的内容是操作数的内存地址。`Rm`实际上是内存操作数的**地址指针**，它在指令中要用" [ ] "括号括起来。也就是说，**[Rm]实际上存的是要进行操作的那块内存单元的地址，其等价于一个指针**

*   **示例：**
    
    * `LDR R0, [R1]`   ; R0 = Memory[R1]  将R1的值作为地址，将内存中该地址单元的数据传给R0寄存器，执行后R1的值不变
    
    * `STR R0, [R1]`   ; Memory[R1] = R0  将R0中的值传到以 R1 的值作为地址的存储器中。
    
      ![image-20250526202415801](./imgs/image-20250526202415801.png)

### 3.2.5 基址变址寻址 (Base-indexed Addressing)

操作数在内存中，其地址由基址寄存器和偏移量计算得到。主要用于 `LDR/STR` 等访存指令。

*   **描述：** 内存地址 = 基址寄存器 `Rn` + 偏移量 (`#offset` 或 `Rm`，可带移位)。
*   **解释**： **某个寄存器(一般称为基址寄存器)`Rn`提供一个基准地址，该基准地址与指令中给出的被称为“地址偏移量”（变址）的数据相加，形成操作数的有效地址**。一般用来访问基准地址附近的地址单元，进行查表、数组操作等。 ---- 寄存器间接寻址可看作偏移量为 0 的基地址变址寻址
*   **子类型：**
    *   **预变址 (Pre-indexed)/前索引变址寻址：** **先计算地址，再访问内存**。也就是先将`Rn`的值加上偏移量(立即数或者寄存器)后的值作为寻址地址。`Rn` 值不变。
        *   `LDR R0, [R1, #4]`    ; R0 <--- [R1 + 4], 先计算R1的值加上立即数 4 (十进制)得到的地址，按这个地址寻址，将所在地址的内存单元的值赋值给 R0
        *   `STR R0, [R1, R2]`    ; Memory[R1 + R2] = R0
        *   `LDR R0, [R1, R2, LSL #2]` ; R0 = Memory[R1 + (R2 << 2)]
        *   **预变址带回写 (Pre-indexed with Writeback)：** 先计算地址，访问内存，**然后将新地址写回基址寄存器** `Rn`。
        *   `LDR R0, [R1, #4]!`   ; R0 = Memory[R1 + 4]; R1 = R1 + 4
        *   **感叹号（`!`）是回写（Writeback）符号**，表示指令执行后更新存放地址的寄存器的值，也就是将最终地址赋值给基址寄存器`Rn`
    *   **后变址 (Post-indexed)：** **先用 `Rn` 的值作为地址访问内存，然后将 `Rn` 与偏移量相加的结果写回基址寄存器 `Rn`**。和预变址相反，先访问，再计算地址
        *   `LDR R0, [R1], #4`    ; R0 = Memory[R1]; R1 = R1 + 4
        *   `STR R0, [R1], R2`    ; Memory[R1] = R0; R1 = R1 + R2

### 3.2.6 多寄存器直接寻址 (Multiple Register Direct Addressing)

用于 `LDM` (Load Multiple) 和 `STM` (Store Multiple) 指令，一次性操作多个寄存器。

* **描述：** 指令中直接列出要加载或存储的寄存器列表。基址寄存器 `Rn` 提供起始内存地址。

* **格式**： `LDM`(加载多个寄存器)、`STM`(存储多个寄存器)指令可以读写存储器中**多个连续数据传送**到处理器的**多个存储器**。可使用一条指令完成。**要操作的多个寄存器若是连续，则用 " - " (减号)连接，不连续则用 " , " 分隔**

*   **寻址模式 (决定地址增减和 Rn 是否回写)：**
    
    *   `IA` (Increment After): 每次操作后**递增**地址（先访问地址，再更新）。也就是说，先读写，每次**结束后地址加 4字节(**下面同理都是 4
    *   `IB` (Increment Before): 每次操作前**递增**地址（先更新地址，再访问）。
    *   `DA` (Decrement After): 每次操作后**递减**地址（先访问地址，再更新）。
    *   `DB` (Decrement Before):每次操作前**递减**地址（先更新地址，再访问）。**每次读写结束前地址先递减 4**
    *   `!` 后缀表示 `Rn` 会被更新为**最后访问的地址 +/- 一个字长**（取决于操作）。
    
*   **示例：**
    
    * `STMIA R0!, {R1-R5, LR}` ;[R0] <--- R1, [R0 + 4] <--- R2, [R0 + 8] <--- R3,......[R0 + 24] <--- LR, R0 = R0 +24;**回写只在最后一次执行，前面是 IA 在进行递增操作。**
    
      例如，R0 最初地址是 0x1000，则过程为：[0x1000] <---R1,[0x1004] <---R2,......[0x1014]<---LR,最后，此时 R0 的值仍然是 0x1000，R0 = 0x1014 + 4 = 0x1018。
    
    * `LDMDB R1, {R2-R4}`    ;   R2 <--- [R1 - 4], R3 <--- [R1 - 8], R4<--- [R1 -12]   从R1指向的地址（递减）加载R2-R4，R1不回写。**以 R1 寄存器的值作为存储器的寻址地址**，R1先递减 4 (R1 - 4)，将 R1 - 4 的值(这是内存单元地址！)对应内存单元存储的值赋给R2；再递减 4 (R1 - 8)，将R1 - 8 的值所对应内存单元存储的值赋给 R3；R1 - 12 地址对应的值赋给 R4 寄存器。
    
      **请注意：R1相当于一个基址寄存器，是一个指针，存的是内存单元地址！**
    
      ![image-20250526210542244](./imgs/image-20250526210542244.png)

### 3.2.7 堆栈寻址 (Stack Addressing)  Cortex-M4的堆栈模型与堆栈对齐

一种特殊的基址变址寻址，通常使用 SP (R13) 作为基址寄存器，并结合 `LDM/STM` 指令实现。

* **描述：** `LDM/STM` 指令配合不同的寻址模式 (IA, IB, DA, DB) 和 SP 寄存器来实现堆栈操作 (压栈 PUSH, 出栈 POP)。

* ![image-20250526212044163](./imgs/image-20250526212044163.png)

* ![image-20250526211942018](./imgs/image-20250526211942018.png)

* ![image-20250526212006179](./imgs/image-20250526212006179.png)

*   **ARM 堆栈约定：**
    
    *   FD (Full Descending): 堆栈满时向下增长。`STMDB SP!, {regs}` / `LDMIA SP!, {regs}`
    *   FA (Full Ascending): 堆栈满时向上增长。`STMIB SP!, {regs}` / `LDMDA SP!, {regs}`
    *   ED (Empty Descending): 堆栈空时向下增长。`STMED SP!, {regs}` / `LDMFD SP!, {regs}`
    *   EA (Empty Ascending): 堆栈空时向上增长。`STMEA SP!, {regs}` / `LDMFA SP!, {regs}`
    *   最常用的是 FD 堆栈。
    
* **示例 (FD 堆栈)：**

  *   `PUSH {R0, R1, LR}`  ; 等效于 STMDB SP!, {R0, R1, LR}
  *   `POP {R0, R1, PC}`   ; 等效于 LDMIA SP!, {R0, R1, PC} (PC用于函数返回)

* **Cortex-M3/4 只使用 FD 的堆栈模型，栈顶指针是SP(R13寄存器)。SP指针的最低两位是`00`，因为栈操作地址必须对齐到32位的字边界上！(出栈、入栈都是 4 字节，也就是一个 字)**

* **示例**:

  ![image-20250526212726459](./imgs/image-20250526212726459.png)

![image-20250526212746945](./imgs/image-20250526212746945.png)

![image-20250526212900751](./imgs/image-20250526212900751.png)

### 综合示例

![image-20250526212959499](./imgs/image-20250526212959499.png)

![image-20250526213014877](./imgs/image-20250526213014877.png)

![image-20250526213027317](./imgs/image-20250526213027317.png)

## 三、ARM 核心指令 (3.3)

### 3.3.1 数据传送指令 (Data Transfer Instructions)

在寄存器之间或立即数到寄存器之间传送数据。

- 两个寄存器间进行数据传送
- 普通寄存器与特殊功能寄存器间的数据传送
- 把一个立即数加载到寄存器

1.  **`MOV` (Move)** 
    *   **语法：** `MOV{S}{cond} Rd, Operand2`
    *   **功能：** `Rd = Operand2`
    *   **标志位{S}**：
    *   **示例：**
        *   `MOV R1, R5`      ;R1 = R5
        *   `MOV R0, #0x12`   ; R0 = 0x12
        *   `MOVS R0, R1`     ;  R0 = R1, **并更新 N, Z 标志**
        *   `MOV R0, R1, LSL #2` ; R0 = R1 << 2
2.  **`MVN` (Move Not)**
    *   **语法：** `MVN{S}{cond} Rd, Operand2`
    *   **功能：** `Rd = NOT Operand2` (按位取反)   
    *   **示例：** 
        - **MVN Rd,Rm** 将寄存器`Rm`的值取反后再传给`Rd`
        - `MVN R0, R1`

### 3.3.2 存储器访问指令 (Memory Access Instructions)

​	Cortex-M微处理器只能通过加载(load)或存储(store)指令对存储器(内存)进行操作。

指令分为两种：单个寄存器加载和存储指令 LDR/STR，多寄存器加载和存储指令 LDM/STM

**字访问（`LDR`/`STR`）**：要求地址 4 字节对齐，否则可能触发异常

1. **`LDR` (Load Register)**: 从存储器的内容加载到寄存器。

   *   **语法：** `LDR{cond}{type} Rd, <addressing_mode>`
   *   在 ARM 架构中，**32 位字（Word）的访问要求地址是 4 字节对齐的**。**4 字节对齐**指的是内存地址是 4 的倍数（即地址的二进制表示中最低两位为 `00`）
   *   **数据类型**：加载的数据类型可以是字节(B)、半字(H)、一个字(无标志位)、双字(D)。即下面的{type}
   *   **`{type}`:**
       *   `B`: Byte (8位)，零扩展。
       *   `SB`: Signed Byte (8位)，符号扩展。
       *   `H`: Halfword (16位)，零扩展。
       *   `SH`: Signed Halfword (16位)，符号扩展。
       *   `D`:双字。
       *   (无): Word (32位)。
   *   **示例：**
       *   `LDRB Rd,[R1 #offset]`；从地址 R1+offset(偏移量) 读取一个字节(8位)到 Rd
       *   `LDR R0, [R1]`        ; R0 <--- [R1]  从地址为 R1 的内存中**读取32位数据**，加载到 R0 
       *   `LDRB R2, [R1, #4]`   ; R2 <--- [R1 + 4]一个字节数据
       *   `LDRSH R3, [R1, R2]`  ; R3 <--- [R1 + R2]一个**两字节数据(高字节是低字节的符号位扩展)**

2. **`STR` (Store Register)**: 将寄存器数据存储到内存。

   * **语法：** `STR{cond}{type} Rt, <addressing_mode>`

   *   **`{type}`:**
       
       *   `B`: Byte (存储 Rt 低8位)。
       *   `H`: Halfword (存储 Rt 低16位)。
       *   (无): Word (存储 Rt 32位)。
       
   *   **示例：**
       
       *   `STR R0, [R1]`        ; **Memory[R1] = R0 将 R0 的32位数据存到地址为 R1 的内存中**
       *   `STRH R2, [R1, #-2]!` ; Memory[R1-2] = R2 (低16位); R1 = R1-2
       
       操作和LDR正好相反。
       
       `LDR R0, [R1]`   ; **R0 = Memory[R1]  从地址为 R1 的内存中读取32位数据，加载到 R0** 
       
       `STR R0, [R1]`   ; **Memory[R1] = R0 将 R0 的32位数据存到地址为 R1 的内存中**

3. **`LDM` (Load Multiple)** / **`STM` (Store Multiple)**

   ​	**`LDM`（Load Multiple）** 和 **`STM`（Store Multiple）** 是用于批量加载或存储多个寄存器的指令，也称为**块传输指令**。它们常用于栈操作、寄存器备份 / 恢复、内存块复制等场景，能显著提高数据传输效率。

   * **语法：** `LDM{cond}<mode> Rn{!}, {reg_list}` / `STM{cond}<mode> Rn{!}, {reg_list}`

   *   ```assembly
       LDM{条件}{模式} 基址寄存器, {寄存器列表}
       STM{条件}{模式} 基址寄存器, {寄存器列表}
       ```

   * **`<mode>`:** `IA`, `IB`, `DA`, `DB` (如 3.2.6 和 3.2.7 所述)。

   *   **指令**：
       
       - `LDMIA Rd!,{寄存器列表}`：从地址 Rd 处读取多个字。每读一个字，Rd 自增一次，每次自增4字节。**将最后一次自增的地址作为 Rd 新的地址**。例如 Rd 初始为 0x0000，自增四次后地址为 0x0010，然后将 0x0010 赋值给 Rd。
       - `STMDB Rd!,{寄存器列表}`：存储多个字(word)到地址 Rd 处，每次存储前先自减，最后一次自减后将此地址赋值给 Rd。
       
   *   **示例：**
       
       * `STMDB SP!, {R4-R7, LR}` ; 压栈 R4-R7, LR。等价于 `PUSH {R4-R7,LR}`。SP - 4, [SP-4] <--- R4, SP - 8, [SP-8] <--- R5,......[SP - 20] <--- LR, SP = SP - 20 - 4. 
       
         ![image-20250527113248319](./imgs/image-20250527113248319.png)
       
       *   `LDMIA SP!, {R4-R7, PC}` <==> `POP {R4-R7,PC}`; 出栈 R4-R7, PC (常用于函数返回)

### 3.3.3 算术运算指令 (Arithmetic Operation Instructions)

1.  **`ADD` (Add)** 
    *   **描述**：常规加法
    *   **语法：** `ADD{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn + Operand2`
    *   **示例：**
        *   `ADD R0, R1`      ; (Thumb 2-operand, 来自用户图一) R0 = R0 + R1
        *   `ADD R2, R3, R4`  ; (来自用户图一) R2 = R3 + R4
        *   `ADDS R0, R1, #10` ; R0 = R1 + 10, 更新 N,Z,C,V
    
2.  **`SUB` (Subtract)**
    *   **描述**：常规减法
    *   **语法：** `SUB{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn - Operand2`
    *   **示例：**
        *   `SUB R2, R3`      ; (Thumb 2-operand, 来自用户图一) R2 = R2 - R3
        *   `SUBS R0, R1, R2, LSL #1` ; R0 = R1 - (R2<<1), 更新 N,Z,C,V
    
3.  **`ADC` (Add with Carry)**
    *   **描述**：带进位的加法
    *   **语法：** `ADC{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn + Operand2 + C_flag` (APSR中的C标志位)
    
4.  **`SBC` (Subtract with Carry/Borrow)**
    *   **描述**：带借位的减法
    *   **语法：** `SBC{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn - Operand2 - NOT(C_flag)`(APSR中的C标志位的取反)
    
5.  **`RSB` (Reverse Subtract)**
    *   **描述**：反向减法，即操作数减去寄存器`Rn`存储的值
    *   **语法：** `RSB{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Operand2 - Rn`
    *   **示例：** `RSBS R0, R1, #0` ; R0 = 0 - R1 (取负并设置标志)
    
6.  **`RSC` (Reverse Subtract with Carry)**
    *   **语法：** `RSC{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Operand2 - Rn - NOT(C_flag)`

### 3.3.4 逻辑运算指令 (Logical Operation Instructions)

1.  **`AND` (Logical AND)**
    *   **语法：** `AND{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn AND Operand2`

2.  **`ORR` (Logical OR)**
    *   **语法：** `ORR{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn OR Operand2`

3.  **`EOR` (Logical Exclusive OR)**
    *   **语法：** `EOR{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn EOR Operand2`

4.  **`BIC` (Bit Clear)**
    *   **语法：** `BIC{S}{cond} Rd, Rn, Operand2`
    *   **功能：** `Rd = Rn AND (NOT Operand2)`

### 3.3.5 移位和循环指令 (Shift and Rotate Instructions)

这些是独立的移位指令，与数据处理指令中的移位操作数不同（后者不直接将移位结果存回操作数寄存器，而是作为 `Operand2`）。

*   **`LSL{S}{cond} Rd, Rm, Rs/#imm`**: Rd = Rm << (Rs 或 #imm)
*   **`LSR{S}{cond} Rd, Rm, Rs/#imm`**: Rd = Rm >> (Rs 或 #imm) (逻辑右移)
*   **`ASR{S}{cond} Rd, Rm, Rs/#imm`**: Rd = Rm >> (Rs 或 #imm) (算术右移)
*   **`ROR{S}{cond} Rd, Rm, Rs/#imm`**: Rd = Rm 循环右移 (Rs 或 #imm) 位
*   **`RRX{S}{cond} Rd, Rm`**: Rd = (C_flag << 31) | (Rm >> 1)

### 3.3.6 符号扩展指令 (Sign Extend Instructions)

主要用于将较小位宽的有符号数或无符号数扩展到32位。

*   **`SXTB{cond} Rd, Rm {, ROR #rotation}`**: 符号扩展字节 (Rm的bit[7:0]) 到32位。
*   **`SXTH{cond} Rd, Rm {, ROR #rotation}`**: 符号扩展半字 (Rm的bit[15:0]) 到32位。
*   **`UXTB{cond} Rd, Rm {, ROR #rotation}`**: 零扩展字节 (Rm的bit[7:0]) 到32位。
*   **`UXTH{cond} Rd, Rm {, ROR #rotation}`**: 零扩展半字 (Rm的bit[15:0]) 到32位。
    *可选的 `ROR` 允许在扩展前对 `Rm` 进行循环右移 (0, 8, 16, 24)。*

### 3.3.7 字节调序指令 (Byte Reordering Instructions)

用于在寄存器内反转字节顺序，处理大小端转换。

*   **`REV{cond} Rd, Rm`**: 反转 `Rm` 中的字节顺序 (e.g., `0x12345678` -> `0x78563412`)。
*   **`REV16{cond} Rd, Rm`**: 在 `Rm` 的每个半字内反转字节顺序 (e.g., `0x12345678` -> `0x34127856`)。
*   **`REVSH{cond} Rd, Rm`**: 反转 `Rm` 低半字的字节顺序，并进行符号扩展 (e.g., `Rm=0x...12F0` -> `Rd=0xFFFFF012`)。

### 3.3.8 位域处理指令 (Bitfield Processing Instructions)

用于对寄存器中的位域进行清除、插入、提取。

*   **`BFC{cond} Rd, #lsb, #width`**: 位域清除。清除 `Rd` 中从 `lsb` 位开始，宽度为 `width` 的位。
*   **`BFI{cond} Rd, Rn, #lsb, #width`**: 位域插入。将 `Rn` 的低 `width` 位插入到 `Rd` 从 `lsb` 位开始的位置，`Rd` 的其他位不变。
*   **`SBFX{cond} Rd, Rn, #lsb, #width`**: 有符号位域提取。从 `Rn` 中提取从 `lsb` 位开始，宽度为 `width` 的位域，并进行符号扩展到 `Rd`。
*   **`UBFX{cond} Rd, Rn, #lsb, #width`**: 无符号位域提取。从 `Rn` 中提取从 `lsb` 位开始，宽度为 `width` 的位域，并进行零扩展到 `Rd`。

### 3.3.9 比较和测试指令 (Compare and Test Instructions)

执行运算但不保存结果，仅用于更新 APSR 中的条件标志位。

1.  **`CMP` (Compare)**
    *   **语法：** `CMP{cond} Rn, Operand2`
    *   **功能：** 计算 `Rn - Operand2`，更新 N, Z, C, V 标志。**不保存结果。**
    *   **示例：** `CMP R2, R3` ; (来自用户图一) 比较 R2 和 R3

2.  **`CMN` (Compare Negative)**
    *   **语法：** `CMN{cond} Rn, Operand2`
    *   **功能：** 计算 `Rn + Operand2`，更新 N, Z, C, V 标志。**不保存结果。**

3.  **`TST` (Test)**
    *   **语法：** `TST{cond} Rn, Operand2`
    *   **功能：** 计算 `Rn AND Operand2`，更新 N, Z 标志 (C 标志可能由移位器设置)。**不保存结果。**

4.  **`TEQ` (Test Equivalence)**
    *   **语法：** `TEQ{cond} Rn, Operand2`
    *   **功能：** 计算 `Rn EOR Operand2`，更新 N, Z 标志 (C 标志可能由移位器设置)。**不保存结果。**

### 3.3.10 子程序调用与无条件转移指令 (Branch/Call Instructions)

1.  **`B` (Branch)**
    *   **语法：** `B{cond} label`
    *   **功能：** 跳转到 `label`。

2.  **`BL` (Branch with Link)**
    *   **语法：** `BL{cond} label`
    *   **功能：** 将下一条指令地址存入 LR (R14)，然后跳转到 `label` (用于子程序调用)。

3.  **`BX` (Branch and Exchange)**
    *   **语法：** `BX{cond} Rm`
    *   **功能：** 跳转到 `Rm` 中的地址。`Rm` 的 bit[0] 决定目标指令集状态 (0=ARM, 1=Thumb)。
    *   **示例：** `BX LR` ; 常用于函数返回。

4.  **`BLX` (Branch with Link and Exchange)**
    *   **语法：** `BLX{cond} label` 或 `BLX{cond} Rm`
    *   **功能：** 类似 `BL` 和 `BX` 的组合。`BLX label` 通常用于在 ARM 和 Thumb 状态间调用。`BLX Rm` 行为类似 `BX Rm` 但保存返回地址到 LR。

### 3.3.11 饱和运算指令 (Saturating Arithmetic Instructions)

Cortex-M4 等支持 DSP 扩展的内核提供。运算结果如果超出特定范围会被“饱和”到最大值或最小值，而不是溢出回绕。通常会影响 APSR 中的 Q (saturation) 标志位。

*   **`QADD{cond} Rd, Rm, Rn`**: 饱和加法。
*   **`QSUB{cond} Rd, Rm, Rn`**: 饱和减法。
*   **`QDADD{cond} Rd, Rm, Rn`**: 饱和双倍加法 (Rm + 2*Rn)。
*   **`QDSUB{cond} Rd, Rm, Rn`**: 饱和双倍减法 (Rm - 2*Rn)。
*   还有 8位/16位版本，如 `QADD16`, `QASX` (饱和加减交换) 等。

### 3.3.12 其他指令 (Other Instructions)

这是一些无法完全归入上述类别的常用指令。

1.  **乘法指令 (Multiply Instructions)**
    *   **`MUL{S}{cond} Rd, Rn, Rm`**: `Rd = Rn * Rm`
    *   **`MLA{S}{cond} Rd, Rn, Rm, Ra`**: `Rd = (Rn * Rm) + Ra` (乘加)
    *   **`MLS{cond} Rd, Rn, Rm, Ra`**: `Rd = Ra - (Rn * Rm)` (乘减，Thumb-2)
    *   **长乘法 (结果为64位):**
        *   `SMULL{cond} RdLo, RdHi, Rn, Rm` (有符号乘法, RdHi:RdLo = Rn * Rm)
        *   `UMULL{cond} RdLo, RdHi, Rn, Rm` (无符号乘法)
        *   `SMLAL{cond} RdLo, RdHi, Rn, Rm` (有符号乘累加, RdHi:RdLo += Rn * Rm)
        *   `UMLAL{cond} RdLo, RdHi, Rn, Rm` (无符号乘累加)

2.  **除法指令 (Division Instructions)** (Cortex-M3 及之后的部分内核支持，如 M4/M7)
    *   **`SDIV{cond} Rd, Rn, Rm`**: `Rd = Rn / Rm` (有符号除法)
    *   **`UDIV{cond} Rd, Rn, Rm`**: `Rd = Rn / Rm` (无符号除法)

3.  **`CLZ{cond} Rd, Rm` (Count Leading Zeros)**: 计算 `Rm` 中从最高位开始连续的0的个数，存入 `Rd`。

4.  **`IT{pattern}{cond}` (If-Then)**: Thumb-2 指令，用于后续最多4条指令的条件执行。
    *   `pattern` 最多包含3个字符 `T` (Then, 执行) 或 `E` (Else, 执行反条件)。
    *   **示例：**
        ```armasm
        CMP R0, #0
        ITE EQ      ; If R0 == 0 then
        MOVEQ R1, #1 ; R1 = 1
        ADDNE R1, R2 ; Else R1 = R1 + R2
        ```

### 3.3.13 伪指令 (Pseudo-instructions)

伪指令是汇编器提供的指令，在汇编时会被替换成一条或多条真实的机器指令。

1.  **`LDR Rd, =value` / `LDR Rd, =label`**
    *   **功能：** 加载一个32位立即数或地址到寄存器 `Rd`。
    *   **汇编器转换：**
        *   如果 `value` 可以用 `MOV` 或 `MVN` 的立即数形式表示，则转换成 `MOV` 或 `MVN`。
        *   否则，汇编器会将 `value` 存储在一个称为“文字池 (literal pool)”的内存区域，并生成一条 PC 相对的 `LDR` 指令从文字池加载该值。
    *   **示例：** `LDR R0, =0x12345678`
2.  **`ADR{cond} Rd, label` (Address to Register)**
    *   **功能：** 加载 `label` 的地址到 `Rd`。`label` 必须在当前代码段内，且地址通常是相对于 PC 的。
    *   **汇编器转换：** 通常转换成一条 `ADD Rd, PC, #offset` 或 `SUB Rd, PC, #offset` 指令。
3.  **`NOP` (No Operation)**
    *   **功能：** 空操作，不执行任何实际计算，仅占用一个指令周期 (或几个，取决于流水线)。
    *   **汇编器转换：** 通常转换成一条不产生副作用的指令，如 `MOV R0, R0` (在Thumb中可能是特定的16位编码)。

## 四、关键概念总结

*   **`S` 后缀：** 大部分数据处理指令（算术、逻辑、传送）可以通过添加 `S` 后缀来使其结果影响 APSR 中的 N (负)、Z (零)、C (进位/借位)、V (溢出) 标志位。`CMP`, `CMN`, `TST`, `TEQ` 指令默认就会影响标志位，无需 `S`。
*   **条件执行 ` {cond} `：**
    *   **ARM 状态：** 几乎所有指令都可以条件执行。
    *   **Thumb-1 状态：** 主要有条件分支指令。
    *   **Thumb-2 状态：** 通过 `IT` 指令块实现更灵活的条件执行。
*   **`Operand2`：** ARM 数据处理指令的第二个操作数非常灵活，可以是：
    *   立即数 (`#imm`)
    *   寄存器 (`Rm`)
    *   寄存器移位后的值 (`Rm, <shift> #val` 或 `Rm, <shift> Rs`)

## 五、注意

*   **Cortex-M 系列特性：**
    *   仅支持 Thumb/Thumb-2 指令集，不支持 ARM 指令集。
    *   拥有高效的 NVIC (嵌套向量中断控制器)。
    *   Cortex-M4/M7/M33/M35P/M55/M85 等内核包含 DSP 扩展指令 (SIMD 指令) 和可选的单精度 FPU (浮点单元) 指令。
*   **文档范围：** 本文档主要涵盖了通用的 ARM 核心指令。特定于 DSP 或 FPU 的指令集更为庞大和专业。



## 六、转移指令 (Transfer Instructions)

转移指令用于改变程序正常的顺序执行流程，通过修改程序计数器 (PC, R15) 的值来实现跳转到新的执行地址。它们是实现程序分支、循环、子程序调用和返回等控制结构的基础。

![image-20250521213453883](./imgs/image-20250521213453883.png)

### 1. 标号 (Labels)

在汇编语言中，**标号 (label)** 是一个符号名称，代表一个内存地址。转移指令通常跳转到这些标号所指示的位置。汇编器在汇编过程中会将标号解析为具体的内存地址或相对于当前 PC 的偏移量。

*   **示例 (来自图片)：** `not_e:`, `go_on:`, `again:` 都是标号。

### 2. APSR 标志寄存器与条件码

许多转移指令是**条件执行**的，它们是否跳转取决于 APSR (应用层序状态寄存器) 中的条件标志位。

*   **APSR 标志位：**
    *   **N (Negative):** 运算结果为负时置1。
    *   **Z (Zero):** 运算结果为零时置1。
    *   **C (Carry):**
        *   加法：产生进位时置1。
        *   减法 (包括比较)：无借位时置1 (即 `Rn >= Operand2` 时对于 `Rn-Operand2`)。
        *   移位：最后移出的位。
    *   **V (Overflow):** 有符号运算发生溢出时置1。

*   **标志位的设置：**
    *   数据处理指令（如 `ADD`, `SUB`, `MOV` 等）如果带有 `S` 后缀 (如 `ADDS`, `SUBS`, `MOVS`)，会根据运算结果更新 APSR 标志位。
    *   比较指令 (`CMP`, `CMN`) 和测试指令 (`TST`, `TEQ`) 专门用于更新标志位，它们执行运算但不保存结果。
        *   `CMP R0, R1` 会计算 `R0 - R1` 并更新标志位。
        *   `CMPS R0, R1` (在某些汇编器中 `CMPS` 是 `CMP` 的同义词，因为 `CMP` 总是更新标志位；或者在特定上下文中 `CMPS` 可能指代 Thumb-2 中某些特定版本的比较指令)。图中的 `cmps r0, r1` 就是一个比较操作，用于设置标志位。

*   **用户图片中的问题解答：**
    > "CPU里有一个独立且唯一的标志寄存器APSR 用于判断CPU当前执行的指令的标志位，该标志位与存储数值的寄存器r0, rn无关，只和当前操作得到标志位有关?"

    **回答：** 这个理解基本正确。
    1.  APSR 是一个独立的寄存器，用于存储条件标志位。
    2.  这些标志位的值是由**最近一次影响标志位的指令的运算结果**决定的。
    3.  例如，`CMPS R0, R1` 指令会根据 `R0 - R1` 的结果来设置 APSR 中的 N, Z, C, V 标志。标志位本身与 `R0` 或 `R1` 寄存器之前存储的数值**没有直接的、孤立的关系**，而是与这两个值**参与运算后得到的结果**紧密相关。如果之后没有其他影响标志位的指令执行，APSR 的状态会保持，直到下一个影响标志位的指令执行。

*   **条件码 ` <cond> `：** 转移指令（和其他条件执行指令）使用条件码来测试 APSR 中的标志位组合。

    | 条件码    | 描述 (含义)                                            | 标志位组合  | 反向条件码 |
    | :-------- | :----------------------------------------------------- | :---------- | :--------- |
    | `EQ`      | Equal (相等)                                           | Z=1         | `NE`       |
    | `NE`      | Not Equal (不相等)                                     | Z=0         | `EQ`       |
    | `CS`/`HS` | Carry Set / Unsigned Higher or Same (无符号大于或等于) | C=1         | `CC`/`LO`  |
    | `CC`/`LO` | Carry Clear / Unsigned Lower (无符号小于)              | C=0         | `CS`/`HS`  |
    | `MI`      | Minus / Negative (负)                                  | N=1         | `PL`       |
    | `PL`      | Plus / Positive or Zero (正或零)                       | N=0         | `MI`       |
    | `VS`      | Overflow Set (溢出)                                    | V=1         | `VC`       |
    | `VC`      | Overflow Clear (无溢出)                                | V=0         | `VS`       |
    | `HI`      | Unsigned Higher (无符号大于)                           | C=1 AND Z=0 | `LS`       |
    | `LS`      | Unsigned Lower or Same (无符号小于或等于)              | C=0 OR Z=1  | `HI`       |
    | `GE`      | Signed Greater than or Equal (有符号大于或等于)        | N=V         | `LT`       |
    | `LT`      | Signed Less than (有符号小于)                          | N!=V        | `GE`       |
    | `GT`      | Signed Greater than (有符号大于)                       | Z=0 AND N=V | `LE`       |
    | `LE`      | Signed Less than or Equal (有符号小于或等于)           | Z=1 OR N!=V | `GT`       |
    | `AL`      | Always (无条件，不常用在B指令中，但在IT块中可能用到)   | (任何状态)  |            |

### 3. 无条件跳转 (Unconditional Jumps/Branches)

无条件跳转指令总是会改变程序的执行流程到目标地址。

*   **`B label` (Branch)**
    *   **语法：** `B label`
    *   **功能：** 程序计数器 PC 被设置为 `label` 指向的地址。
    *   **范围：** 跳转范围是有限的，通常是相对于当前 PC 的偏移量。Thumb-1 (16位 `B` 指令) 的跳转范围比 Thumb-2 (32位 `B` 指令) 的要小。
    *   **示例 (来自图片)：** `b go_on`

### 4. 条件跳转 (Conditional Jumps/Branches)

条件跳转指令会检查 APSR 中的条件标志位，只有当指定的条件满足时，才会跳转到目标地址。

*   **`B<cond> label` (Branch if Condition is met)**
    *   **语法：** `B<cond> label` (例如：`BEQ loop_start`, `BNE not_e`)
        *   图片中 `bxx` 的 `xx` 即代表这里的 `<cond>`。
    *   **功能：** 如果 APSR 中的标志位满足 `<cond>` 指定的条件，则 PC 被设置为 `label` 指向的地址。否则，程序继续执行下一条顺序指令。
    *   **示例 (来自图片)：**
        *   `cmps r0, r1` ; 比较 r0 和 r1，设置标志位 (如 Z=1 如果 r0==r1)
        *   `bne not_e`   ; 如果 **N**ot **E**qual (即 Z=0)，则跳转到 `not_e` 标签。

### 5. 子程序调用与返回相关转移指令

这些指令不仅进行跳转，还处理子程序调用的链接（保存返回地址）。

*   **`BL label` (Branch with Link)**
    *   **语法：** `BL label`
    *   **功能：**
        1.  将下一条指令的地址 (即返回地址) 保存到链接寄存器 LR (R14)。
        2.  PC 被设置为 `label` 指向的地址，从而调用子程序。
    *   **用于：** 调用子程序。

*   **`BX Rm` (Branch and Exchange instruction set)**
    *   **语法：** `BX Rm`
    *   **功能：**
        1.  PC 被设置为寄存器 `Rm` 中的地址。
        2.  `Rm` 的最低位 (bit 0) 决定目标指令集状态：
            *   如果 `Rm[0] == 1`，处理器切换到 Thumb 状态执行。
            *   如果 `Rm[0] == 0`，处理器切换到 ARM 状态执行 (仅在支持 ARM 状态的处理器上)。Cortex-M 系列只支持 Thumb 状态，所以 `Rm[0]` 必须为1。
    *   **用于：**
        *   **函数返回：** `BX LR` 是最常见的函数返回方式。LR 中通常保存着调用者设置的返回地址，并且其 bit 0 已被正确设置为1 (对于 Thumb 目标)。
        *   跳转到由函数指针（存储在 `Rm` 中）指定的地址。

*   **`BLX Rm` / `BLX label` (Branch with Link and Exchange instruction set)**
    *   **`BLX Rm`:**
        *   功能：类似于 `BX Rm`，但同时也将下一条指令的地址保存到 LR (R14)。用于通过寄存器中的地址调用子程序并切换状态。
    *   **`BLX label` (Thumb-2 中常见):**
        *   功能：将返回地址存入 LR，并跳转到 `label`。此形式通常用于在 Thumb 状态下调用可能在不同地址范围（甚至不同指令集，如果处理器支持）的子程序，可以进行长跳转。

### 6. 构建判断和循环结构

转移指令是实现高级语言中 `if-else`、`while`、`for` 等控制结构的关键。

*   **判断 (If-Else) 结构 (参考图片 "分支" 示例):**
    ```armasm
    假设我们要实现: if (r0 == r1)
    				{ /* then_code */ } 
    			  else 
    			    { /* else_code */ }
    cmps r0, r1   比较 r0 和 r1, 设置APSR标志
                  如果 r0 == r1, 则 Z=1
                  如果 r0 != r1, 则 Z=0
    bne else_block  如果不相等 (Z=0), 跳转到 else_block
    
    then_block: (r0 == r1)
    			... then_code ...
    			例如图片中的: nop
    b end_if      执行完 then_block 后跳转到 if 结构末尾
    
    else_block:   (r0 != r1)
    			... else_code ...
    例如图片中的: mov r0, r1
    
    end_if:
    ... 后续代码 ...
    ```
    图片中的示例简化为：
    ```armasm
    cmps r0, r1
    bne not_e     ; If (r0 != r1) goto not_e
    nop           ; /* if equal */ (then part)
    b go_on       ; Skip else part
    not_e:
    mov r0, r1    ; /* not equal */ (else part)
    go_on:
    ...
    ```
    
*   **循环 (Loop) 结构 (参考图片 "循环" 示例):**
    ```armasm
    ; 示例: r1 = 0; r0 = 4; do { r1 += r0; r0--; } while (r0 != 0);
    mov r1, #0
    mov r0, #4
    again:          ; 循环体开始 (标号)
    add r1, r0    ; r1 = r1 + r0
    subs r0, #1   ; r0 = r0 - 1, 并且更新APSR标志位
                  ; 如果 r0 变为 0, 则 Z=1
    bne again     ; 如果 r0 不等于 0 (即 Z=0), 则跳转回 again
                  ; (图片中注释 "while( CPU里的APSR标志寄存器存储的标志位里的ZF == 1)" 是错误的，
                  ;  BNE 的条件是 Z=0，即 Not Equal。如果想在 ZF==1 时跳转，应该用 BEQ)
    
    ; 循环结束，r0 为 0
    ```

### 7. Thumb-2 `IT` 指令与条件执行

虽然图片中提到 "任何指令都可附加条件来执行" (如 `moveq`)，这在传统的 ARM 指令集 (A32) 中更为直接。在 Cortex-M 系列主要使用的 Thumb-2 指令集中，这种通用的条件执行是通过 `IT` (If-Then[-Else]) 指令实现的，它允许其后的最多4条指令根据指定的条件和模式进行条件执行。

*   **`IT{pattern} <cond>`**
    *   `<cond>`: 基本条件。
    *   `{pattern}`: 由最多三个 `T` (Then) 或 `E` (Else) 组成，指定后续指令如何响应 `<cond>`。
        *   `T`: 如果 `<cond>` 为真，则执行。
        *   `E`: 如果 `<cond>` 为假 (或 `<cond>` 的反条件为真)，则执行。

*   **示例 (`moveq` 的 Thumb-2 实现方式):**
    ```armasm
    CMP R0, R1    ; 比较 R0 和 R1
    IT EQ         ; If-Then, 条件为 EQ (Equal)
    MOVEQ R2, R0  ; 如果 R0 == R1, 则 R2 = R0
    ```
    这里的 `MOVEQ` 是在 `IT` 指令块的上下文中条件执行的。

总结来说，转移指令通过直接或有条件地修改 PC，为程序提供了灵活的控制流能力，是现代处理器执行复杂逻辑的基础。理解它们如何与 APSR 标志位交互至关重要。