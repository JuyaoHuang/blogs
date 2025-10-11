---
title: python语法合集-函数
author: Alen
published: 2025-10-10
description: "Python合集第三集：函数的基本语法"
first_level_category: "python"
second_level_category: "基础语法"
tags: ['python']
draft: false
---

# Python合集第三集：函数的基本语法

## 函数：

 定义：提供已经封装好的一个实现了特定功能代码片段的接口，该代码段可重复使用
 使用 def 去定义函数
 例子1：使用自己编写的函数，实现对字符串长度的统计

```python
import string
def LEN(data):
    res = 0
    for i in data:
        res += 1
    print(f"字符串{data}的长度是{res}")

str = "dgbah"
LEN(str)
```

## 函数基础语法：

### 函数一般形式

```py
def 函数名(传参):
 	函数体
 return 返回值 //若返回值不需要，可以省略
```

### 函数传参

**函数传入参数的作用： 即函数调用时，使用外部提供的数据(参数)** 

例如：

```python
def ADD(x,y):# x,y是形参
    return x + y
ADD(5,6)# 5,6是传入的实参
```

该函数实现了任意量 x y 的求和，而非两个特定值（常量）
其中 ADD(x,y) 中的 x,y 称为形式参数（形参),而调用函数时传入的参数称为实参

例2：

```python
def AAA(data):
    if data <= 37.5:
        return f"欢迎，{data}"
    else:
        return "Damn!"

print(AAA(15),AAA(333))
```

#### 不定长参数

​	不定长参数用于当传入的参数个数不确定时的情况。

**传入形式**

1. 位置传递

   1. 实际上就是使用一个指针，将传入的所有参数作为一个**元组**，指针指向这个元组的头，每有一个参数传进来，该参数就会加入此元组，并且指针指向该参数的下标。

   2. 使用    def a(*var)

   3. 例子：

      ```py
      def tt(*var):
          return var
      
      na,ag,ge = tt(19,"mark","a")
      print(f"{na},{ag},{ge}")
      # 19,mark,a
      ```

2. 关键字传递

   1. 使用    def a（**var）

   2. 传递的参数为 **键=值**  的形式，此时所有的 键-值对 会被`var`接收，并且根据接收的键值对形成**字典**，就是说，`var` 是一个字典指针。

   3. 例子：

      ```py
      def tt(**var):
          return var
      
      a = tt(age = 19,name = "mark",bb = "a")
      print(f"{a}")
      # {'age': 19, 'name': 'mark', 'bb': 'a'}
      ```

​		

#### 位置传参

​	调用函数时根据函数定义的参数位置传参。

例如：

```py
def tt(name,age,gender):
    return name,age,gender

na,ag,ge = tt("alen",1,"男")
print(f"{na},{ag},{ge}")
# alen,1,男
```

**注意**：传递的参数要和定义的参数的个数、顺序一致。

#### 关键字传参

​	函数调用时使用 **键 = 值** 的形式传参。这样就可以避免位置传参中的`顺序`要求。

```py
def tt(name,age,gender):
    return name,age,gender

na,ag,ge = tt(age=19,name="mark",gender="q")
print(f"{na},{ag},{ge}")
# mark,19,q
```

​	函数调用时，若有位置参数要传，则先传位置参数，再传关键字参数。

#### 默认参数

```py
def tt(name,age,gender="女"):
    return name,age,gender

na,ag,ge = tt(age=19,name="mark")
print(f"{na},{ag},{ge}")
# mark,19,女
```

### 返回值 return

#### 单个返回值

 返回值：函数体执行完毕后"应该"返回一个值，告诉调用体函数体已经执行完毕。

一般可以用一个变量去存储这个返回值

 **特殊返回值：None**
 **若函数没有写 return 语句，等价于 return None，表示函数体返回值为None，即"空"。**
l例如

```python
def a(data):
    data
def b(data)
	return data+1
print(a(1),type(a(1))) # 结果：None 且None 在判断语句中等价于 false
c = b(2)
print(c) # c = 3
```

#### 多个返回值

```
return a,b,c
```

就是用一个逗号把返回变量隔开，并且支持不同类型的数据



## 函数说明文档：

为了给他人或自己看自己写的函数具体是实现什么功能的，可用多行注释进行函数说明
范式：//可以参考标准库里别人的注释文档

```py
 def func(x,y):
 """
 函数说明//实现了什么功能
 :param x: 形参x的说明
 :param y: 形参y的说明
 :return: 返回值的说明
 """
 函数体
 return 返回值
```

## 函数的嵌套调用：

即在函数里调用了另一个函数，例如递归中对自身的调用
例如：斐波那契数列

```python
def digui(val):
    if val == 1 or val == 2:
        return 1
    else:
        return digui(val-1) + digui(val-2)

for i in range(1,6):
    print(digui(i))
```

## 变量的作用域：

### 局部变量和全局变量

局部变量作用于函数体内部的变量，通常在函数体内定义，例如函数体的形式参数
全局变量：顾名思义，作用于整个.py文件的变量。
例如：

```python
num = 100
def a(data):
    global num
    return data
a(1)
```

num就是全局变量，data为局部变量
若想将data变为全局变量，可使用关键字 "**global**" 将其变为全局变量 ---- **不推荐这么做**，毕竟你都在函数体内定义此变量了，为什么还要扩大它的作用域。

**注意：**

在 Python 中，**全局变量**确实可以在函数外部直接访问，但**如果你要在函数内部修改（赋值）全局变量的值**，就必须使用 `global` 关键字声明，否则 Python 会把它当作**局部变量**处理。

## 综合案例 ATM取款机

你需要完成一个ATM取款机的编写。

它应该具有以下功能：

1. 查询余额
2. 存款
3. 取款
4. 退出登录				

参考代码:

```py

def check_balance():
    print(f"查询余额:{account_balance}")

def deposit(profit):
    global account_balance
    account_balance += profit

def withdraw(sub_profit):
    global account_balance 
    account_balance -= sub_profit

account_balance = 0
op = 0
op = int(input("请输入您想要的操作:\n 1.查询余额\n 2.存款\n 3.取款\n 4.退出\n"))
while op != 4:
    if op == 1:
        check_balance()
    elif op == 2:
        profit = int(input("请输入存款金额:"))
        deposit(profit)
        print("存款成功")
    elif op == 3:
        sub_profit = int(input("请输入取款金额:"))
        if sub_profit > account_balance:
            print("余额不足！")
        else:
            withdraw(sub_profit)
            print("存款为:")
    elif op == 4:
        print("退出成功!")
    else:
        print("输入错误，请重新输入!")
    op = int(input("请输入您想要的操作:\n 1.查询余额\n 2.存款\n 3.取款\n 4.退出\n"))

```

