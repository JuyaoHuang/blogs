---
title: Tensor基础
author: Alen
published: 2025-10-26
description: "Tensor的基础语法和数据操作"
first_level_category: "AI"
second_level_category: "PyTorch"
tags: ['python']
draft: false
---

# Tensor

在 PyTorch 中，**张量 (Tensor)** 是进行所有计算的核心。

你可以把它看作是 NumPy ndarray 的一个功能更强大的替代品。它不仅提供了类似 NumPy 的多维数组操作功能，还带来了两个关键特性：

1. **GPU 加速**：张量可以轻松地在 CPU 和 GPU 之间转移，从而利用 GPU 强大的并行计算能力来加速运算
2. **自动求导**：张量能够自动追踪其计算历史，用于高效地计算梯度，这是深度学习模型训练的核心
