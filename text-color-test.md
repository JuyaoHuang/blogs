---
title: 'Test: Colored Text Rendering'
description: 'Testing colored text syntax'
publishDate: 2026-04-11
tags: ['test']
first_level_category: "项目实践"
second_level_category: "Demo与示例"
draft: false
---

## {red}Basic Colors{/red}

{red}这段文字是红色的{/red}

{blue}This text is blue{/blue}

{green}This text is green{/green}

{yel}This text is yellow{/yel}

{pur}This text is purple{/pur}

{gray}This text is gray{/gray}

{pink}这段文字是粉红色的{/pink}

{emer}This text is emerald{/emer}

{rose}This text is rose{/rose}

{vio}This text is violet{/vio}

## Mixed with Markdown

**{red}Bold red text{/red}**

{red}**Bold red text**{/red}

{blue}*Italic blue text*{/blue}

*{blue}Italic blue text{/blue}*

{green}[A green link](https://example.com){/green}

{pur}**Bold** and *italic* purple{/pur}


{red}$E = mc^2${/red} 

## Adjacent Colors

{red}Red{/red} normal text {blue}Blue{/blue} more text {green}Green{/green}

## Inline Usage

This is normal text with {pink}pink words{/pink} in the middle of a sentence.

## Should NOT be affected

`{red}code block stays raw{/red}`

~~GFM strikethrough still works~~

$E = mc^2$ (math still works)
