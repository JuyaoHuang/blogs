---
title: VUE框架
published: 2025-10-09
author: Alen
description: 'vue框架基础知识和架构理解'
draft: false
first_level_category: "Web全栈开发"
second_level_category: "前端技术"
tags: ['vue']
---
# VUE基础知识

## 基础架构

```
my-vue3-project/
├── .vscode/             # VS Code 编辑器特定设置 (可选)
├── node_modules/        # 项目依赖包 (由 npm/yarn/pnpm 管理)
├── public/              # 静态资源目录 (不会被 Vite 处理)
│   └── vite.svg         # 示例图标 (可能会有 favicon.ico 等)
├── src/                 # 项目源代码目录 (核心开发区域)
│   ├── assets/          # 存放会被 Vite 处理的资源 (如图片, 字体, CSS的一部分)
│   │   └── vue.svg      # 示例 Vue Logo
│   ├── components/      # 存放可复用的 Vue 组件
│   │   └── HelloWorld.vue # 示例组件 (你的截图里没展开，但通常会有一个)
│   ├── App.vue          # 根 Vue 组件
│   ├── main.js          # **应用程序入口文件**
│   └── style.css        # 全局 CSS 样式文件
├── .gitignore           # Git 忽略文件配置
├── index.html           # **SPA 的主 HTML 文件** (Vite 的入口)
├── package-lock.json    # 锁定依赖版本 (npm 生成)
├── package.json         # 项目配置文件 (依赖、脚本等)
├── README.md            # 项目说明文件
└── vite.config.js       # Vite 配置文件
```

### 各部分文件/文件夹的作用及关联:

1. **main.js (位于 src 目录):**

   - **作用:** 	这是整个 Vue 应用的 **入口点**。它负责初始化 Vue 应用实例并将其挂载到 HTML 页面上。

   - **代码解析:**
     - import { createApp } from 'vue': 
     
       ​	从 vue 包中导入 createApp 函数，这是创建 Vue 应用实例的核心方法。
     
     - import './style.css': 
     
       ​	导入全局 CSS 样式文件。Vite 会处理这个导入，并将样式应用到页面上。
     
     - import App from './App.vue': 
     
       ​	导入根组件 App.vue。这个 .vue 文件包含了应用的顶层模板、逻辑和样式。
     
     - createApp(App).mount('#app'):
       - createApp(App): 
       
         ​	使用导入的 App 组件作为根组件，创建一个 Vue 应用实例。
       
       - .mount('#app'): 
       
         ​	将这个 Vue 应用实例挂载（渲染并控制）到 index.html 文件中 id 为 app 的那个 DOM 元素上。
     
   - **关联:** 

     ​	main.js 是连接 index.html 和 Vue 组件 (App.vue 及其他所有组件) 的桥梁。

2. **index.html (位于项目根目录):**

   - **作用:** 

     ​	这是单页应用 (SPA) 的 **主 HTML 页面**。在 Vite 中，它也是开发的入口。浏览器首先加载这个文件。

   - **关键内容 (通常包含):**
     
     - `<div id="app"></div>`: 
     
       ​	这个 div 是 Vue 应用将被挂载的目标容器。main.js 中的 mount('#app') 就是寻找这个元素。
     
     - `<script type="module" src="/src/main.js"></script>`: 
     
       ​	**这是最关键的连接点**。它告诉浏览器以 ES Module 的方式加载并执行 src/main.js 文件。Vite 开发服务器会拦截这个请求，并按需提供 main.js 及其依赖。
     
   - **关联:** 

     ​	index.html 提供了 Vue 应用挂载的 DOM 节点 (#app)，并通过 <script> 标签加载并启动了 main.js。

3. **App.vue (位于 src 目录):**

   - **作用:** 

     ​	这是应用的 **根组件**。所有其他的页面视图或组件通常都会嵌套在 App.vue 内部（直接或间接通过路由）。它通常包含应用的整体布局结构，比如导航栏、页脚，以及一个用于显示当前页面内容的区域（如果使用 Vue Router，则是` <router-view>`）。

   - **结构:** 

     ​	是一个标准的单文件组件 (SFC)，包含 `<template>` (HTML结构)、`<script setup>` (组件逻辑，使用 Composition API) 和` <style scoped> `(组件样式)。

   - **关联:** 

     ​	被 main.js 导入并作为 createApp 的参数，成为整个应用的起点。它内部可以导入并使用 src/components/ 目录下的其他组件。

4. **src/ 目录:**

   - **作用:** 

     ​	存放项目的所有 **源代码**。

   - **assets/:** 

     ​	存放会被构建工具 (Vite) 处理的静态资源。例如，在 CSS 或 Vue 组件中 import 的图片。Vite 会对这些资源进行优化（如路径处理、哈希文件名等）。

   - **components/:** 

     ​	存放可复用的 UI 组件，比如按钮、弹窗、卡片等。这些组件可以在 App.vue 或其他组件中导入和使用。

   - **style.css:** 

     ​	存放全局应用的 CSS 样式。被 main.js 导入后，其样式会作用于整个应用。

5. **public/ 目录:**

   - **作用:** 
      存放 **纯静态资源**。这里的文件**不会**被 Vite 的构建流程处理，它们会按原样复制到最终构建输出目录 (dist) 的根目录下。
   - **适用场景:** 
      放置必须保留原始文件名或路径的资源（如 robots.txt），或者那些完全不需要 Vite 处理的旧库或图片。
   - **关联:** 
      在代码中引用 public 目录下的资源时，需要使用绝对路径（例如，在 index.html 或 CSS 中用 /my-image.png 来引用 public/my-image.png）。

1. **vite.config.js(mjs):**

   - **作用:**
         Vite 构建工具的 **配置文件**。你可以在这里配置开发服务器、构建选项、插件等。

   - **关键内容 (初始项目通常包含):**

     ```html
     import { defineConfig } from 'vite'
     import vue from '@vitejs/plugin-vue' // 导入 Vue 插件
     
     // https://vitejs.dev/config/
     export default defineConfig({
       plugins: [vue()], // 启用 Vue 插件，让 Vite 能处理 .vue 文件
     })
     ```

   - **关联:** 
      控制 Vite 的行为，包括它如何处理 .vue 文件、如何运行开发服务器、如何构建项目等。

2. **package.json:**

   - **作用:** 
      Node.js 项目的 **清单文件**。定义了项目名称、版本、依赖项 (dependencies 如 vue，devDependencies 如 vite, @vitejs/plugin-vue) 以及可运行的脚本 (scripts 如 dev, build, preview)。
   - **关联:** 
      npm install 根据它来安装 node_modules；npm run dev 等命令根据 scripts 部分来执行相应的 Vite 命令。

**总结一下工作流程 (开发时 npm run dev):**

1. 浏览器访问 Vite 启动的本地开发服务器地址。

2. Vite 提供 index.html 文件。

3. 浏览器解析 index.html，发现 `<script type="module" src="/src/main.js">`。

4. 浏览器请求 /src/main.js。

5. Vite 拦截请求，按需编译和提供 main.js 及其依赖（如 vue, ./style.css, ./App.vue）。Vite 的 HMR (热模块替换) 机制使得修改代码时，只更新变化的模块，速度极快。

6. main.js 执行，创建 Vue 应用实例，并将其挂载到 index.html 的 #app div 上。

7. App.vue 组件及其子组件被渲染到页面上。


### index.html和App.vue的联系

1. **index.html：舞台 / 容器 / 应用程序的“外壳”**
   - 浏览器 **直接加载** 的唯一 HTML 文件。当你访问你的 Vue 应用的 URL 时，服务器首先发送的就是这个 index.html。
   - 提供一个 **基本的 HTML 结构**，包括 `<html>, <head>, <body> `标签。
   - 最关键的部分是`<body> `里的一个 **占位符 DOM 元素**，通常是 `<div id="app"></div>`。这个 div 本身是空的，它就像一个预留好的空舞台。
   - 包含一个 `<script>` 标签，用于 **加载并执行 main.js** (`<script type="module" src="/src/main.js"></script>`)。这是启动 Vue 应用的入口。
2. **App.vue：第一个演员 / 应用程序的核心内容 / 根组件**
   - 一个 Vue **组件**，是整个 Vue 应用的 **根组件**。
   - 定义了应用的最顶层的 **UI 结构 (template)**、**行为 (script)** 和 **样式 (style)**。
   - 可以把它想象成将要登上 index.html 那个空舞台的第一个、也是最主要的“演员”或“场景”。应用内的所有其他视图和组件，最终都会被渲染在 App.vue 的` <template> `内部（通常是通过路由 `<router-view> `或直接嵌套）。
   - 它**不是**由浏览器直接加载或理解的。需要被 Vue（在 main.js 的帮助下）编译和处理。

**它们如何连接起来？**

连接的桥梁是 **main.js**：

1. index.html 通过` <script src="/src/main.js">` 加载并执行 main.js。
2. main.js 导入 vue 的 createApp 函数和根组件 App.vue。
3. main.js 调用 createApp(App)，使用 App.vue 作为蓝图，创建了一个 Vue 应用实例。
4. main.js 调用 .mount('#app') 方法。这个方法告诉 Vue 应用实例：“去 index.html 文件里找到那个 id 为 app 的 div 元素，然后把你自己（也就是 App.vue 组件渲染后的内容）挂载到那个 div 里面去。”

**总结:**

- index.html 是静态的 **HTML 容器**，是浏览器看到的入口，它提供了一个挂载点 (#app) 并启动了 JavaScript (main.js)。
- App.vue 是动态的 **Vue 根组件**，定义了应用的实际内容和结构。
- main.js 是 **粘合剂**，它初始化 Vue 应用，将 App.vue 组件“注入”或“挂载”到 index.html 的 #app 元素中，从而让用户看到动态的应用界面。

所以，index.html 是外壳，App.vue 是内核（或者说是内核的起点），main.js 负责把内核装进外壳里。

### Vue项目里各个文件的运行逻辑

1. index.html是网页主体，首先呈现在浏览器的就是index.html的内容，例如`<head></head>`里面的内容。然后到了 `<body>` 里面的内容。`<script>` 标签用来加载并执行 main.js。`<body>`中通常包含一个空的` <div> `元素，它有一个 id 属性，值为 app (即` <div id="app"></div>`)。这个 div 作为一个 **挂载点 (Mounting Point)** 或 **容器 (Container)**，等待 Vue 应用来填充内容。它本身在初始加载时是空的，所以用户一开始看不到由 Vue 渲染的任何东西。

2. 然后 index.html 通过` <script>` 标签请求 main.js。浏览器发出这个请求时，Vite 开发服务器会拦截它，**实时编译** main.js 及其依赖（包括将 App.vue 编译成 JS 和 CSS），然后将编译后的代码返回给浏览器。浏览器执行 main.js 中的代码：import App from './App.vue' 导入了根组件，createApp(App) 使用这个根组件创建了一个 **Vue 应用实例**。

   ```javascript
   import App from './App.vue'
   createApp(App).mount('#app')
   createApp(App) 是用 App.vue 这个组件作为根组件来创建一个 Vue 应用实例。这个实例代表了整个 Vue 应用。
   ```

   ​	.mount('#app') 这个方法指示 Vue 应用实例：“找到 index.html 中那个 id 为 app 的 DOM 元素，然后将应用实例所代表的、由 App.vue 渲染出的内容渲染到 这个元素里面。” 它不是创建了一个 "id 为 app 的实例"，而是将实例挂载到那个具有特定 ID 的元素上。

   ​	最后，.mount('#app') 指示这个实例去 **查找** index.html 中 id 为 app 的那个` <div>` 元素，并将该实例渲染的 **DOM 内容** 插入到这个 `<div>` 元素**内部**，从而完成挂载。用户这时就能看到 App.vue 定义的界面了。

3. .**vue 文件的本质**：

   ​	App.vue (以及其他 .vue 文件) 是 Vue 的**单文件组件 (SFC)**。它不是一个标准的 HTML 文件，但它允许你在同一个文件中，使用专门的**块**来组织相关代码：`<template> `用于编写 HTML 结构的模板，`<script>` (或 `<script setup>`) 用于编写 JavaScript 逻辑，`<style>` 用于编写 CSS 样式。这种方式将组件的关注点（结构、逻辑、样式）分离在不同的块中，但又将它们**聚合**在同一个文件里，便于开发和维护。这个 .vue 文件**需要被构建工具（如 Vite）**编译成浏览器可以理解的 JavaScript 和 CSS。

4. 且在.vue文件中，你可以像Js一样使用 import导入其他vue文件 

   ------

#### 当html中 id 为 app的 div元素未被使用：

或者说，如果 main.js 中的 .mount('#app') 找不到 index.html 中 id 为 app 的元素，会发生以下情况：

1. **Vue 应用无法渲染到页面:**

   ​	这是最直接的后果。Vue 应用实例（由 createApp(App) 创建）已经准备好了，它知道自己应该渲染成什么样的 DOM 结构，但是它找不到指定的“舞台”（即 id="app" 的元素）来展示自己。因此，你在 App.vue 以及其子组件中定义的任何模板内容都不会出现在浏览器页面上。

2. **浏览器控制台通常会报错或警告:** 

   ​	Vue 在设计时考虑到了这种情况。当 mount 方法接收的选择器无法找到对应的 DOM 元素时，Vue 通常会在浏览器的开发者工具控制台（Console）中打印一条**警告 (Warning)** 或 **错误 (Error)** 信息。 这条信息会明确告诉你挂载失败，因为它找不到目标元素。

   - **可能看到的警告信息类似：** [Vue warn]: Failed to mount app: mount target selector "#app" returned null. （具体的措辞可能因 Vue 版本略有不同）。

3. **main.js 的其他代码可能继续执行:**

   ​	createApp(App) 已经成功执行，Vue 应用实例在内存中是存在的。mount 方法尝试执行但失败了。如果 mount('#app') 后面还有其他 JavaScript 代码，它们可能会继续执行。但是，由于应用的核心 UI 没有渲染出来，这些后续代码的执行往往失去了意义，或者可能会因为依赖于已挂载的 DOM 而出错。

4. **用户看到的是空白或不完整的页面:**

   ​	用户最终在浏览器中看到的将是 index.html 中 **除了** Vue 应用应该渲染的内容之外的所有静态内容。如果` <div id="app"></div>` 是` <body>` 中唯一的主要内容容器，那么用户很可能看到一个几乎空白的页面（可能只有 index.html 中` <head>` 定义的标题和一些静态资源）。

**深究 index.html中 id 为 app 的div元素的实现本质**

**Q：**是mian.js去找？意思是html那个是为app建了个占位元素，如果没有id 为app的挂载（假设此挂载要找的id是a）找这个空位，也只是该占位不呈现到网站上而已，并不影响body里其他的元素么？

**A：**

1. **谁去找**？
   - 是在 main.js 中运行的 Vue 代码去找。更具体地说，是 createApp(App) 创建的那个 Vue 应用实例，在执行 .mount('#app') 这个方法时，Vue 的运行时库 (runtime library) 会使用浏览器提供的功能（比如 document.querySelector('#app')）去 当前加载的 HTML 文档 (也就是 index.html) 中查找 id 为 app 的那个 DOM 元素。
   - 所以，不是 main.js 文件本身去找，而是 main.js **执行时**，里面的 Vue 指令（.mount()）触发了查找动作。
2. **index.html 的角色：**
   - index.html 提供了那个 `<div id="app"></div> `元素。它确实是一个占位符或者说容器。在 Vue 应用接管它之前，它就是 index.html 里的一个普通 HTML 元素。
3. **挂载失败的后果：**
   -  如果 Vue 在执行 .mount('#app') 时找不到 id="app" 的元素（比如你把它写成了`<div id="a"></div>`或者根本没写），那么：
     - Vue 应用的 内容（也就是 App.vue 组件渲染出来的 HTML 结构）无法被插入到 index.html 中。
     - **但是**， index.html 文件中` <div id="app"></div>` 之外 的所有其他 HTML 元素（比如` <body> `里直接写的` <header>...</header> `或 `<footer>...</footer>`，或者其他 `<p>`, `<img>` 等标签）**会照常被浏览器渲染和显示**。
     - **Vue 的挂载失败** **只影响** **它试图控制的那部分 DOM（即 #app 元素及其内部），不会破坏 index.html 中其他独立的部分**。

**总结一下：**

- index.html 提供舞台 (`<div id="app"></div>`) 和其他静态布景。
- main.js 指挥 Vue 这个演员 (createApp(App)) 登台表演 (.mount('#app'))。
- Vue 演员需要找到指定的舞台 (#app) 才能开始表演。
- 如果找不到指定的舞台，Vue 演员就无法登台（应用内容不显示），并且会通知后台（控制台报错/警告）。
- 但 index.html 里的其他静态布景（#app 之外的元素）不受影响，它们从一开始就在那里，会正常显示。



------

## 项目常用框架

```
head-up-rate-detection-frontend/
├── public/
│   └── favicon.ico         # 网站图标 (示例)
│
├── src/                    # ⭐ 主要源代码目录
│   ├── assets/             # 静态资源 (会被 Vite 处理)
│   │   ├── images/         # 图片 (如登录页的猫猫图)
│   │   ├── base.css        # 全局样式重置
│   │   └── main.css        # 主要全局样式
│   │
│   ├── components/         # 可复用的 UI 组件 (细粒度)
│   │   ├── common/         # (可选) 通用基础组件
│   │   │   └── LoadingSpinner.vue # 加载指示器
│   │   │   └── AlertPutup.vue     # 提示框组件 (替代 alert())
│   │   └── layout/         # (可选) 布局组件
│   │       └── LoginLayout.vue   # 登录页布局 (如果需要)
│   │       └── MainLayout.vue   # 操作/检测页布局 (如果需要)
│   │
│   ├── views/ (或 pages/)   # ⭐ 页面级组件 (对应你的三个网页)
│   │   ├── Login.vue     # 对应 Welcome.html
│   │   ├── Operation.vue # 对应 Operation.html
│   │   └── Detection.vue # 对应 Head-upRate.html
│   │
│   ├── router/             # ⭐ 路由配置
│   │   └── index.js        # 定义路径和视图组件的映射，处理导航
│   │
│   ├── services/ (或 api/)  # ⭐ API 请求服务 (与后端交互)
│   │   ├── login.js         # 登录/认证 API 调用
│   │   ├── classroom.js    # 获取教室列表 API 调用
│   │   └── detection.js    # 上传图片、获取抬头率 API 调用
│   │   └── request.js      # (可选) 封装 axios 或 fetch, 处理基础 URL, Token 等
│   │
│   ├── stores/ (或 store/)  # ⭐ (可选, 但推荐) 状态管理 (Pinia)
│   │   └── user-status.js         # 存储用户登录状态/信息
│   │   └── classroom.js    # (可选) 存储教室列表或选中的教室
│   │
│   ├── App.vue             # ⭐ 根组件, 通常包含 <router-view>
│   └── main.js             # ⭐ 应用入口, 初始化 Vue, Router, Pinia 等
│
├── .gitignore              # Git 忽略配置
├── index.html              # ⭐ SPA 的 HTML 入口文件 (唯一的 HTML)
├── package.json            # 项目依赖和脚本
├── vite.config.js          # Vite 构建配置
└── README.md               # 项目说明
```

​	Vue3项目的根目录主要用来放项目的**配置文件**（如 package.json, vite.config.ts, tsconfig.json, .gitignore 等）、说明文件 (README.md) 以及一些顶层目录（如 public, src, node_modules）。如果要编写代码，应该在**src**目录(**source/源代码**)下编写.vue文件

​	更具体地说，根据 .vue 文件的作用，你应该把它放在 src 目录下的不同子目录中：

1. ##### **主要页面/视图 (Views/Pages):**

   - **位置:** src/views/ 目录下。

   - **作用:** 
      
      ​	代表应用中的不同“页面”，比如主页、关于页面、用户资料页面等。这些视图通常会组合使用 src/components 目录下的可复用组件。
      
   - **示例:** 

      ​	可能需要创建  **src/views/HomeView.vue 和 src/views/AboutView.vue**。然后需要在 src/router/index.ts (或 .js) 文件中配置路由，将 URL 路径（如 / 和 /about）映射到这些视图组件。

2. **可复用的小组件 (Components):**

   - **位置:** src/components/ 目录下。

   - **作用:** 
      
      ​	创建可以在多个地方重复使用的 UI 元素，比如按钮、卡片、表单输入框、导航栏、页脚等。
      
   - **示例:** 
      
      ​	创建一个`src/components/BaseButton.vue` 文件来定义一个自定义按钮样式和行为，然后在视图 (views) 或其他组件中导入并使用它。图中 App.vue 使用的 `<HelloWorld msg="You did it!" />`就是一个例子，它的代码应该在 src/components/HelloWorld.vue 文件里。

3. **应用程序根组件:**

   - **位置:** src/App.vue
   - **作用:** 
      这是整个 Vue 应用的根组件。它通常包含：
     - **整体布局:** 比如通用的页头 (Header)、页脚 (Footer)、侧边栏(Sidebar)。
     - **路由出口:** `<RouterView />` 组件，它是一个占位符，用来显示当前路由匹配到的视图 (views) 组件。
     - **全局元素:** 可能有一些全局性的、不随路由变化的元素。
   - **何时修改:** 
      通常会修改  App.vue 来调整整体布局，或者添加/移除全局  UI 元素。但具体页面的内容和可复用的功能块不应该直接写在这里，而应该写在 views 和 components 中。

4. **路由配置 (Routing):**

   - **位置:** src/router/index.ts (或 .js)
   - **作用:** 
      定义应用程序有哪些 URL 路径，以及每个路径应该显示哪个视图 (views) 组件。

5. **状态管理 (State Management):**

   - **位置:** src/stores/ 目录下 (如果使用了像 Pinia 这样的状态管理库)。
   - **作用:** 
      管理整个应用共享的数据状态，比如用户登录信息、购物车内容等。

6. **主入口文件:**

   - **位置:** src/main.ts (或 .js)。
   - **作用:** 
      初始化 Vue 应用，挂载根组件 (App.vue) 到 index.html 中的某个 DOM 元素上，并注册插件（如 Router, Pinia）

7. **静态资源:**

   - **位置:** **src/assets/** 目录下。
   - **作用:** 
      存放会被构建工具（如 Vite/Webpack）处理的静态文件，比如图片、全局 CSS 文件、字体等。在 .vue 文件中可以通过相对路径或别名 (@/assets/...) 引用。
   - **public/ 目录:** 
      这个目录下的文件会原样复制到最终构建输出的根目录，适合放不需要构建处理的文件，比如 favicon.ico。

**总结一下，主要编码工作通常在：**

- **src/components/:** 编写可复用的 UI 组件。
- **src/views/:** 编写代表应用页面的组件。
- **src/router/index.ts:** 配置页面路由。
- **src/stores/:** 管理全局状态 (如果需要)。
- **src/App.vue:** 调整应用根布局和全局元素。

​	当然，这只是通用架构，如果你有更多需求，自然可以自己拓展新的分层架构。例如API层处理（解码）后端的api请求，services层处理具体的业务请求，例如需要向后端用某种方法（get、post、put、delete）发送某一api请求，并得到响应的json数据。

## 安装VUE框架

###     配置环境：

1. 在VScode中打开集成终端 Ctrl+`

2. 用npm指令创建Vue3项目（Vite方式）：

   ```cmd
   npm create vue@latest
   ```

3. 相关组件安装：

   1.  **TypeScript**

      - **核心作用**: 
            为代码提供**类型安全**。它要求在定义变量、函数参数和返回值时，明确指出它们应该是 string、number、boolean 还是其他更复杂的类型。
      - **什么时候应该选择它**：
            如果不喜欢TS的类型检查，也可以不选。尤其当项目会变得复杂，或者需要长期维护时，TypeScript 能在早期就帮你避免掉无数潜在的类型相关的 Bug。

   2.  **JSX 支持 (JSX Support)**

        - **核心作用**: 
            	允许在 JavaScript/TypeScript 文件中，使用一种**类似 HTML 的语法**来编写组件的结构。这是 React 框架默认的写法。
            
        - **通俗比喻**: 
            
            ​	Vue 默认的 `<template> `写法，就像是在**画板上（HTML区域）画画**，然后在旁边的调色盘（`<script>`区域）准备颜料。而 JSX 则是允许直接用**颜料（JavaScript）来画画**。
            
        - **什么时候应该选择它**:
          
          1. 如果有很强的 React 背景，非常习惯用 JSX。
          2. 需要编写一个**渲染逻辑极其复杂**的组件时。例如，一个组件需要根据 5、6 个不同的条件，来渲染出完全不同的 HTML 结构。用 v-if, v-else-if, v-else 可能会变得很臃肿，而用纯粹的 JavaScript if/else 或 switch 逻辑配合 JSX 可能会更清晰。

   3.  **Vue Router (用于单页面应用开发)**

        - **核心作用**: 
            
            ​	在网站内实现“**前端路由**”，管理 URL 和页面视图的对应关系，而**不需要每次都向服务器请求一个全新的页面**。
            
        - **通俗比喻**: 
            Vue Router 就像是网站的**“内部邮递员”或“交通指挥”**。
          
          - 当用户点击一个指向 /about 的链接时，Router 会拦下这个请求，阻止浏览器进行整页刷新。
          - 然后，它会悄悄地把 URL 地址改成 yourdomain.com/about，并把页面上主要的内容区域替换成 About.vue 这个组件。
          - 这使得页面切换感觉非常快、非常流畅，就像一个桌面应用，因此被称为**单页面应用 (SPA)**。
          
        - **什么时候应该选择它**:    只要网站包含多个“页面”（如首页、关于我们、产品列表），就几乎必须选择“是”。

   4.  **Pinia (用于状态管理)**

        - **核心作用**: 
            
            ​	提供一个“全局状态”的存储中心，让应用中任何一个组件，都能方便地读取或修改共享的数据。
            
        - **通俗比喻**:
            
            Pinia 就像是应用里的一个“中央公告板”或“共享仓库”。
          
          - 假设有一个“用户登录状态”信息，Header 组件需要根据它显示“登录/注销”，Profile 页面需要显示用户名，Settings 页面需要用户 ID。
          - 如果没有 Pinia，可能需要把这个用户信息从最顶层的 App.vue 组件，一层一层地传递给这些子组件（这个过程被称为“Prop Drilling”，非常繁琐）。
            - 有了 Pinia，任何组件都可以直接去“中央公告板”上查看或更新这个用户信息，无需层层传递。
            
        - **什么时候应该选择它**:

          1. 多个**没有直接父子关系**的组件需要共享同一份数据时。
          2. 厌倦了将一个 Prop (属性) 传递超过两层组件时。

          - 对于中型到大型项目，Pinia 几乎是必需品。

   5.  **Vitest (用于单元测试)**

        - **核心作用**: 

          ​	为代码提供一个**单元测试**框架。单元测试是指对代码中最小的可测试单元（比如一个独立的函数、一个独立的 Vue 组件）进行隔离测试。

        - **通俗比喻**: 

          ​	Vitest 就像一个**“零件质检员”**。它会把一个函数（比如一个计算价格的函数）拿到一个隔离的测试台上，给它输入各种预设的参数，然后检查它的返回值是否和预期完全一样。

        - **什么时候应该选择它**: 

          ​	希望构建一个高质量、可靠、可长期维护的应用时。编写单元测试可以确保代码的每一个小“零件”都是正常工作的，并且未来修改代码时，能立刻发现是否不小心弄坏了其他东西。

   6.  **端到端测试 (End-to-End Testing)**

        - **核心作用**: 

          ​	提供一个**端到端（E2E）测试**的方案。它会模拟一个真实用户，在真实的浏览器环境中，对整个应用进行操作，以测试完整的用户流程。

        - **通俗比喻**: 如果说 Vitest 是“零件质检员”，那么 E2E 测试就是**“整车试驾员”**。

          ​	它会启动一个由程序控制的真实浏览器，自动访问网站，模拟点击登录按钮、输入用户名密码、跳转到用户中心、再点击退出... 整个流程走一遍，来验证所有“零件”组合在一起后，这辆“整车”是否能正常驾驶。

        - **什么时候应该选择它**: 

          ​	对于非常重要的、业务逻辑复杂的核心流程（如注册、登录、支付），E2E 测试是保证应用整体功能正常的最后一道防线。

   7.  **ESLint (用于代码检查和错误预防)**

        - **核心作用**: 

          ​	在编写代码时，实时地**检查代码是否符合一系列预设的“语法和风格规则”**，从而在早期就发现潜在的逻辑错误和不规范的写法。

        - **什么时候应该选择它**: 永远选择“是”。它是保证代码质量和团队协作规范性的基石。

        **Prettier (用于代码格式化)**

        - **核心作用**: 

          ​	自动地、强制地**统一整个项目的代码格式**（比如缩进用几个空格、要不要加分号、单引号还是双引号等）。

4. 进入项目根目录并安装依赖包：

   注意：不是当前目录，应该是此目录的下一级。

   ```cmd
   cd your-vue-project #进入项目根目录
   npm install  # 安装依赖
   ```

   ### 运行项目

   1. 运行Vue开发服务器命令(即运行Vue3项目)：

      ```bash
      npm run dev # 或者使用yarn命令：yarn dev
      ```

      运行后终端应该输出：

      ```bash
      ➜  Local:   http://localhost:5173/
      ➜  Network: use --host to expose
      ➜  press h to show help
      ```

   2. 复制那个 Local: 后面的地址 (例如 http://localhost:5173/)，然后在你的网页浏览器（如 Chrome, Firefox, Edge）中打开这个地址。

   3. **查看运行的应用:** 你应该就能看到你的 Vue 应用在浏览器中运行起来了！并且当你修改代码并保存时，页面会自动更新（HMR - 热模块替换）。

   4. 调试Vue应用：

      调试 Vue 应用通常在**浏览器**的开发者工具中进行，而不是直接在 VS Code 像调试 Node.js 或 C++ 那样（虽然有方法可以配置 VS Code 调试，但浏览器调试更常用直接）：

      1. 浏览器开发者工具 (F12):

         - **Console**：查看 console.log 输出和运行时错误。

         - **Sources**: 可以找到你的组件代码（可能在 localhost:xxxx/src/ 下），并在 `<script>` 部分设置断点来调试 JavaScript 逻辑。

         - **Network:** 查看网络请求。

      2. **Vue Devtools**

         - 为 Chrome、Firefox、Edge 安装 **Vue Devtools** 扩展。
         - 安装后，在你的 Vue 应用页面打开开发者工具 (F12)，会看到一个新的 "Vue" 选项卡。
         - **功能:** 检查组件层级、查看和修改组件的 props 和 data (或 ref/reactive 状态)、追踪事件、检查路由信息、检查 Pinia 状态等。这是调试 Vue 应用的神器！
