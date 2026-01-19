---
title: 'Draft Page'
publishDate: '2026-01-19'
updatedDate: '2026-01-19'
description: 'Draft page for any purpose'
tags:
  - Draft
language: 'English'
# Remove or set false to turn draft page into normal ones
draft: true
first_level_category: 'project'
second_level_category: 'project'
---

```log title="hello.log"
test
test
```

> Test

Test `inline code`

| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Data 1   | Data 2   |
| Row 2    | Data 1   | Data 2   |
| Row 3    | Data 1   | Data 2   |

test 1:
```mermaid
graph TD;
  A[pnpm] --> B[安装成功];
```
-----

test 2:
```mermaid
graph TD;
    Start((Start)) --> Init[Initialize System];
    Init --> Check{Is DB Ready?};
    Check -- Yes --> Process[Process Data];
    Check -- No --> Error[Log Error];
    Process --> Finish((End));
    Error --> Finish;
```

-----
test 3:
```mermaid
sequenceDiagram
    participant User
    participant System
    participant Database

    User->>System: Login Request
    System->>Database: Query User Data
    Database-->>System: Return User Profile
    System-->>User: Login Success
    
    Note over User,System: This is a secure connection
```

-----
test 4:饼图

```mermaid
pie
    title Coding Time Distribution
    "Writing Code" : 40
    "Debugging" : 30
    "Reading Docs" : 20
    "Coffee Break" : 10
```
-----
test 5:柱状图

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Running : Event Start
    Running --> Paused : Event Pause
    Paused --> Running : Event Resume
    Running --> [*] : Event Finish
```

-----
test 6:类图

```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +eat()
        +sleep()
    }
    class Dog {
        +bark()
    }
    class Cat {
        +meow()
    }
    Animal <|-- Dog
    Animal <|-- Cat
```

