---
title: Matthew Prince访谈：互联网历史和按次付费爬取
author: Alen
published: 2025-10-12
description: "Cloudflare创始人兼首席执行官 Matthew Prince 访谈：互联网历史和按次付费爬取"
first_level_category: "资讯与生活"
second_level_category: "前沿快讯"
tags: ['cloudflare']
draft: false
---

# 中文

早上好，

今天的 Stratechery 访谈对象是[Cloudflare](https://www.cloudflare.com/)联合创始人兼首席执行官[Matthew Prince。Prince](https://x.com/eastdakota)进入硅谷的历程引人入胜——我们将在本次访谈中探讨这一点——但他最为人熟知的是 Cloudflare，他于 2009 年在哈佛商学院创立了该公司。Cloudflare 为云端网站提供网络服务，并拥有[科技界最有效、最引人入胜的免费增值商业模式之一](https://stratechery.com/2021/cloudflare-on-the-edge/)。

在本次采访中，我们探讨了 Prince 的背景、Cloudflare 最初的构想以及 Cloudflare 的现状——以及它是如何凭借机遇发展成为如今这样的公司的。Prince 的最新关注点是互联网内容网站的经济效益；他非常担心人工智能对谷歌创立的传统流量商业模式的影响，并正在利用 Cloudflare 的力量尝试打造一种新的内容商业模式。我们探讨了 Prince 的动机和担忧，以及为什么 Prince 认为这是 Cloudflare 权力的合理运用，即使互联网未来的最终决策者是谷歌。

提醒一下，所有 Stratechery 内容（包括访谈）均可作为播客提供；单击此电子邮件顶部的链接将 Stratechery 添加到您的播客播放器。

进入采访：

### Cloudflare 创始人兼首席执行官 Matthew Prince 访谈：互联网历史和按次付费爬取

*本次采访内容经过少量编辑，以保证内容清晰。*

**主题**

[背景](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#background)| [Cloudflare 的理念](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflare-idea)| [Cloudflare 的现状](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflare-today)| [Cloudflare 的利基市场](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflares-niche)|[按次付费爬取](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#pay-per-crawl)| [Cloudflare 的力量](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflares-power)| [Google 的问题](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#google-problem)| [Cloudflare 的动机](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflares-motivation)

#### 背景

**马修·普林斯，欢迎回到 Stratechery。**

**马修·普林斯**：谢谢，本。

**您实际上是 Stratechery 最早的采访对象之一。事实上，我已经和您谈过两次了。然而，两次[采访](https://stratechery.com/2019/8chan-and-el-paso-cloudflare-drops-8chan-an-interview-with-cloudflare-ceo-matthew-prince/)都非常有[针对性](https://stratechery.com/2021/interviews-with-patrick-collison-brad-smith-thomas-kurian-and-matthew-prince-on-moderation-in-infrastructure/)，主要集中在内容审核问题上。而且，这两次采访都是在我发布采访录音之前，也就是 Stratechery 的现代采访模式。因此，我仍然可以针对首次采访的对象提出我惯常的问题：我从未问过您的背景，比如您是如何开始创业的，您是如何对科技产生兴趣的，能不能跟我讲讲最初的情况。**

**MP**：我在犹他州的山区长大，六岁的时候——其实我以前以为是七岁，但我妈妈纠正了我——1980年，我六岁的时候，祖母送给我一台[Apple II Plus](https://en.wikipedia.org/wiki/Apple_II_Plus)作为圣诞礼物，我对它简直如鱼得水。犹他大学有一个世界一流的计算机科学系，我妈妈以前在那里上继续教育课，她会偷偷带我去，说我是个早熟的孩子，她假装做作业，但实际上我做的都是作业。

**哦，她实际上已经报名上课了，而你只是跟着去的？**

**MP**：没错。高中时我做过一些让你很受欢迎的事，比如我参加过电脑夏令营，那里有电脑夏令营，还有凯特，还有圣巴巴拉的寄宿学校。我真希望自己能和那些人保持联系，因为我敢打赌，那些和我一起的人后来都做出了非常了不起的事情。我上大学时也想过要学计算机科学，但当时我怀着18岁年轻人的傲慢，选了计算机科学105之类的课程，结果觉得无聊透顶，因为我已经学了很久了。

我实际上转了专业，学的是英语文学，方向不同，但我仍然对计算机很了解。1992年我上大学，当时互联网正处于蓬勃发展时期，学校需要懂点计算机的学生。我当时是学生网络管理员之一，就这样我学会了交换机、路由器以及各种设备的工作原理。我宿舍有一条光纤线路直通学校的路由器，最初几年的网速比之后很多年都快。

**哦，是的。我之前说过，威斯康星州的人通常只在宿舍住一年，我和我的一些朋友住了两年，因为那里的网速更快。但我们没有光纤线路，只有T2之类的。**

**MP**：是的，就是这样。大学毕业时我收到了一份工作邀请，说实话我当时根本不知道是什么，但那是一个产品经理，我当时就想，“这到底是什么意思？”

**我认为现在是 2025 年，但仍然没有人知道这意味着什么。**

**MP**：而且不得不再次重塑自我。但我记得我收到过微软、雅虎和网景的录用通知，我说：“不，我不会那样做”，所以我决定去读法学院。结果我去了法学院，还以为自己会当一段时间的律师。要不是互联网泡沫破灭，我可能还会当律师。

**这可不是你想象中的答案。你可能会想：“互联网泡沫破灭让我离开了科技行业。” 互联网泡沫破灭怎么就把你从法律行业赶出来，让你进入科技行业了呢？这听起来好像有点反了。**

**MP**：嗯，这多少有点让我对科技行业产生了兴趣。我发现我喜欢的法律类型是证券法，也就是公司上市。所以1999年夏天，我在旧金山一家大型律师事务所实习，一个夏天就帮助六家公司上市，这感觉太棒了，非常有趣。我的计划是先去律师事务所工作一段时间，然后找到一家我认为很棒的公司，帮他们融资之类的。

**是的，成为他们的 GC 或类似的东西。**

**MP**：我最终还是去了公司法务部门，我以为我的职业道路就是这样的。后来，2000年3月互联网泡沫破灭了，律所打来电话说：“嘿，好消息，坏消息。好消息，你还有工作。坏消息，我们不需要证券律师了。但破产法基本上和破产法一样，我们认为你能搞定。” 作为一名律师，我当时的反应是：“这和破产法差不多。”

**少了很多乐趣。**

**MP**：公司破产后，内部人员就不剩了。

有个叫道格·利希特曼的年轻法学教授，他经常来我的公寓，我们一起喝葡萄酒。他说：“我哥哥在保险行业开了一家B2B公司，他们想找一个和你技能相符的人，你有兴趣吗？”我说：“嗯，听起来很棒。”他们或许会给你匹配的薪水，还会给你股票。我说：“当然可以。”于是我就这么做了，我们融资了大约600万美元。

它的商业模式几乎和今天的Rippling一样，但远远超越了它的时代。我们当时太蠢了，用尽各种办法把它搞砸了，所以在大约18个月的时间里，我们浪费了600万美元，这是一个巨大的失败。但看到一群人能够聚在一起，带着纸上的想法，努力创造一些东西，最终失败，而且是光荣地失败，这真是太不可思议了。

再说一次，我觉得时机不对，而且没人进监狱，这真是太棒了。我们亏光了投资者的钱，投资者们说：“是啊，真糟糕，如果你们有什么新东西，一定要告诉我们。”我当时就想：“哇，这真是个我以前不知道的神奇世界。”于是，我花了接下来的八年时间试图回归那种状态，尽管大部分时间都像在荒野中徘徊。我做过调酒师，教过LSAT备考课程，还做过一些零工，勉强维持生计。

**这是[您开始使用 Unspam 的](https://www.unspam.com/about.html)时候，对吗？**

**MP**：是的。

**你[写了一篇关于《反垃圾邮件法案》的论文](https://repository.law.uic.edu/jitpl/vol22/iss1/3/)，我记得你当时在教一门相关的法律课程。为什么你对垃圾邮件这么感兴趣呢？**

**MP**：我觉得这是一个有趣的法律问题。除了极少数例外，在互联网出现之前，很少有办法可以坐在一个地方，在地球的另一端犯罪。以前有一些，比如邮政诈骗之类的，但突然之间，这种犯罪就变成了一件真实存在的事情，而且规模越来越大。

其次，这是第一次，你坐在世界的某个地方，向另一个地方发送电子邮件，可能构成犯罪，甚至浑然不知，因为你没有任何相关的管辖权。所以我认为，如何将司法管辖权的法律应用于网络空间等问题，是一个非常有趣的问题。再说一次，我主要扮演的是法学教授的角色，我是一个兼职教授，是一所已经不存在的法学院的最低级别的教授，情况就是这么糟糕。

**（笑）这是芝加哥伊利诺伊大学还是……？**

**MP**：不是芝加哥大学，芝加哥大学很棒，而是芝加哥的约翰·马歇尔法学院，现在已经不存在了。但它给了我思考和写作的借口，我有一个今天我们称之为博客的东西，但那时“博客”这个词甚至还没有被创造出来。我觉得，就像你写作和谈论你感兴趣的事情一样，那是我人生中一段能够做到这一点的时期。我的父母非常担心我永远无法有所成就，说实话，我也有点担心，但我认为，拥有这样的视角，对于我后来创办Cloudflare来说，真的非常重要。

**那么 Unspam 具体是什么呢？**

**MP**：Unspam 最初是我注册的一个很酷的域名，我当时想，“我能用它做什么？”，我们尝试做的第一件事就是一次性电子邮件地址，就像苹果现在做的那样，你基本上可以拥有一个电子邮件地址，然后如果有人开始向你发送你不想要的东西，你就关闭这个电子邮件地址，它就会消失。

**我从 1990 年代上大学时就开始这样做了，我仍然使用这个域名来管理所有这些电子邮件地址，事实证明，这实际上是一种非常困难的生活方式。**

**MP**：（笑）我也有同样的问题。

**在不同地方使用同一个邮箱地址，这其中有很多预设，所以这真的很难。我现在就被困住了！**

**MP**：是的。嗯，我们应该找个时间互相体谅一下，因为我也有同样的问题。而且，如果你跟别人解释，比如跟Nordstrom的客服人员通话，他们会问：“好的，你的邮箱是？”，我会说：“嗯，是Nordstrom……”

**Nordstrom@domain.com。**

**MP**：然后他们就说：“不，不，不。*你的*邮箱地址是什么？”

**然后他们就会问，“等等，你是我们公司的员工吗？”**

**MP**：然后就说：“这根本说不通。” 是的，我遇到过很多这样的问题。我们就是这么开始的。后来，我们接到了“禁止致电”名单的申请，于是问题来了：“你们能不能建立一个类似“禁止致电”名单的东西，但只用于电子邮件？” 就这样，我们最终莫名其妙地成为了一家政府承包商，与各州合作。

最初，我们实际上是在查克·舒默和《反垃圾邮件法案》的推动下开始的，但联邦政府从未实施过。但密歇根州、犹他州和其他几个州实施了一项法案，基本上规定，如果你把你的邮箱地址添加到一个列表中，他们就会维护这个列表。我们所做的就是提供哈希技术，对营销人员的列表和政府的列表进行哈希处理，然后比较两者。所以双方都不知道对方的列表里有什么，这也不是改变世界的技术，但这家公司仍然存在，虽然盈利，但永远不会成为一门真正意义上的大生意。

**你本科学的是英语，后来上了法学院，然后你决定，“哦，我还要读MBA”，于是你去了哈佛商学院。你当时的想法是——“我不知道人生该做什么，所以我要去哈佛？”——很多人都是这么想的。你当时的想法是“我特别想获得一些知识，我想创业？”，考虑到Cloudflare的想法是在剑桥开始的，那么接下来的事件顺序是怎样的呢？**

**MP**：我创办了 Unspam，当时 Unspam 举步维艰，业绩不佳。我爸爸——我们都是中上阶层出身，拥有好几家餐厅——但都不是好餐厅——比如 Applebee's、Famous Dave's Bar-B-Que，他还拥有犹他州唯一的 Hooters 餐厅。

**这本身就是一次冒险。**

**MP**：是的，当时他快70岁了，他打电话给我说：“儿子，我照顾你一辈子，现在你该来管理家族生意了。” 我无法想象还有什么比管理我爸爸的Hooters餐厅更糟糕的事情了。所以，我萌生了去商学院的想法，当时我正喝着第二瓶酒，琢磨着“我到底该怎么摆脱这个行业？” 因为这是一门好生意，需要有人来经营，所以我想：“好吧，我要申请商学院。” 我申请了八所商学院，在申请截止的那天晚上，我被其中七所拒了。后来，我不知怎么地考上了哈佛商学院，然后打电话给我爸爸说：“嘿……”

**“我真的很想这样做，但我需要先去接受一些训练！”**

**MP**：“我真的很想做，但我需要两年时间尝试做生意，我想我应该对会计更了解一些。” 他说，“是啊，这是个好主意，你应该去做。” 于是我就去了。我在宿舍里有一部 IP 电话，那是在剑桥奥尔斯顿的一间小宿舍。我插上了我的 IP 电话线路，这样我就可以继续为 Unspam 工作。电话响了，我接起来，然后我就可以做商学院的事情了。我比一般的商学院学生年龄大得多，而且我教法律，这让我感觉完全不受束缚，我当时想，“哦，天哪，每个人读研究生都错了，尤其是商学院。”

**为何如此？**

**MP**：他们认为课程的目标是想出一些以前没人想过的聪明点子，这会把你引入歧途。而如果你是老师，真正的目标是带领学生踏上苏格拉底式的旅程来阐述一个观点，从A点出发，到B点结束，最终找到这条弧线。所以我在商学院的教学方式是——每当有人提出一些疯狂的观点时，我都会尽力把我们拉回到正确的弧线上，这让我成为了每位老师最亲密的朋友。

**尤其是哈佛的教学风格，他们试图通过案例等来引导讨论，而不一定是通过讲座，我认为这肯定很有吸引力。**

**MP**：是的。很有意思的是，几乎每节课，因为老师会介绍学生的情况，所以大概上过三节课，老师就会过来问我：“你以前在法学院教书吗？”我会说：“是的。”他们会说：“没有什么比律师更能毁掉一段好的谈话了。”我说：“是啊，没错。”然后他说：“事情是这样的。你不让我难堪，我也不会让你难堪。”我说：“一言为定。”所以，那时候商学院变得非常有趣，我学得也不错。但大部分时间我都在琢磨，我能做什么，才不会让我回到我爸的Hooters餐厅？

#### Cloudflare 理念

**Cloudflare 的想法是从哪里来的？它直接源于 Unspam 的工作吗？你当然可以看到其中的联系。它和克莱顿·克里斯滕森以及颠覆性创新有什么联系？[你回顾过去时谈到过这个问题，](https://stratechery.com/2021/cloudflare-on-the-edge/)但当你回顾过去时，你可以这样描绘：我了解了颠覆性创新，我想创办一家服务于服务不足市场的公司。事情就是这样运作的吗？或者说，实际的进展是怎样的？**

**MP**：这些事情都始于人。有两个人至关重要，一个是 Unspam 的员工，名叫 Lee Holloway，我们刚从大学毕业就聘用了他，他——有些人真的是技术天才，Lee 就是其中一位，他简直不可思议，我们不仅打造了 Unspam 的核心技术，还开发了各种副业。

其中一个副业项目叫做“蜜罐计划”（Project Honey Pot），它实际上培养了Y Combinator创始人[保罗·格雷厄姆（Paul Graham）](https://x.com/paulg)。在保罗创办Y Combinator之前，他曾在麻省理工学院（MIT）主持一个名为“麻省理工学院反垃圾邮件会议”（MIT Anti-Spam Conference）的会议。有一年，他邀请我去做一个演讲，讲的是如何制定法律，把垃圾邮件发送者关进监狱。那次演讲反响不错。他说：“回来，做同样的演讲吧。”我说：“我不会再做同样的演讲了，我是律师，我不会对一群技术人员做同样的演讲，他们第一次还挺客气的。”他说：“哦，你会想出点办法的。”

于是我回到李身边，我说：“我们能不能建立一个系统来追踪坏人是如何窃取你的电子邮件地址的？”，后来这个项目就变成了“[蜜罐计划”](https://www.projecthoneypot.org/)。我做了这个演讲，它非常受欢迎，我把它放在了角落里，在接下来的几年里，超过 10 万人注册了这项服务。当时李就在那里，而我去读商学院了，他继续在 Unspam 做技术工作。但那不是最高、最有趣的技术工作，他打电话给我说：“嘿，你一直对我真的很好，但是……”，在谈话的那个时候，我说：“等等，给我点时间，我会想办法的。”因为李是那种你只想加入团队的人，他收到了谷歌和 Facebook 等公司的录用通知。

另一方面，商学院的很多人都很讨厌，但有一位女士真的只是真心实意地试图找到正确的答案，并且对整个事情没有虚荣心，她就是[米歇尔 [Zatlyn\]](https://en.wikipedia.org/wiki/Michelle_Zatlyn)。米歇尔显然与我相反，我不是最有条理的人，我不是最自律的人，我不注重流程。米歇尔是六西格玛黑带，她具备所有这些素质。在 Unspam 工作时，我和另外两个朋友一起开始，我们吵得很厉害，所以我真的想寻找与我真正不同的人，而米歇尔正是我不擅长的那些事情，她却非常擅长。所以我总是向她推销各种想法。

我的大部分想法，现在回想起来，真的很糟糕。但其中一个想法是我跟她讲“蜜罐计划”，她当时完全被迷惑了。她问：“为什么人们要报名参加这个？”我说：“因为我们可以追踪坏人。”她说：“是啊，但这需要付出努力，你们给他们什么奖励吗？”我们说：“嗯，他们应该得到认可。”她说：“这说不通，为什么有人要这么做？”我当时在剑桥中央广场的一家埃塞俄比亚餐厅里，沮丧地说：“米歇尔，总有一天他们会希望我们阻止他们。”她说：“就是这个主意，我们来实现它吧。”

那天晚上我就打电话给李，跟他说：“嘿，这个方案是这样的，我们打算这样做。” 我花了30分钟向他详细讲解了我和米歇尔勾勒出的整个想法。讲到最后，他停顿了大约五分钟，我以为电话已经断了，最后他说：“嗯，这个方案可行，我们就这么做吧。” 就这样，一切就这么开始了。

所以，Lee、Michelle 和我就是公司最初的三位联合创始人，它最初是一个学校项目。最初的想法是：“你能把防火墙放到云端吗？”所以 Cloudflare 就是我们在云端玩防火墙的游戏。至少在最初的五年里，我们在接下来的几天里勾勒出的蓝图，最终成为了未来五年近乎完美的路线图。

**所以，这项服务对小型网站来说非常有利，你可以免费注册 Cloudflare，保护自己免受分布式拒绝服务 (DDoS) 攻击。这形成了一个良性循环：互联网服务提供商 (ISP) 会感激你的存在，因为你消耗了大量不良流量，他们很高兴你的加入，这为你提供了更好的服务。[我之前写过关于它的](https://stratechery.com/2021/cloudflare-on-the-edge/)成长故事，它真的非常引人入胜，引人入胜。**

**正如您所说，这里有一个颠覆性的角度，即您从完全不同的角度来处理问题，比如使用公共云或本地软件。您的想法和方法截然不同。您和教授们之间有约定吗？您不会让他们出丑。那么您是直接去找[克里斯滕森教授](https://en.wikipedia.org/wiki/Clayton_Christensen)，然后说：“哦，是的，我已经做过了”，还是说这其中有什么联系？**

**MP**：我记得当时我刚上完Clay的课程，我们就开始在Cloudflare上工作了，所以当时感觉很新鲜。我和Clay的关系非常密切，我非常喜欢这门课程，也有机会跟他学习。我觉得这门课程给了我们早期的信心。我们知道，除非是面向消费者的业务，否则创业非常困难，即使面向消费者也很难，但在B2B领域，靠每月20美元的费用做大生意真的很难。

**每个人最终都会组建销售队伍并向大公司销售产品，这是有原因的。**

**MP**：没错。所以我们知道，唯一可行的办法就是，像我们现在这样，把产品卖给大企业，价格动辄几千万甚至几亿美元。但问题是，如何从这里走到那里？

我认为，由于我们需要数据，而且我上过 Clay 的课，所以我们必须构建这个网络，这些都给了我们信心，让我们可以说：“好吧，我们将开始专注于如何免费提供这项服务，这项服务虽然在很多方面有所精简和限制，但会比现有的任何其他服务都要好，然后继续向高端市场发展。” 到了某个时候，我们会跨越一个临界点，即我们拥有的功能将超过人们从其他替代方案（例如 Akamai 和本地硬件供应商）那里获得的功能。然后，超大规模企业开始兴起，事情就是这样发生的。

今天这很有趣，因为在某种程度上，我们只是相信，如果足够多的互联网流经我们，我们就会找到某种方式围绕它开展业务，我们确实相信帮助建立更好的互联网的使命，我认为这两件事加在一起让我们获得了成功并达到了今天的规模。

**这是否具有讽刺意味，或者是有意为之，您对世界各地发生的犯罪行为的这种精品兴趣不再受到地域限制，基本上您需要某种推动力来吸引人们进入您的业务，因此罪犯实际上是您最重要的人，他们是您的销售人员，他们为您创造了市场？**

**MP**：是的，我们从来没有这样想过，而且有很多次，有黑客说“我改过自新了，我想找份工作”，而我们则说“是的，不太确定”。

但我认为互联网是一个复杂的地方。早期人们会问：“你的竞争对手是谁？”我说：“Facebook”，我现在仍然相信这是真的。我认为，如果没有Cloudflare，互联网的更多部分会存在于Facebook上，因为对于一个小网站所有者来说，忍受这种麻烦变得越来越困难，所以人们都在外面的围墙花园里竞相竞争。我们试图做的是说：“嘿，我们仍然给你创新的自由，但帮助你获得某种保护，让你不必担心外面所有令人讨厌的东西。”所以我认为，就像亚马逊之于Shopify，Facebook之于Cloudflare是一样的。这不是一个完美的比喻，但我认为我们试图将数量降到最低，然后让任何类型的创造力或独特性在幕后发生，而围墙花园则是另一回事。

#### Cloudflare 今日

**Cloudflare 现在是什么样的？您觉得它和 10 年或 15 年前的理念有什么不同？这条路是否和您预期的差不多，只是中间有一些明显的岔路？还是最终的走向与预期不同？**

**MP**：我认为 Cloudflare 的真正故事是，我们想在云端放置防火墙，我们知道，为了向摩根大通、政府和大型医院集团等机构销售产品，我们必须从小处着手，逐步发展，所以我们免费提供这项服务。

我们没想到的是，这会引发如此多的问题。世界上所有的人道主义组织都算是我们的第一批客户，因为他们面临着严重的安全问题，而预算却很少，所以他们都注册了，然后有人会试图用各种方式对他们进行 DDoS 攻击或黑客攻击。

**不是理想的客户。**

**MP**：嗯，从某种程度上来说，我们完全是理想的客户，因为我们学到了很多东西，但之后我们会阻止他们。但之后所有的黑客小子都会说：“哇，这些家伙真厉害，我要注册，因为我所有的朋友都在想黑我。”于是所有的黑客都盯上我们了，然后他们就会想尽一切办法来攻击我们。

那么，我们为什么要[建立域名注册商](https://domains.cloudflare.com/)呢？这生意很糟糕，它是一种商品，我们为什么要做呢？因为我们考察了其他所有公司，没有一家是安全的，而且cloudflare.com差点就被黑客入侵了，那将是一场灾难。所以我们当时想，“解决这个问题的唯一方法就是把它带回公司内部”。我们为什么要开发[VPN替代系列产品](https://www.cloudflare.com/zero-trust/solutions/vpn-replacement/)？因为我们考察了该领域所有其他公司，我们觉得，这些人根本无法应对我们所有产品的规模和复杂性，所以我们只能自己动手。我们为什么要开发[开发者平台](https://developers.cloudflare.com/)？不是因为我们打算把它卖掉，而是我们自己构建了它，我们需要它，我们需要一个沙盒环境，这样我们才能构建东西。

现在，所有这些对我们来说都变成了实质性的产品线。但 Cloudflare 的真正故事是，我们创造了一系列问题，然后自己解决，并在这个过程中将这些问题转化为产品。

我从未想过——我们第一次与风险投资人会面时，这个人叫[雷·罗斯洛克](https://www.rayrothrock.com/)(Ray Rothrock )，他是 Cloudflare 的第一笔投资人，我们向他推销，他说：“太好了，我只有一个问题”，我们说：“好吧，什么？”，他问：“你打算如何处理死亡威胁？”，我说：“死亡威胁？”，米歇尔说：“你到底在说什么？”。

**（笑）六西格玛黑带不希望生活中出现不确定性。**

**MP**：他说，“如果你这么做，你会惹恼一大群人。”所以我从来没想过我会受到俄罗斯政府的个人制裁，甚至被列入普京的暗杀名单，这可不是我一开始就想做的事情，所以我觉得我们做的非常不同。我想，因为我们把它发布得如此广泛，最终会触及很多内容，这意味着时不时地——虽然这很有意思，你说我们之前两次谈话都是关于内容审核的。

**我正想问你，这个话题是不是已经过时了？好像现在很少再提起了。**

**MP**：嗯，到九月份我们就成立15周年了。我们之前发生过三起重大事件，其中有些是非法内容，我们一直在删除。但对于那些不道德但并非违法的内容，我们有一个灰色地带，15年内只发生过三起。所以平均事件发生间隔是五年，我们该再发生一次了。肯定会发生一些事情，但我不知道会是什么。

但本质上，如果你身处互联网的背后——现在，我们在意大利、西班牙和其他一些地方被起诉，主要是因为足球队的老板不喜欢别人非法播放他们的内容。我们不喜欢别人非法播放他们的内容，这会浪费大量的带宽。我们试图阻止这种行为，但他们很聪明，总能找到不同的方法。所以我们参与这些斗争，我们试图避免这种情况发生，所以我认为我们所做的事情的本质就必然会引发一些争议。但坦率地说，这也是它如此有趣的原因，因为我们确实发现自己身处一些非常有趣的政策辩论之中。

**我确实想稍后再谈这个问题，但我认为 Cloudflare 的一个特点非常有意思，那就是每周都会发布 47 个产品公告，很难一一解析。我喜欢这样做的公司，就像当年的 Nvidia 一样，他们知道自己在 GPU 和 CUDA 方面有所成就，所以他们会把很多东西都抛到一边。他们每年举办两次 GTC，每次都会有人问我：“他们是怎么做到的？”[我会说](https://stratechery.com/2021/nvidias-gtc-keynote-the-nvidia-stack-the-omniverse/)：“嗯，因为都是同一个平台，所有东西都是软件定义的，所以他们实际上只是发布了一些彼此略有不同的库，但他们可以将它们定义为不同的东西。”**

**这有点像Cloudflare，在商用硬件上采用这种通用架构。从一开始就有这样的目的吗？还是说，这只是一种类似“我们是一家规模较小的初创公司，把服务器放在ISP那里，我们买不起奔腾处理器或其他更高端的处理器”的功能，而最终这实际上成为了我们构建基础设施的方式？**

**MP**：是的，我认为我们真的受到了谷歌故事的启发。想想谷歌推出之前谁是领先的搜索引擎？是AltaVista。AltaVista是谁开发的？是数字设备公司（DEC）。DEC的业务是什么？他们销售大型机。他们为什么要推出搜索引擎？这是为了展示他们的大型机有多强大。

后来，拉里·佩奇和谢尔盖·布林成为了斯坦福大学的学生，谷歌的关键创新实际上不是搜索，而是他们比其他任何人都更高效地存储和处理数据的能力，比任何人都更便宜、更快，而且他们可以有效地将大型机扩展到无限大，而他们只使用一堆商用硬件就做到了这一点。

我想，从一开始，我就坚定地站在这个阵营，李也坚定地站在这个阵营。我们团队里有些人并不坚定地站在这个阵营，所以早期就存在争论，争论的焦点是，我们是应该用专门的硬件来实现各种功能，还是应该用市售硬件，然后编写智能软件来实现调度以及所有你需要的功能。这又是一个争论。我们当时在帕洛阿尔托的一家美甲沙龙楼下，楼上有八个人，事情的发展方向可能有两种。如果当时的方向不同，我们就不会是今天的公司了。

如今，Cloudflare 几乎每台服务器都属于不同的代系。我想我们现在是第 13 代，我们会批量购买服务器。服务器现在有五个主要组件，包括网络、CPU、GPU、长期存储和短期存储（也就是 SSD 和 RAM）。我们不断地问自己，这也是很多创新的源泉。我们的基础设施团队和产品团队每季度都会举行一次会议，他们称之为“冷热会议”，会问每个人都会问的问题，那就是“我们哪些地方运行得比较热？”，然后问题是“我们可以通过工程设计来提高效率吗？还是我们必须在这些地方部署更多硬件？”

但他们提出了一个同样重要的问题：“我们在哪里停滞不前？”我认为这才是更有趣的问题，即“我们已经为某种资源付费的机会在哪里？它存在于野外，但尚未得到充分利用，我们可以利用这种资源做什么？”我对此的类比可能有点令人反感：如果有人想出在浴室和酒吧的小便池上方出售广告空间的主意，我可以说有人想过这样做，它一文不值，但一旦有人这样做，你就会突然从中产生新的收入来源。

Cloudflare 一直在问这个问题，即“那种额外的空间、额外的容量在哪里？”，如果我们可以出售它，那么我们就会有更高的利润，这让我们能够继续做我们所做的所有事情。

**那么，机会主义商业和有意图的商业之间的区别是什么呢？您长期以来一直强调这一点吗？**

**MP**：我们不是那种能拿出几百页商业案例的地方。我的意思是，我们开发[Cloudflare Workers](https://workers.cloudflare.com/)是因为我们需要它，而且现在它是我们业务中增长最快的部分。

#### Cloudflare 的利基市场

**您之前提到过成为第四种云，[我写过相关文章](https://stratechery.com/2021/cloudflares-disruption/)，但感觉最近我听到的关于它的消息并不多。公有云对此的响应速度是否比您预期的要快，[比如 Lambda](https://en.wikipedia.org/wiki/AWS_Lambda)之类的？**

**MP**：我认为发生的事情是，我们每个人都开辟了自己的道路，虽然我们在边缘竞争，但我们在这些道路上却截然不同。

类比一下，我认为公司有个性，就像从事不同职能的员工一样。像AWS、谷歌、微软Azure这样的超大规模企业，他们的职能个性是DBA，即数据库管理员。如果你曾经和DBA共事过，你会发现他们很聪明，但往往比较死板。他们认为数据库是世界的中心，他们所做的一切都是为了确保所有数据都必须存储在数据库中。

**公平地说，在他们的防御中，数据库是世界的中心。**

**MP**：有可能。我想，这也是你们展示立场的一种方式。另一种说法，更符合我们的说法，是网络管理员。

**有趣的是，你提到，实际上，对于 Stratechery 和 Passport 的业务，也就是我所从事的工作，我经常会考虑数据库。这真的非常重要！**

**MP**：我经常思考网络，你也经常思考数据库。但实际上，这两者的功能并非截然相反，只是存在着矛盾，因为数据库的目的是无论如何都要保存数据。而网络的目的是尽可能快地移动数据并将其从系统中移除，这是两件截然不同的事情。

所以，如果你非常擅长数据库，你可能不会擅长网络；如果你非常擅长网络，你可能不会擅长数据库，我认为这是事实。我认为超大规模数据中心的网络产品有点糟糕，因为他们实际上不希望数据离开，而我们的数据库产品，说实话，不如超大规模数据中心的产品好。当然，它们确实有某些优势，有时你想用它们，却不想用它们，但相对于我们业务本身而言，它们绝对是次要的。

我认为，对 Cloudflare 来说，一个糟糕的结果是，未来世界会变得更加“单云化”，每个人都会说：“我百分之百地把所有资源都放在 AWS 上了”，这样一来，我们能做的事情就不多了。对 Cloudflare 来说，一个好的结果是，未来世界会更加“多云化”，这样一来，网络就成为了在不同云提供商之间实现合理化的关键。

**这是人工智能实际上对您有益的领域吗？与其说是打造人工智能产品，不如说是它激励企业更多地采用多云技术？**

**MP**：是的，我认为确实如此。而且我也认为，是的，如果你拥有需要以各种方式移动的大型数据集，我们在这方面非常擅长。有很多事情出于这样或那样的原因——如果你现在要构建一个代理，Cloudflare Workers 可能是构建它的最佳平台。你可以启动它，也可以关闭它，它可以连接所有地方。无论如何，它都必须经过我们，因为我们是互联网的很大一部分，所以我认为实际上这种人工智能和人工智能代理一直是 Cloudflare Workers 的杀手级应用，但这与如何运行无关——我们永远不会成为运行[SAP HANA 的](https://en.wikipedia.org/wiki/SAP_HANA)合适场所，但我们绝对是运行必须与互联网其他部分交互并在互联网上移动的代理的合适场所。

#### 按次付费

**现在我们聊这个话题的原因在于，我们今年和去年都线下聊过几次，是因为你们一直在推动按次[付费的爬取模式](https://stratechery.com/2025/cloudflares-content-independence-day-googles-advantage-monetizing-ai/)。能否从你的视角，从整体上给我讲一下你的方案？我认为这个方案已经有所改进。我想根据我的一些反馈来思考一下，但2025年9月的方案是什么呢？**

**MP**：我们先暂时把 Cloudflare 放在一边，讨论一下——

**谈谈英语专业的学生马修吧？他是学生报纸的编辑。**

**MP**：这是我作为法学教授的内心独白。我来给你讲讲互联网的历史，互联网为什么会以现在的方式存在，以及它正在发生哪些变化。

**这通常是我的工作，但请继续。**

**MP**：你可以告诉我哪里错了，但这是我对互联网的简要介绍，并向讨厌历史课的米歇尔表示歉意。

过去25年，互联网的界面一直是搜索，而谷歌一直主宰着这个领域。谷歌作为一家公司，其动机是让互联网尽可能地发展，因为如果出现混乱，搜索就会成为混乱的组织者。但你需要激励人们真正地创造内容，所以谷歌不仅要创造组织互联网的东西，还要创造吸引流量的东西，然后帮助人们将其货币化，主要通过广告，尽管他们也提供订阅服务。谷歌是过去25年来互联网的伟大赞助人。如果没有像谷歌这样的公司来创造激励机制，网络就不会像现在这样存在。

围绕流量的激励机制存在很多问题，我们创建的系统让人们尝试创建煽动性的标题来吸引人们点击某些内容，这样他们就可以投放广告，这并不完美，但如果没有谷歌和搜索的资助，我们就不会拥有今天的互联网。

情况正在改变。世界正在发生变化，网络界面正在从搜索引擎（搜索引擎会给你一张藏宝图，然后说：“嘿，点击这 10 个蓝色链接找出你的答案”）转变为答案引擎。所以，如果你看看 OpenAI、Anthropic、Perplexity，甚至看看现在的谷歌，你会发现它们并不是搜索引擎，它们不会给你藏宝图。相反，它们会在页面顶部直接给出答案。对于大多数用户，95% 的用户，95% 的时间来说，这个答案是一个更好的用户界面。我并不反对答案引擎，也不反对人工智能，我认为，从各方面来看，它都应该是我们所有人交互的界面。

但问题是，如果你得到了答案，却没有得到藏宝图，那么你就无法产生流量。如果你无法产生流量，那么整个基于流量的网络商业模式就会开始崩溃。你会发现，这种情况在电商网站上并不常见，在那些实际向你出售实物的产品上也并不常见。因为如果你问哪款相机最好，即使你得到了答案，你仍然需要去别的地方购买。这需要电商和销售产品的人来推动，但写评论的人——

**实体产品的优点在于其定义上的稀缺性，而互联网上的文本的问题在于它并不稀缺。**

**MP**：它并不稀缺，完全正确。谷歌设定了这样的期望：每个人都可以免费抓取互联网内容，但这从来都不是免费的。互联网从来就不是免费的。谷歌为此付出了很长时间，与内容创作者的交换条件是：“我们获得你的内容副本，作为交换，我们会给你流量，并帮助你将这些流量货币化。”

随着我们从搜索引擎转向问答引擎，这种交换条件就失效了，所以有些事情将会改变。我认为会有三种可能的结果。再说一次，这一切都不涉及——如果Cloudflare明天消失，这种情况仍在发生，以下三件事之一将会发生。第一，世界上所有的记者、学者和研究人员都会饿死。这太疯狂了，就像你在推特上发布这些内容时，有多少人会说：“好吧，我们真的不再需要记者了，我们有无人机了”，而我会说：“我觉得我们仍然需要记者”。

**我经常强调这一点，因为人们会把 Stratechery 视为未来新闻业、订阅以及整个小规模订阅模式的典范。我为 Stratechery 在探索这一方向所发挥的作用感到非常自豪，但我一直说，你看，我的文章通常都是全新的内容，但我的更新，我该如何打开它们呢？我会引用记者的话，比如“好吧，现在我要分析一下这条有人真正收集到的新闻”，所以我同意你的观点。**

**MP**：你怎么知道该写什么呢？你只是这些东西的衍生品。这很棒，我认为这增加了巨大的价值。而且我认为，正如你之前所说，仍然会有一些大制作内容存在，人们需要关注。但还有很多其他的事情，比如世界上发生的事情，需要有人去报道。而且，如果商业模式彻底消亡，这些内容也随之消失，这确实存在风险，我认为这将是一种损失。我不认为最终结果会是这样。

**你不觉得这种情况在某种程度上已经发生了吗？人们担心，抱怨全国性报道太多，但说到地方，地方报纸却被互联网摧毁，既赚不到钱，也没有记者，谁知道市政厅的情况呢，等等。**

**MP**：我认为谷歌的付费流量模式对本地媒体的支持力度不够。我和妻子买下了我们家乡的一份地方报纸，结果出乎意料的是，它竟然成了一门相当不错的生意。

**[我非常提倡](https://stratechery.com/2017/the-local-news-business-model/)这一点。我认为地方新闻确实可以做到这一点。**

**MP**：嗯，我认为它会变得更有价值，所以我们会讨论这个问题。第二种可能的结果——

**抱歉，你没跟教授们达成一致，你得告诉我让你做饭，别老是打扰你，惹你生气。抱歉，你继续吧。**

**MP**：不，没关系。

**抱歉，请继续。**

**MP**：这是您的时事通讯和节目。

**继续，继续。**

**MP**：我只是来娱乐一下！第二种选择是——我不觉得这有什么稀奇古怪的，但我觉得这有点像《黑镜》里的那种选择，也就是说，你能想象萨姆·奥特曼明天宣布他要成立一个他自己版本的美联社吗？

**是的，我可以。**

**MP**：明白我的意思吧，你能想象他宣布收购Gartner吗？这并不疯狂。我们可能不会回到20世纪的媒体时代，而是回到15世纪的媒体时代，就像美第奇家族一样，有五个强大的家族控制并资助各种不同的内容创作。在这种情况下，很可能有一个保守的家族，一个自由的家族，一个中国家族，还有一个印度的家族。欧洲人会尝试建立一个，但行不通，他们会使用自由的美国家族，而内容创作很可能就是这样。

我认为这并非不可想象，实际上与 Scale AI 所做的并无太大区别，Scale AI 为人们整理内容并将其标记化。现在有很多记者失业，建立一些机构的成本并不高，但我认为这是一个非常危险的结果，因为它极其倒退，而互联网曾是伟大的平衡器，伟大的知识传播者。如果我们突然之间每年都要花费数千美元购买任何 AI 系统，即使是富人也可能会选择其中之一，而如果你获得的知识都来自那个系统，那么我们突然之间又陷入了孤立状态，这看起来非常非常危险，但并非完全不可想象。

**第三个是什么？**

**佩珀马斯特**：第三点是，我们要找到一种新的商业模式，它必须类似于人工智能公司和现有的问答引擎创造的收入，其中一部分要返还给内容创作者。至于这如何运作，我想这正是我们正在努力探索的。

但我认为，如今成为一家人工智能公司确实需要三件事。你需要获得GPU，OpenAI每年在这方面投入超过100亿美元，但这是硅，就像所有硅一样，随着时间的推移，它会越来越成为一种商品，这是必然的，因为硅短缺从未演变成硅过剩。世界上每个人都在竞相与英伟达竞争，英伟达可能会在很长一段时间内保持领先地位，但AMD、高通、苹果以及所有超大规模计算厂商都会推出新产品。

**我们只需要保持台湾的完整。**

**议员**：我们确实需要保卫台湾的安全。据我所知，你们一直在行动，是防御的第一线。

**[家庭原因……](https://stratechery.com/2025/a-personal-update-and-vacation-break/)**

**MP**：唯一阻止中国人来的就是本。

第二件事是人才——如今我们面临着巨大的人才短缺。坦白说，五年前如果你要读人工智能博士学位，那你只会被人嘲笑。那是一个死胡同。如今，人工智能是最热门的领域，每个人都在推出新的东西，所以我们可能不会从人才短缺变成人才过剩，但肯定不会——如果你是一位优秀的人工智能研究员，能拿到10亿美元来Meta工作，那种日子已经不多了，而且不会永远这样。我们会有更多的人涌向这个领域，因为教育市场和就业市场都很高效。

第三件事是内容。同样，由于谷歌的存在，我们期望所有机器人都能免费获得内容，但这是不可持续的。最终的结果要么是人人饿死，要么是让我们重回美第奇家族时代，所以我们必须找到新的交换条件。

你可以想象，任何一家人工智能公司，每位月活跃用户每年1美元的收入都会进入一个资金池，然后分配给内容创作者。算一下，这相当于今天大约100亿美元的内容收入，完全可以取代如今“无围墙花园”互联网产生的所有广告收入，比如Instagram、Facebook、TikTok等等。

**比如 Google 网络、TradeDesk 等。**

**佩珀马斯**：但算上《华尔街日报》、《纽约时报》、《金融时报》、Reddit 以及其他所有媒体，每年大约有 100 亿美元。这笔钱不少，但也不是疯狂的数字。如果我们能解决这个问题，就能创造出更好的商业模式。如果我们做得对，我认为它实际上能同时鼓励创造更好的媒体和内容。

**您之前谈到了如何从这里到达那里，这当然也适用于此。此外，还有一个问题是，我们想要什么，而不是为了应对经济问题。事实上，文本可以无限复制和传播，即使你花了数年时间担心垃圾邮件，垃圾邮件仍然是一个问题。**

**MP**：算是吧。现在不像以前那么严重了。说实话，如果你——我记得比尔·盖茨说过，“我们会在未来五年内解决垃圾邮件问题”，而我当时就说，“不，你做不到，这是一个棘手的问题”。实际上，他们基本上做到了。

**这一切都归功于像我这样的人，他们只是想向人们的收件箱发送一份诚实的新闻通讯。**

**MP**：是的，这让你的日子更难过了。

**当你思考这个问题以及你能做什么时，你如何区分“好的，这就是我想要发生的事情”与“这是可能发生的事情，而且实际上我可以利用一些杠杆来实现它”？**

**MP**：我认为首先所有市场都需要稀缺性，如果没有稀缺性就不会有市场，市场中必须存在某种稀缺性。

**但互联网的问题在于，45% 的人可能会决定接受这个概念，但另外 55% 的人却不予理会。**

**MP**：有可能，但如果他们无法获得内容，那就更难了。所以我们刚才看到的是，对于内容创作者来说，尤其是如果他们创造了稀缺性，他们实际上就能够达成交易。

看看Reddit。Reddit一直非常积极地阻止机器人抓取其内容，包括谷歌。他们说：“谷歌，除非你付费，否则你无法访问这些内容。” 结果，谷歌和OpenAI与Reddit达成了一项协议，他们支付费用——我们从他们的公开文件中得知，这不是什么机密——但在2024年，他们获得了1.2亿美元的报酬，我听说2025年的协议甚至更好。我认为你会看到这个数字继续上涨。

我认为真正有趣的是，如果你把Reddit的代币数量和《纽约时报》总目录中的代币数量相加，结果发现它们的数量级大致相同，但Reddit的代币数量却是《纽约时报》的七倍。问题是为什么？我认为答案是因为《纽约时报》——你可能喜欢《纽约时报》，但从法学硕士的角度来看，《纽约时报》、《华尔街日报》、《金融时报》、《华盛顿邮报》和《波士顿环球报》之间的区别其实并不大。事实就是事实。所以，是的，它们之间可能存在一些偏见和倾向，观点版面也可能略有不同，但大体上是一样的。

所以我认为真正令人鼓舞的是，我们已经看到了稀缺性和独特内容的结合，比如，如果你没有Reddit，你就没有Reddit，所以你需要拥有这些内容。我认为，如果我们能够在这种情况下创造稀缺性，市场就会表明，本地化、独特、差异化的内容才是最有价值的。随着这个内容市场的存在和发展，无论Cloudflare如何，如果Cloudflare消失，我仍然认为出版商不可避免地会说：“好吧，我们要关闭机器人访问我们内容的权限，然后我们要为最独特的内容创造一个市场。” 内容越独特，你就能获得越多的报酬，我认为这是不可避免的。

我认为这是我们转变网络商业模式的关键时刻之一，而且我实际上非常鼓舞，因为内容的未来将更像早期的互联网，而不是像我的朋友本·史密斯 (Ben Smith) 和 BuzzFeed。

#### Cloudflare 的力量

**但如果这是不可避免的，那Cloudflare为什么需要如此激进呢？你们制定了这些政策，竭尽全力屏蔽机器人程序，制定了识别其价值、支付等协议，当然，这些都还处于萌芽阶段，还有很多问题有待解决。但你们并没有摆出一副公司姿态，认为这是不可避免的，而且会取得巨大成功，而是在非常积极地努力推动事情发生。**

**MP**：嗯，我想就算我们不做，也会有人做的。但我认为我们的独特之处在于，我们非常擅长阻止机器人之类的攻击，因为我们每天都在做这件事。

所以，再说一遍，我们不是坐在那里想着“嘿，下一步该怎么办？不如去改变网络的商业模式吧”，而是我们的客户，也就是出版商，来找我们说：“我们快要完了，我们没有足够的技术手段来阻止它，但我们必须阻止这种情况，请帮帮我。” 说实话，当 Dotdash Meredith 的[Neil [Vogel\]](https://www.iac.com/business-management/neil-vogel)告诉我这些的时候，我翻了个白眼，心想：“出版商，他们真是勒德分子，总是抱怨新技术，总是抱怨接下来的事情，这没什么大不了的。”尼尔和其他一些人最终说，“直接去提取数据”，而只有当我们真正看到数据时，我们才发现，在过去 10 年里，在同样的基础上从谷歌获得同样数量的内容的点击变得困难 10 倍，现在对于 OpenAI 来说是困难 750 倍，对于 Anthropic 来说是困难 30,000 倍。

互联网流量作为货币的模式正在消亡，所以，要么内容创作将消亡，变得毫无意义，要么我们必须创造一种新的商业模式。再说一次，如果我们的使命是帮助构建一个更好的互联网，那么这似乎完全符合我们应该努力的方向。

**那么[为什么 Garry Tan 说](https://x.com/garrytan/status/1961115612996145381)你们是 Browserbase 的邪恶轴心，并且你们应该使 AI 代理合法化呢？**

**MP**：我真的不明白。我的意思是，我对Garry感到困惑，我想部分原因可能是他是Perplexity的投资者。

每个故事都需要四个角色：一个受害者，一个反派，一个英雄，以及一个村里的傻瓜或傀儡。仔细想想，任何新闻报道都有这四个角色。目前，扮演反派最多的[是“困惑”（Perplexity）](https://blog.cloudflare.com/perplexity-is-using-stealth-undeclared-crawlers-to-evade-website-no-crawl-directives/)，他们为了绕过内容公司，不择手段。

我给你举个我们见过的他们的例子：如果他们被阻止获取某篇文章的内容，他们实际上会查询像 Trade Desk 这样的服务，这是一个广告投放服务，Trade Desk 会向他们提供文章的标题，以及文章内容的粗略描述。他们会根据这两点，编造文章内容，然后像事实一样发布，比如“这篇文章是由这位作者在某个时间发表的”。

所以你可以想象，如果 Perplexity 无法获取 Stratechery 的内容，他们就会说：“哦，本·汤普森写了这个”，然后他们就会编造一些内容，把你的名字也写上去。别管版权了，这简直就是欺诈，这就是一些科技公司的不良行为，我认为这些行为应该受到谴责和惩罚。

#### 谷歌问题

**不过，时间回溯到 2024 年，我不记得是你还是其他人指责 Perplexity 忽视了“禁止抓取网站内容”的规定，人们对此非常愤怒。[我当时的反驳是](https://stratechery.com/2024/perplexity-and-robots-txt-perplexitys-defense-google-and-competition/)：“我们到底想不想和谷歌竞争？”因为谷歌的做法是肯定的，OpenAI 开创了他们自己的网络机器人和协议，你可以要求 OpenAI 不要抓取你的网站，他们尊重这一点。谷歌也做了同样的事情，但这个谷歌机器人是经过 Googlebot 扩展的，真正的 Googlebot 会说：“好吧，你想参与搜索”，而事实证明，AI Overviews 是由 Googlebot 控制的。那么，为什么谷歌可以做到，而 Perplexity 不能呢？**

**MP**：我认为我们应该打个开放式的赌，你想赌什么我都接受。这在某个时候可能就像一场篮球比赛。无论它是什么，都可以，比如12个月后，谷歌是否会为出版商提供退出AI概览的选项？我的答案是肯定的，但我想你会说否定的。

**我不知道，我得考虑一下，我还没有承诺，我还没有承诺。**

**MP**：好的。但我同意你的观点，你也提到了，问题在于谷歌。我们做这件事的时候非常谨慎。我们实际上屏蔽了训练，他们用 Gemini 来做这件事，所以我们可以屏蔽它，而且我们已经在各个地方屏蔽了它。我们没有屏蔽 RAG，也没有屏蔽任何公司的搜索，包括 Perplexity，即使他们做了一些非常不道德的事情。顺便说一句，OpenAI 证明了你可以做正确的事情，你可以成为优秀的参与者，你仍然可以拥有更好的产品，并成为谷歌的有力竞争对手。

所以我认为谷歌会，而且我对他们充满希望，因为他们是一家真正相信生态系统的公司，他们看到了生态系统中正在发生的事情，他们明白如果不做出改变，生态系统将会受到影响。我认为他们会在未来12个月内自愿为出版商提供退出AI Overviews的途径。实际上，我乐观地认为这会比这更快发生，即使他们不主动这样做，我知道有很多监管机构会很乐意强迫他们这样做。

**[本周的谷歌案件](https://stratechery.com/2025/google-remedy-decision-reasonable-remedies-the-google-patronage-network/)让您感到鼓舞还是沮丧？**

**MP**：嗯，首先，它有 280 页长。

**它真的很长，我一直熬夜到凌晨 5 点，才把整篇文章读完。**

**MP**：我看过相关总结，我的团队也正在研究这个问题。所以我不确定强迫他们出售 Chrome 是否是正确的解决办法。

**不，我觉得这很蠢。我认为最重要的是，我对谷歌的看法是，[他们基本上已经付清了所有人的钱](https://stratechery.com/2024/friendly-google-and-enemy-remedies/)，然后允许他们继续付款的理由是，每个人都依赖谷歌支付。这就像一个循环论证，虽然没有错，但基本上就是这么回事。**

**MP**：再说一次，我不是这方面的专家——正如我们正在录制的，这是昨天发布的，所以我还不是这方面的专家——最让我担心的是，法官提出的一些建议是谷歌必须与生态系统的其他部分共享数据，因此，这样做的可怕结果实际上会摧毁生态系统，到目前为止，我看到的裁决中最令人沮丧的一句话是，“这似乎可能会对出版业产生不良影响，但没有一个出版商作证”，我想，“哦，伙计们，如果你们想要赢得比赛，你们就必须上场”。

所以我确实认为，无论Cloudflare做什么，互联网的经济模式都在发生变化。这种变化是因为包括谷歌在内的许多人正在提供更好的界面。如果这个生态系统想要繁荣发展并生存下去，我们就必须改变内容创作者的报酬模式。我们可以用一种我认为非常合理的方式来做到这一点。我预测，随着时间的推移，人工智能公司将更像YouTube或Netflix，它们在原创内容方面展开竞争，而不是在算法方面。我们已经看到，算法在很大程度上是商品，它们一直在相互超越，我认为这种情况还会持续相当长一段时间。我认为，未来将会有成千上万种不同的人工智能界面，我们可以为它们付费，或者它们会以各种方式获得广告支持，但最终决定你选择哪一种的，取决于它们是否能够访问你特别关心的内容。

**顺便说一句，我确实认为我同意你的赌注，因为在这种情况下获得巨大回报的谷歌核心战略基本上就是让每个人都得到回报。**

**MP**：没错。

**所以这就是为什么这是一个让所有人得到回报的计划。**

**MP**：没错。现在我认为这真的很重要——再说一次，我不知道具体是什么——但我们必须确保，无论采用什么支付方案，都要考虑到规模小的支付少，规模大的支付多。所以，我的假设是，每个月活跃用户每年支付1美元，你把钱存入一个资金池，然后我们如何分配给内容创作者，这又是另一个棘手的问题了。

**如果有人能坐在中间并成为做市商，那么这可能会带来非常丰厚的利润。**

**MP**：有可能。

**可能是 Cloudflare 吗？**

**MP**：不管我们是否如此，如果它盈利，就会有很多人竞相这么做。如果我们能够创造价值，我们就能获得部分价值。但我认为最重要的是，我们如何才能找到某种方式，让绝大部分利润真正流向那些创造内容的人？理想情况下，我们奖励的不是那些煽动愤怒的人，而是那些真正充实人类知识的人。

#### Cloudflare 的动机

**[我几个月前就写过这篇文章](https://stratechery.com/2025/the-agentic-web-and-original-sin/)，你知道我们的想法一致，只是你这样做引人注目的地方在于，也许有人必须真正打破一些鸡蛋才能实现这一点。**

**MP**：当然。

**但这引出了我最后一个问题，这个问题和我们之前讨论的内容审核以及现在讨论的内容审核息息相关：你们试图利用权力来推动这项新的内容标准，而你们声称在内容审核方面不想要权力，这两者之间是否存在着一种对立？“我们不想参与”，那么，你们到底想拥有权力还是不想拥有权力？**

**MP**：我们仍然希望在网络攻击来袭时能够及时阻止。我们正是凭借这种能力，才建立了一个相当有价值的业务。我认为我们之所以感到紧张，是因为我们不知道这些内容是好是坏。

举例来说，我认为确定谁作为内容提供商获得报酬的正确方法不是 Cloudflare 设计一个算法说：“这是重要内容，你应该拥有它，这是不重要的内容，你不应该拥有它”。每家人工智能公司都应该设计自己的算法，并将其接入到我们的系统或竞争对手的系统，无论是谁的系统，然后说：“这是我们的系统，你看到内容，根据我们的算法为我们进行排名，OpenAI 的排名将与 Anthropic 的排名不同，也与 Perplexity 的排名不同”。你应该具备这种能力，我认为排名应该基于两个不同的维度进行评分，即内容的信誉度和新颖性。它实际上对声誉有多大帮助？如果你做得正确，并且拥有真正多样化的——如果你拥有大量不同的人工智能公司，并且拥有大量不同的内容创建者，那么我认为，我们当然扮演着技术推动者的角色，但对于哪些内容重要、哪些不重要，实际的决定仍然是，我认为，那不是我们做出这些决定的正确地方。

**你想成为一名做市商，而不是挑选者。**

**MP**：我们不是编辑。

**就是这个意思。**

**MP**：是的。再说一次，我认为我们现在最擅长的一件事就是创造稀缺性，每个市场都依赖于此。无论我们还是其他人最终如何利用这种稀缺性盈利，这都很好。顺便说一句，我认为绝大多数情况下，大型出版商会与大型人工智能公司直接合作。

**但这对所有小人物来说都很糟糕。**

**MP**：我们也可以参与其中，说：“好的，我们现在代表小公司，和你们一起研究如何让你们也能参与到这个市场中来。” 双方都是小公司——小型人工智能公司和初创公司。我们必须确保在设计过程中，它能够以这种方式运作，然后是小型内容创作者。

我想问的是，既然你拥有一份相对有影响力的名单，如果你是一位学者，正在阅读本的文章，并且觉得这真的很有趣，谷歌最终能解决这个问题的原因是什么？我认为，作为世界上一股伟大的善的力量，谷歌为世界做了很多好事。他们这样做的原因是，他们非常仔细地思考了这些市场动态，因此，我们正努力与顶尖的学术经济学家和市场理论家合作，弄清楚“这个市场未来应该如何发展？”如果你正在读这篇文章，请联系我们，我们很乐意与你探讨这个问题。

**我们一开始花了很多时间——我们聊得有点长，感谢你坚持——谈论你的背景以及你是如何走到今天的，这真的很有趣。你的技术路线并非传统意义上的，这和我们最后的话题是否密不可分？如果你只是传统意义上的，从计算机行业起步，去斯坦福大学，创办了一家公司，你会选择这场斗争吗？或者说，这是你从现在已经不存在的法学院毕业后继续教书的后续经历，只是为了摆脱Hooters？**

**MP**：只是为了不再经营我爸爸的 Hooters 餐厅。

我不知道。我认为最有价值的是——我主修英语，辅修计算机科学，拥有法学学位，也拥有商学院学位。很多时候，我觉得在法学院的三年完全浪费了。但这三年真的非常宝贵。我可以坐下来读谷歌法官的裁决，了解其中哪些重要，哪些不重要，这些都非常重要。我认为，能够说、写、沟通，并理解创造优质内容需要付出多少努力，这些都非常有帮助。我想，过去我们讨论内容审核的时候，我们一直在努力解决这些问题，也一直在纠结什么才是正确的做法。

**它回到了你在床的哪一侧醒来，对吗？**

**MP**：是的，完全正确。

**（笑）对于那些不记得以前争议的人来说，[这是一个参考。](https://gizmodo.com/cloudflare-ceo-on-terminating-service-to-neo-nazi-site-1797915295)**

**MP**：当时你的女朋友，也就是现在的妻子，对你有多么生气？

但回过头去读亚里士多德，我觉得这很棒。所以今天我收到邀请，有几家人工智能公司邀请我加入他们的董事会之类的，但我总是拒绝。但我和他们交流过，我肯定是亚里士多德《[政治学》](https://en.wikipedia.org/wiki/Politics_(Aristotle))最大的非学术买家之一，因为几乎每个人工智能CEO都收到过我的签名版，上面写着：“你所做的事情非常重要，但你必须考虑道德问题以及如何建立信任。”

我确实认为，我希望那些正在构建这些具有巨大变革意义的系统的人们，他们确实会花一些时间在文科上，他们确实会花一些时间停下来阅读和思考，“好吧，如果我们成功了”——我认为有很多人说，“如果我们成功了，那就是詹姆斯·卡梅隆和终结者的某个版本”。但我认为还有另一种说法，那就是“好吧，如果我们构建了这些强大的系统，我们如何确保它们经久耐用？”，这些都是人们长期以来一直在努力解决的问题，我很幸运能有时间坐下来思考这些问题，我确实认为这对今天的 Cloudflare 公司很有帮助和指导意义。

**嗯，我很高兴也很幸运能接受您的采访。感谢您的到来。**

**MP**：本，谢谢你的邀请。

------

这篇每日更新访谈也提供播客版本。如需在您的播客播放器中收听，[请访问 Stratechery](https://stratechery.passport.online/member)。

每日更新仅限单人订阅，但偶尔转发也完全没问题！如果您想为您的团队订购多份订阅，并享受团体折扣（至少5份），请直接联系我。

感谢您的支持，祝您度过愉快的一天！

# English

## [An Interview with Cloudflare Founder and CEO Matthew Prince About Internet History and Pay-per-crawl](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/)

Thursday, September 4, 2025

Good morning,

Today’s Stratechery Interview is with [Cloudflare](https://www.cloudflare.com/) co-founder and CEO [Matthew Prince](https://x.com/eastdakota). Prince took a fascinating path to Silicon Valley — which we explore in this interview — but is most well-known for Cloudflare, which he started at Harvard Business School in 2009. Cloudflare provides networking services for websites in the cloud, and has [one of the most effective and fascinating freemium business models in tech](https://stratechery.com/2021/cloudflare-on-the-edge/).

In this interview we discuss Prince’s background, the original Cloudflare idea, and what Cloudflare is today — and the opportunistic way in which it became the company that it is. Prince’s latest focus is the economics of Internet content sites; he is very worried about the impact of AI on the traditional traffic business model that Google created, and is using Cloudflare’s power to try and create a new business model for content. We discuss Prince’s motivations and concerns, and why Prince believes this is a legitimate use of Cloudflare’s power, even if the ultimate decision-maker about the future of the web is Google.

As a reminder, all Stratechery content, including interviews, is available as a podcast; click the link at the top of this email to add Stratechery to your podcast player.

On to the Interview:

### An Interview with Cloudflare Founder and CEO Matthew Prince About Internet History and Pay-per-crawl

*This interview is lightly edited for content and clarity.*

**Topics:**

[Background](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#background) | [The Cloudflare Idea](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflare-idea) | [Cloudflare Today](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflare-today) | [Cloudflare’s Niche](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflares-niche) | [Pay-per-crawl](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#pay-per-crawl) | [Cloudflare’s Power](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflares-power) | [The Google Problem](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#google-problem) | [Cloudflare’s Motivation](https://stratechery.com/2025/an-interview-with-cloudflare-founder-and-ceo-matthew-prince-about-internet-history-and-pay-per-crawl/#cloudflares-motivation)



#### Background

**Matthew Prince, welcome back to Stratechery.**

**Matthew Prince:** Thanks, Ben.

**You’re actually one of the earliest Stratechery interview subjects. In fact, I’ve talked to you twice. In both instances, however, [the interviews](https://stratechery.com/2019/8chan-and-el-paso-cloudflare-drops-8chan-an-interview-with-cloudflare-ceo-matthew-prince/) were [very focused](https://stratechery.com/2021/interviews-with-patrick-collison-brad-smith-thomas-kurian-and-matthew-prince-on-moderation-in-infrastructure/), primarily on content moderation issues. This was also before I released my recordings of the interview, the modern Stratechery Interview format. To that end, that means I have an opening to still do my regular question for a first-time interview subject: I’ve never asked you your background like how did you get started, how you became interested in tech, take me back to the beginning.**

**MP:** I grew up in the mountains of Utah and when I was six years old — I actually used to think it was seven but my mother corrected me — when I was six years old in 1980, my grandmother for Christmas gave me an [Apple II Plus](https://en.wikipedia.org/wiki/Apple_II_Plus) and I took to it just like a duck to water. The University of Utah has a really actually world-class computer science department, and my mom used to take continuing education classes there and she would sneak me in basically as her precocious kid and she’d pretend to do the work but I’d really do all the homework.

**Oh, she was actually registered for classes and you were just tagging along?**

**MP:** That’s right. And I did things that made you really popular in high school, I went to computer camp, there was a computer camp, we had Cate, the boarding school out in Santa Barbara and I wish I’d stayed in touch with those people because I bet the people who were there with me have gone on to do pretty amazing things. And I went to college thinking I was going to study computer science and then having the hubris of an 18-year-old, I took Computer Science 105 or whatever it was and was bored out of my mind because I’ve been doing this for quite some time.

I actually transferred my major to study instead English Literature which is a different direction, but I still knew a lot about computers. I started college in 1992, it’s right as the Internet is taking off and they needed students who had some understanding. I was one of the student network administrators and that’s how I learned how switches and routers and everything worked. I had a fiber optic line straight from my dorm room back to the university’s router, had faster Internet for those first few years than I did for many years since.

**Oh, yeah. At Wisconsin people generally, I’ve said this before, but people usually only stay in the dorms one year, and me and some of my friends stayed in two years just because that was where you got faster Internet. But we didn’t have fiber optic lines, they were T2s or something like that.**

**MP:** Yeah, and that was that. I had offers at the end of college to go do a job that honestly I had no idea what it was but it was a product manager, which I was like, “What does that even mean?”.

**I think it’s 2025 and no one still knows what it means.**

**MP:** And are having to reinvent themselves again. But I think that I had offers at Microsoft and Yahoo and Netscape and I said, “No, I’m not going to do that”, and instead I’ll go to law school, and so I actually went to law school and thought I was going to be a lawyer for a while. And probably but for the dot-com bust, I probably would be.

**That’s not the answer you would expect, you would think, “The dot-com bust drove me out of tech”, how does the dot-com bust drive you out of law and into tech? That seems backwards.**

**MP:** Well, it kind of drove me into tech a little bit. I had found the type of law that I enjoyed was securities law, which is basically taking companies public and so the summer of 1999, I worked in San Francisco as an intern at a large law firm, we took six companies public over the one summer and it was amazing, it was so much fun. And my plan was go and work for the law firm for a while and then find a company that I thought was really great and help them in financing or whatever.

**Yeah, be their GC or something.**

**MP:** And I’d eventually go in-house, and that was what I thought my journey was going to be. Then the dot-com bust happened in March of 2000 and the law firm called up and said, “Hey, good news, bad news. Good news, you still have a job. Bad news, we don’t need any more securities lawyers. But bankruptcy law is basically the same thing and we think you can handle it”. And again, as a lawyer I’m like, “It kind of is the same thing”.

**A lot less fun.**

**MP:** There’s no in-house left after the company has gone bankrupt.

So there’s a guy named Doug Lichtman who’s a young law professor who would always come over to my apartment and we’d share a bottle of wine and he’s like, “My brother is starting a B2B company in the insurance space and they’re looking for someone with your skill set, would you be interested?”, and I was like, “Yeah, that sounds amazing”, and maybe they’d match your salary and they’ll give you this thing called stock. I was like, “Yeah, sure”, and so I did that and we raised about $6 million.

It was the same business model as almost like Rippling today but it was way before its time. And we were idiots and we blew it up in every possible way and so we blew through $6 million over the course of about 18 months and it was a colossal failure, but it was just incredible to see that a group of people could come together with an idea on a piece of paper, try to build something, fail, fail honorably.

Again, I think it just wasn’t the right time and then no one went to jail, which was pretty amazing. We lost all this money from the investors and the investors were like, “Yeah, that sucks, if you guys start something new, let us know”, and I was like, “Wow, this is some sort of magical world I didn’t know existed”. So I then spent the next eight years trying to find my way back to that, although largely wandered in the wilderness. I was a bartender, I taught LSAT test prep, I did a bunch of just odd jobs just to make ends meet.

**This is when [you started Unspam](https://www.unspam.com/about.html), right?**

**MP:** Yeah.

**And [you wrote a paper about the CAN-SPAM Act](https://repository.law.uic.edu/jitpl/vol22/iss1/3/), I think you were teaching a law course about that. What got you interested in spam of all things?**

**MP:** I thought it was an interesting legal question. It is with a very few exceptions, until the Internet comes along, there aren’t a lot of ways to be sitting in one place and commit a crime on the opposite side of the earth. There were some, you could do postal fraud and a couple of things, but all of a sudden that became a real thing that was happening at real scale.

And then secondly, it was the first time that you could be sitting in one place of the world, send an email to another place, potentially commit a crime and not even know you were doing it because you don’t have any sort of jurisdiction which is attached to it and so I thought that the question of how you were going to apply jurisdictional laws to things like cyberspace was a really interesting question. Again, I was playing law professor largely, I was an adjunct which was the lowest tier of professor at a law school that doesn’t exist anymore, that’s how bad it was.

**(laughing) Was this University of Chicago Illinois or…?**

**MP:** It’s not the University of Chicago, University of Chicago is terrific, it was John Marshall Law School in Chicago, which does not exist anymore. But it gave me an excuse to think about and write about things and I had a thing that was today we call blog but it was before that term had been even coined. I think the same way that you get to write and talk about things that you’re interested in, that was a period of my life that I got to do that. My parents were deeply worried that I would never make anything of myself and I honestly was a little bit worried about the same thing, but I think it was actually really important for just having the perspective to then go on and start Cloudflare.

**What was was Unspam specifically though?**

**MP:** Unspam initially started out as being a cool domain that I had registered and I was like, “What can I do with this?”, and the first thing we tried to do is disposable email addresses, the same thing Apple does now where you can basically have an email address and then if people start sending you stuff you don’t want, you turn the email address off and it goes away.

**I started doing this in college in the 1990s where I still have this domain that I use for all these email addresses, it turns out it’s actually a very difficult way to live.**

**MP:** (laughing) I have that same problem.

**There are so many assumptions built into having a common email address across different places that it’s actually pretty tough. I’m stuck with it now!**

**MP:** Yes. Well, we should commiserate over this at some point because I have the same problem. And also just explaining to someone and you’re talking to the customer service agent at Nordstrom, and they’re like, “Okay, what’s your email?”, and I’m like, “Well, it’s nordstrom…”

**Nordstrom@domain.com.**

**MP:** And they’re like, “No, no, no. What’s *your* email?”

**And they’re like, “Wait, do you work for us?”**

**MP:** And then, “This doesn’t make any sense.” So yes, I’ve had many of those. That’s where we started. We then got the Do Not Call list was coming along so the question was, “Could you build something like the Do Not Call list but for email?”, and so we ended up being strangely a government contractor where we worked with various states.

Originally, we started out actually with Chuck Schumer and the CAN-SPAM Act but the federal government never implemented it. But Michigan and Utah and a couple of other states implemented this thing that basically said if you put your email on a list, then they would maintain the list and then what we did was we just provided hashing technology where we’d hash the marketer’s list, we’d hash the government’s list, and we’d compare the two. So neither side knew what was on the others list and it wasn’t world-changing technology but the company is still around, it’s profitable but it was never going to be a really, really big business.

**You have an undergraduate degree in English, you go to law school, and then you decide, “Oh, I need to get an MBA as well”, you go to HBS. Did you have an idea that you want — was that a, “I don’t know what to do with my life so I’m going to go to Harvard?” — which is a lot of people do things like that. Was it a, “I specifically want to gain some sort of knowledge, I want to start a business?”, what’s the sequence of events here given that Cloudflare the idea started there in Cambridge?**

**MP:** I started Unspam and Unspam was floundering along and not doing very well. And my dad — we’d grown up upper-middle class and my dad owned a bunch of restaurants — not good restaurants — like Applebee’s, Famous Dave’s Bar-B-Que, he had the only Hooters in Utah.

**An adventure in its own right.**

**MP:** Yeah, and he was in his late 60s and he called me up and he said, “Son, I’ve taken care of you a lot of your life and it’s time for you to come run the family business”, and I couldn’t imagine anything worse than running my dad’s Hooters. So the idea of going to business school was, I was on my second bottle of wine trying to figure out, “How the hell am I going to get out of this?”, because it was a good business and someone needed to run it and I thought, “Well, I’ll apply to business school”. I applied to eight business schools, the night the applications were due, I ended up getting rejected from seven of them and then somehow got into HBS and called my dad and said, “Hey…”.

**“I really want to do it but I need to go get some training first!”**

**MP:** “I really wanted to do it but I need two years to try and do this business thing, I think I should understand accounting a little bit better”, and he was like, “Yeah, that’s a really good idea, you should do that”. So I went off and I had a Voice-over-IP phone that I was in the dorms, this tiny little dorm in Alston, in Cambridge, and I plugged my Voice-over-IP line in so I still was working for Unspam, the phone rang, I’d pick it up, and then I was doing business school stuff. I was much older than the usual business student and I taught law, which it was like this total unlock where I was like, “Oh my gosh, everyone does grad school wrong and especially business school wrong”.

**How so?**

**MP:** They think that the goal of the class is to think of the clever thing that no one has ever thought of before, which leads you down a bad path. Whereas really the goal if you’re faculty is to take students on a Socratic journey to make a point, and you start at point A and end in point B and you can figure out what that arc is. So the way that I approached business school was I would just say — whenever someone would have some crazy point, I just try and get us back on the arc, and that made me every faculty member’s closest friend.

**Especially with the Harvard teaching style with the cases and things like that where they’re trying to guide a discussion, it’s not necessarily a lecture, I can see that appealing for sure.**

**MP:** Yeah. It was funny, in almost every class because they’d get descriptions of who the students were, the faculty about three lectures in would come up to me and they’d say, “So you used to teach law school?”, and I was like, “Yeah”, and they’d say, “Nothing like a lawyer to ruin a good conversation”, and I said, “Yeah, that’s true”, he said, “Here’s the deal. I won’t embarrass you if you don’t embarrass me”, and I said, “Deal”. So business school at that point became just a lot of fun and I did pretty well at it. But mostly I was spending that time just trying to figure out what can I figure out to do that won’t have me go back and run my dad’s Hooters?

#### The Cloudflare Idea

**Where did the Cloudflare idea come from? Was that directly downstream from the Unspam work? You can certainly see the links there. What’s the connection to Clayton Christensen and disruption? [You’ve talked about that in retrospect](https://stratechery.com/2021/cloudflare-on-the-edge/) but it’s one of those things when you look backwards, you can paint this picture of I learned about disruption, I wanted to start this company that served an underserved market. Is that how it worked or what was the actual progression?**

**MP:** These things all start with people. There were two people who were critically important, one was an employee at Unspam, this guy named Lee Holloway who we’d hired straight out of college and was just — there are some people that are just real technical geniuses, and Lee was this incredible technical genius and we built the core technology for Unspam but we also had all these side projects.

One of those side projects was a thing called Project Honey Pot which actually built — [Paul Graham](https://x.com/paulg), the Y Combinator guy, before Paul did Y Combinator, he would host a conference at MIT called the MIT Anti-Spam Conference, and he invited me out one year to give a talk on how to write laws to throw spammers in jail, that went over pretty well. He’s like, “Come back, just do the same talk”, I’m like, “I’m not going to do the same talk, I’m the lawyer, I’m not going to do the same talk to a bunch of technologists, they were polite the first time”, and he’s like, “Oh, you’ll come up with something”.

So I went back to Lee and I was like, “Could we build a system to track how basically bad guys steal your email address?”, that turned into something called [Project Honey Pot](https://www.projecthoneypot.org/). Gave the talk, it was wildly popular, put it in the corner, and over the course of the next few years, over 100,000 people signed up for this service. And Lee was there, Lee at the time, while I’d gone off to business school, he continued to do technical work at Unspam. But it was not the highest, most interesting technical work, and he had called me and said, “Hey, you’ve always been really good to me but…”, and at that point in the conversation, I was like, “Stop right there, give me some time, I’ll figure something out”. Because Lee was one of those people you just wanted to have on your team, he had offers at Google and Facebook and that.

On the other side, a lot of people in business school are obnoxious, but there was this one woman who was really just genuinely trying to find the right answer and had no vanity about the whole thing, and that was [Michelle [Zatlyn\]](https://en.wikipedia.org/wiki/Michelle_Zatlyn). Michelle was clearly the opposite of me, and I’m not the most organized person, I’m not the most disciplined, I’m not about process. Michelle was a Six Sigma Black Belt and she’s just all of those things and with Unspam, I had started with two other friends and we fought like crazy and so I was really trying to look for people that had real difference from who I was, and Michelle was just this all of the things that I was bad at, she was amazing at. So I was always pitching her on various ideas.

Most of my ideas were — they were really, in retrospect, terrible ideas. But one of them I was telling her about Project Honey Pot, and she was just perplexed by the thing. She was like, “Why do people sign up for this?”, and I was like, “Because we track the bad guys”, and she’s like, “Yeah, but it takes effort, do you give them anything?”, and we’re like, “Well, they get recognition for what they’ve done”, she’s like, “That doesn’t make any sense, why would anyone do this?”. And I, frustrated at an Ethiopian restaurant in Central Square in Cambridge, I said, “Michelle, someday they want us to stop them”, and she said, “That’s it, that’s the idea, let’s build that”.

And literally that night I called Lee and I pitched, “Hey, here’s how it’s going to work, we’re going to do it this way”, and I spent 30 minutes just walking him through the entire idea that Michelle and I had sketched out. At the end of it, he stopped for about five minutes, to the point I thought the line had gone dead, and at the end of it he was like, “Yeah, that’ll work, let’s do that”, and that was the start.

So it was Lee, Michelle and I were the original three co-founders of the company, it started as a school project. The original idea was, “Could you take a firewall and put it in the cloud?”, so Cloudflare is us playing with firewall in the cloud. And for at least the first five years, what we kind of outlined over the course of the next few days turned out to be a almost perfect roadmap for what happened over the next five years.

**So you have this service that is super beneficial for tiny websites, you can sign up for Cloudflare for free to protect you from Distributed Denial of Service (DDoS) attacks. It has this really virtuous cycle where ISPs are grateful that you exist because you consume a lot of bad traffic, that they’re happy to have you on board, this gives you a better service. It’s really the growth story, [which I’ve written about](https://stratechery.com/2021/cloudflare-on-the-edge/), is really compelling and interesting.**

**And as you’ve articulated, there is this disruption angle where you’re approaching the problem from a completely different approach of say, a public cloud or certainly on-premise software. You couldn’t be more different than that. Is this a thing where you had the agreements with your professors? You’re not going to make a fool out of them. So did you walk in to [Professor Christensen](https://en.wikipedia.org/wiki/Clayton_Christensen) and be like, “Oh yeah, I’ve already done this”, or what is the connection there?**

**MP:** I was taking, I have to remember, I think I had just finished Clay’s course when we started working on Cloudflare so it was coming really fresh out of that and Clay and I had, I think, a really very strong relationship and I just loved the course and got to take it from him. Early on, I think that it gave us the confidence. We knew it’s really hard to build businesses unless they’re consumer, but even then it’s really hard, but it’s really hard to build big businesses in a B2B space on 20 bucks a month fees.

**There’s a reason everyone ends up getting sales forces and selling to big companies.**

**MP:** That’s right. So we knew the only way it would work is if we got to the point where we’re selling for tens or hundreds of millions of dollars, which we do today, to big enterprises. But the question is how do you get from here to there?

I think because of the nature that we had to build this network out because we needed data and because I’d come out of Clay’s class, I think that all gave us the confidence to say, “Okay, we’re going to start focusing on how do we provide a service for free that’ll be stripped down and limited in a bunch of ways, but make it better than anything else that’s out there and then continue to move up market”. And at some point, we would cross the point where the features that we had would be more than what people could get from the alternatives, which were companies like Akamai and the on-premise hardware folks. Then the hyperscalers were sort of emerging at that point and that’s exactly what happened.

It’s interesting today because at some level we just had the belief that if enough of the Internet flowed through us, we’d find some way to make a business around it and we really did believe in the mission of helping build a better Internet and I think those two things together have allowed us to have the success and get the scale that we have today.

**Is it ironic or maybe intentional that this sort of boutique interest you had in crimes being committed across the world, no longer limited by geography, basically you need some sort of push to get people into your business and so the criminals were actually your most important, they were your salespeople, they created the market for you?**

**MP:** Yeah, we’ve never thought about it that way, and there have been various times that there have been hackers who have been like, “I’m reformed, I want to get a job”, and we’ve been like, “Yeah, not so sure”.

But I think that the Internet is a complicated place. Early on people would say, “Who’s your competition?”, and I said, “Facebook”, and I actually still believe that’s really true. I think had Cloudflare not come to exist, much more of the Internet would’ve existed on Facebook because it was becoming too difficult as a small website owner to put up with the hassle of it and so people were racing behind the walled gardens that were out there. What we tried to do is say, “Hey, let’s give you still the freedom to innovate, but help you have the kind of protection that you don’t have to worry about all of the nasty things that are out there”, and so I think in the same way that Amazon is to Shopify, Facebook is to Cloudflare. Not a perfect analogy, but I think we try to be the minimum amount and then let just any type of creativity or uniqueness happen behind the scenes, whereas the walled gardens are a different thing.

#### Cloudflare Today

**What is Cloudflare today and how would you contrast that to the idea 10, 15 years ago? And is this a path that is about the path you expected to walk down with some obviously diverges along the way, or have you ended up in a different spot than you expected?**

**MP:** I think the real story of Cloudflare is that we wanted to put a firewall in the cloud, and we knew that in order to sell to JP Morgan and governments and big hospital groups and things like that, we had to start small and work our way up, so we made the service free.

What we didn’t expect was the number of problems that that would create. All the world humanitarian organizations were sort of our first customers because they had big security problems and small budgets, and so they all signed up and then people would try and DDoS attack them or hack them in various ways.

**Not ideal customers.**

**MP:** Well, again, in some way, perfectly ideal customers because we learned a ton, but then we would stop them. But then all the hacker kids are like, “Wow, these guys are pretty good, I’m going to sign up because all my friends are trying to hack me”. So then all the hackers are on us, but then the hackers try and push us in every possible way.

So why did we [build a domain name registrar](https://domains.cloudflare.com/)? It’s a terrible business, it is a commodity, why would we do it? Because we looked at everybody else and no one else was secure, and we came this close for cloudflare.com getting hacked, and that would’ve been a disaster. So we were like, “The only way we can solve this is to bring it in-house”. Why did we build our [VPN replacement series of products](https://www.cloudflare.com/zero-trust/solutions/vpn-replacement/)? Because we looked at all of the other folks that were in the space and we’re like, these guys, they just can’t deal with the scale and complexity of everything that we have, so we’ve just got to go build it ourselves. Why did we build [a developer platform](https://developers.cloudflare.com/)? It wasn’t because we thought we were going to sell it, we built it, we needed it ourselves, we needed a sandbox environment so we could build things.

Now, all of those things turned into substantial kind of product lines for us. But really Cloudflare is the story of we create a series of problems and then we solve them ourselves, and in the process that turns into products.

I would’ve never imagined — the very first meeting we had with venture capital, this guy named [Ray Rothrock](https://www.rayrothrock.com/) who was the first money into Cloudflare, and we pitched him and he says, “Great, I only have one question”, and we’re like, “Okay, what?”, he’s like, “What are you going to do about the death threats?”, and I was like, “Death threats?”, and Michelle was like, “What are you even talking about?”.

**(laughing) The uncertainty that the black belt six sigma does not want in their life.**

**MP:** And he’s like, “If you’re going to do this, you’re going to piss a whole bunch of people off”, and so I would’ve never imagined that I’d be personally sanctioned by the Russian government and supposedly on some Putin kill list, that wasn’t what I was signing up for and so I think that we have done something which is very different. I think because of the fact that we’ve made it available so broadly, we end up touching a bunch of content, that then means from time to time — although it’s interesting, you said the two times we’ve talked before were all about content moderation.

**I was going to ask you, is the topic dead? It doesn’t seem to come up much anymore.**

**MP:** Well, so we’re 15 years old in September. We’ve had three major incidents of this where there’s stuff that’s illegal, we take down all the time, but for the kind of stuff that’s immoral but not illegal, there’s this sort of gray line that’s there, three incidents over 15 years. So the mean time to incident is five years, we’re due for another one. There’s going to be something, I don’t know what it will be.

But inherently, if you sit behind enough of the Internet — right now, we’re being sued in Italy, Spain, a bunch of other places and it is largely because people who own football teams, soccer teams, they don’t like people illegally streaming their content. We don’t like people illegally streaming their content, it burns through a bunch of bandwidth. We try and shut it down, but they’re very clever, so they find different ways to do it. So we’re in these fights where we’re kind of like, we’re trying to not have this happen, and so I think that the nature of what we do is going to always have some controversy that’s around it. But frankly, that’s kind of what makes it so interesting as well because we do find ourselves in the middle of some really interesting policy debates.

**I do want to come back to that in a little bit, but one thing that I think defines Cloudflare, and this is always interesting, you have these product weeks where you have 47 announcements, it’s hard to parse them all. What I like about companies that do that, this was like Nvidia back in the day, they knew they had something with GPUs and CUDA and they would throw so much spaghetti against the wall, every single GTC, they’d have it twice a year and people would ask me, “How do they do this?”, [I’m like](https://stratechery.com/2021/nvidias-gtc-keynote-the-nvidia-stack-the-omniverse/), “Well, because it’s all the same platform, it’s just everything is sort of software defined, so they’re actually just releasing these libraries that are marginally different from each other, but they can frame them as being different things”.**

**That’s sort of like Cloudflare, you have this common architecture on commodity hardware. Was that very purposeful from the beginning or is that a similar function of, “We’re a scrappy little startup putting a server in an ISP, we can’t afford anything more than a Pentium or whatever it might be”, and it turns out that actually ended up being the way to build out infrastructure?**

**MP:** Yeah, I think we were really inspired by the Google story of, if you think of who was the leading search engine before Google launched? It was AltaVista. And who built AltaVista? Digital Equipment Corporation. What was Digital Equipment Corporation’s business? They sold mainframes. Why did they launch a search engine? It was a demo on how powerful their mainframes were.

And so then Larry [Page] and Sergey [Brin] come around as Stanford students, and the key innovation of Google isn’t actually search, it’s their ability to store and process data much more efficiently than anyone else, being cheaper and faster than anyone else, and that they could scale the mainframe to be infinitely large effectively and they did it by just taking a bunch of commodity hardware.

We, from the beginning I think, I was firmly in this camp, Lee was firmly in this camp. There’s some people on our team who were not firmly in this camp, and so there were fights early on about whether we should have specialized hardware to do various functions, or we should take commodity hardware and then write the clever software to do scheduling and everything else that you needed. Again, that’s one of those debates. We were eight people above a nail salon in Palo Alto, and it could have gone either way. If it had gone a different direction, we wouldn’t be the company we are today.

Today still, almost every server that makes up Cloudflare, is one of a different generation. I think we’re on generation 13 now and we’ll buy the servers in bulk. The servers now have five major components. So they’ve got network, CPU, GPU now, long-term storage and then short-term storage, so SSDs and then RAM. We’re constantly asking ourselves, and this is where a lot of innovation, there’s a once a quarter meeting that our infrastructure team and our product team have that they call a hot and cold meeting and they ask the question everyone asks, which is, “Where are we running hot?”, and then the question is, “Can we do engineering to make that more efficient or do we have to deploy more hardware in those places?”.

But they ask the equally important question of, “Where are we running cold?”, and that I think is the more interesting thing, which is, “Where are those opportunities where we’ve already paid for a resource, it exists in the wild, but it’s not being utilized enough, what can we go build on that resource?”. My somewhat distasteful maybe analogy to this is if somebody came up with the idea of selling the advertising space above urinals in bathrooms and bars, I can tell someone thought of doing that, it wasn’t worth anything, the minute someone did it, you’re all of a sudden generating a new stream of revenue from that.

Cloudflare is asking that question all the time, which is, “Where is that kind of extra space, extra capacity?”, and then if we can sell that, then we have higher margins and it allows us to continue to do all the things we do.

**What’s the split then of business that’s opportunistic versus very intentional and you’re pointing to it for a long time?**

**MP:** We are not the place where we come up with hundred page business cases. I mean, again, we built [Cloudflare Workers](https://workers.cloudflare.com/) because we needed it and now it’s the fastest growing part of our business.

#### Cloudflare’s Niche

**You’ve talked about being a fourth cloud in the past, [I wrote about that](https://stratechery.com/2021/cloudflares-disruption/), I feel like I haven’t heard as much about it as late. Did the public clouds respond faster to that than you expected, [whether it be Lambda](https://en.wikipedia.org/wiki/AWS_Lambda) and things like that or what?**

**MP:** I think what’s happened is that we’ve each carved out our lanes and while we compete at the margins, we are very distinct in those lanes.

The analogy is, I think that companies have personalities just like employees that do various jobs functions have. The hyperscalers, so AWS, Google, Microsoft Azure, their job function personality is the DBA, the database administrator. And if you’ve ever worked with DBAs, they are brilliant, but they’re oftentimes kind of rigid. They see the database as the center of the world, the entire thing they’re doing is all the data has to be in the database.

**To be fair, the database is kind of the center of the world, in their defense.**

**MP:** It can be. Again, that is one way you are showing your colors, I guess. The other version, which is more us, is the network administrators.

**It’s funny you mentioned that, actually for the business of Stratechery and Passport, the stuff that I work on, I think a lot about databases. It’s actually really important!**

**MP:** I think a lot about networks, you think a lot about databases. But the functions of those two things are actually really, they’re not diametrically opposed, but there’s tension between them, because the purpose of the database is to hold onto the data and store it no matter what. The purpose of the network is to move the data and get it off your system as fast as you possibly can and those are two very different things.

So if you’re really good at databases, you’re probably not going to be really good at networks and if you’re really good at networks, you’re probably not going to be really good at databases, and I think that’s true. I think that the networking products of the hyperscalers kind of suck because they don’t actually want data to leave and our database products, honestly, aren’t as good as the hyperscaler products. Again, they have certain advantages, there are times when you want to use them and you don’t want to use theirs, but they’re definitely tertiary to what it is that we are doing as a business.

I think that a bad outcome in the future for Cloudflare is one where the world becomes more unicloud where everyone says, “I am a hundred percent all in on AWS”, in that case, there’s not a lot for us to do. A good outcome for Cloudflare in the future is a world where you are more multi-cloud, in which case then the network becomes the things that rationalizes between the different cloud providers.

**Is this the area where AI has actually been positive for you? Not so much in building AI products, but in the fact that it’s given a motivation for companies to be more multi-cloud?**

**MP:** Yeah, I think that’s true. And I think also, yeah, and you’ve got big data sets that you have to move around in ways, we’re really good at that. There’s just a lot of things where for one reason or another — if you’re building an agent today, probably Cloudflare Workers is the best place to build it. You can spin it up, you can spin it down, it can connect all around. It’s got to pass through us anyway because we’re such a large portion of the Internet and so that has been, I think that actually the sort of AI and AI agents has been the killer app for Cloudflare Workers, but it’s less about how — we’re never going to be the right place to run [SAP HANA](https://en.wikipedia.org/wiki/SAP_HANA), but we’re definitely the right place to run the agents that have to interact with the rest of and move around the Internet.

#### Pay-per-crawl

**The reason to talk now, and we’ve talked offline about this a few times, both this year and last year, is your push for this [pay-per-crawl concept](https://stratechery.com/2025/cloudflares-content-independence-day-googles-advantage-monetizing-ai/). Why don’t you give me the high level overview, the pitch from your perspective, which I think has evolved? I would like to think partially based on some of my feedback, but what’s the pitch in September 2025?**

**MP:** Let’s take Cloudflare out for a second and just talk about—

**Talk about Matthew, the English student? The student newspaper editor.**

**MP:** This is me channeling inner law professor. Let me give you the history of the Internet and why the Internet exists the way that it does and what’s changing.

**This is usually my job, but go ahead.**

**MP:** And you can tell me where I’m wrong, but this is my quick history of the Internet, and apologies to Michelle who hates history lessons.

For the last 25 years, the interface of the Internet has been search, and Google has dominated that space, and Google, their incentives as a company were to have the Internet grow as much as possible because if you have chaos, then the search becomes the organizer of the chaos. But you need incentives for people to actually create content and so Google not only had to create the thing that organized the Internet, but they then had to take the thing that took the traffic of where people went and then helped people monetize that, largely through advertising, although they also helped with subscriptions, and Google was the great patron of the Internet for the last 25 years. The web would not exist the way it does if there were not something like Google out there to create the incentives around.

There were a lot of problems with incentivizing around traffic, we created systems where people would just literally try and create rage-baity headlines to get people to click on things so that they could put ads against them and so not perfect, but we don’t have the Internet that we have today unless we have Google and search funding that.

That is changing. The world is shifting where the interface of the web is shifting from search engines and search engines give you a treasure map and say, “Hey, go figure out what your answer is by clicking on these 10 blue links”, to what are effectively answer engines. So if you look at OpenAI, if you look at Anthropic, if you look at Perplexity, even if you look at modern Google, they are not a search engine, they don’t give you a treasure map. Instead, they give you an answer right at the top of that page. That answer, for most users, 95% of the users, 95% of the time, it’s a better user interface. I’m not anti-answer engines, I’m not anti-AI, I think it’s better in every possible way for that to be what the interface is that we all interact with.

But the problem is that if you get the answer and you don’t get a treasure map, then you don’t generate traffic and if you don’t generate traffic, then the entire business model of the web, which has been based on traffic starts to break down and you can see that, not so much in e-commerce sites, not so much in things that actually sell you the physical thing because if you asked what’s the best camera to buy, even if you get an answer, you’ve still got to go buy it from somewhere. It’s going to take the e-commerce and the people who are selling things that’s going to work but the person who wrote the review—

**The great thing about physical products is by definition they are scarce and the problem with text on the Internet is it is not scarce.**

**MP:** It’s not scarce, that’s exactly right, and Google set this expectation that everybody can scrape the Internet for free, but it was never free. The Internet has never been free. Google paid for it for a really long time and the quid pro quo with the content creators was, “We get a copy of your content and in exchange we’ll send you traffic and help you monetize that traffic”.

That quid pro quo breaks down as we shift from search engines to answer engines and so something is going to change. I see three possible outcomes for that. And again, none of this involves — if Cloudflare disappeared tomorrow, this is still happening, one of these three things will happen. One, all of the journalists, academics, and researchers in the world will starve to death and die. And it’s crazy, like when you post this stuff on Twitter, how many people were like, “Well, we don’t really need journalists anymore, we have drones”, and I’m like, “I think we still need journalists”.

**This is the point that I make a lot because people will hold up Stratechery as a model for journalism in the future, and subscriptions and the whole small scale subscription model. I’m very proud of Stratechery’s role in figuring that out, but I’ve always said, look, my Articles are generally completely fresh content, but my Updates, how do I always open them? By quoting from a journalist like, “Okay, now I’m going to analyze this piece of news that someone actually collected”, so I’m with you on that point.**

**MP:** And how would you know what to write about, you’re derivative of those things. That’s great, and I think that adds an enormous value, and I think as you’ve said in the past, there will still be tentpole content that’s out there that people will have to have. But there’s a whole bunch of other stuff like what happened in the world that somebody has to report on, and again, there is a real risk that if the business model completely dies, that that goes away, and I think that would be a loss. I don’t think that’s going to be the outcome.

**Don’t you think that that’s already happened to a certain extent though? People worry, complain about there’s lots of national coverage, but say on a local basis, the local newspapers being decimated by the Internet and don’t make any money, don’t have reporters, who knows the City Hall is doing, et cetera.**

**MP:** I think that that’s the Google model of paying for traffic doesn’t support local media in the same way. My wife and I bought the local newspaper in our hometown, it actually turns out to be a pretty good little business, surprisingly.

**[I’m a big advocate](https://stratechery.com/2017/the-local-news-business-model/). I think that local news actually could be.**

**MP:** Well, I think it’s going to become massively more valuable and so we’ll get to that. The second possible outcome —

**Sorry, you didn’t make the agreement that you made with your professors, you had to tell me to let you cook and not keep interrupting you and being a pain. Sorry, continue.**

**MP:** No, it’s fine.

**Sorry, continue.**

**MP:** It’s your newsletter and show.

**Continue, continue.**

**MP:** I’m just the entertainment! The second option is that — and I don’t think this is outlandish, but I think it is a Black Mirror option, which is that, can you imagine Sam Altman announces tomorrow he’s standing up his own version of the Associated Press?

**Yes, I can.**

**MP:** See what I’m saying, can you imagine he announces he’s bought Gartner? It’s not insane. It could be that we don’t go back to a media time of the 1900s, we go back to a media time of the 1400s, and it’s like the Medici’s, you’ve got five powerful families that control and pay for all of the different content creation which is out there, in which case it is likely that there’s a conservative one, there’s a liberal one, there’s a Chinese one, there will be an Indian one. The Europeans will try to build one and then it won’t work, and they’ll use the liberal US one and that could very well be what happens with content.

I think that that’s not inconceivable, and it’s not actually that many far degrees away from what Scale AI was doing where you’re sorting and organizing content for people and tokenizing it. There are a lot of unemployed journalists, it’s not that expensive to stand up a bunch of bureaus, but I think that’s a really dangerous outcome because it’s incredibly regressive where the Internet was this great leveler, this great distributor of knowledge. If all of a sudden we’re all spending thousands of dollars a year on whichever AI system, even the wealthy will probably pick one, and if the knowledge that you get all comes from that one, all of a sudden we’ve siloed things again and that seems really, really risky, but not totally inconceivable.

**What’s number three?**

**MP:** Number three is we figure out a new business model, and it has to be some version of the revenue that is generated by the AI companies, the answer engines that are out there, some portion of it goes back to the content creators. And how that works again, I think is what we’re trying to figure out.

But I think that there are really three things today that you need to be an AI company. You need access to GPUs and OpenAI spends over $10 billion a year on that, but it’s silicon and like all silicon over time it will become more and more of a commodity, it just will because there’s never been a silicon shortage that hasn’t turned into a silicon glut. Everyone in the world is racing to compete with Nvidia, Nvidia might stay in the lead for a really long time, but you’re going to have AMD and Qualcomm and Apple and all the hyperscalers come up with new things.

**We just need to keep Taiwan intact.**

**MP:** We do need to keep Taiwan intact. And you moving, you were the front line of defense as far as I heard.

**[Family reasons…](https://stratechery.com/2025/a-personal-update-and-vacation-break/)**

**MP:** The only thing that kept the Chinese away was Ben.

The second thing is talent — today we have an enormous shortage because frankly, if you were getting a PhD in AI five years ago, you were a laughingstock. It was a dead space. Today it’s the hottest space, everyone’s turning up a new thing and so we’re going to go probably not from a shortage to a glut, but you’re certainly not — the days of a billion dollars to come work at Meta if you’re a great AI researcher, those days are limited and it’s not going to be that way forever. We’re going to have a lot more people who are flocking to this because education markets and employment markets are efficient.

The third thing that you need is content. And again, we have this expectation because of Google, that content should be available to all the robots for free, but that’s not sustainable. You end up in either everyone dies of hunger or we live in the time of the Medicis in those models, so in some way we’ve got to figure out a new quid pro quo.

You could think about it as $1 per monthly active user per year for any of the AI companies goes into a pool that then gets distributed to content creators. If you do the math, that’s about $10 billion today in terms of content, that would entirely replace all of the ad revenue that is generated by the non-walled garden Internet today, so take Instagram and Facebook and TikTok and those-

**Like the Google networks, TradeDesk, etc.**

**MP:** But include the Wall Street Journal, The New York Times, the FT, Reddit, everything else, it’s about $10 billion a year. That’s a lot of money, but that’s not a crazy amount of money. And if we can figure that out, we can actually then create a better business model, which if we do it right, I think it actually encourages better media and better content to be created at the same time.

**You talked about earlier the question of getting from here to there, which is certainly the question that applies here. There’s also the issue of what we want to happen versus fighting economics and the fact that text is infinitely replicable and can be spread and you put years into worrying about spam, spam is still a problem.**

**MP:** Sort of. Not the problem it used to be. Honestly, if you’re — I remember when Bill Gates was like, “We’re going to solve spam in the next five years”, and I was like, “No, you’re not, this is an intractable problem”. Actually, they largely did.

**It all comes out on the backs of people like me just trying to send an honest newsletter to people’s inboxes.**

**MP:** It makes it harder for you, yes.

**When you think about this and what you can do, how do you separate the, “Okay, this is what I want to happen”, versus, “This is what is possible to happen and there are actually levers that I can pull to make it happen”?**

**MP:** I think the first thing is that all markets require scarcity, you can’t have a market if you don’t have scarcity, there has to be some scarcity that exists in that market.

**But that’s the problem with the Internet is 45% of the people might decide to buy into this concept, but then the 55% just run the table.**

**MP:** Potentially, but if they don’t have access to the content, then it becomes harder. And so what we’ve just seen is that for content creators that are out there, especially if they create scarcity, they actually are able then to do deals.

So you look at Reddit. Reddit has been very aggressive at stopping bots from scraping their content, including Google. They said, “Google, you don’t get access to this unless you pay for it”, and as a result Google and OpenAI struck a deal with Reddit where they pay — we know from their public filings that this isn’t anything that is confidential — but in 2024 they got $120 million for 2024 and I’m told that the deal in 2025 was even better. I think that you’ll see that going up.

What I think is actually really interesting is if you count up the number of tokens in Reddit and the number of tokens in the total catalog of The New York Times, it turns out about the same within an order of magnitude, and yet Reddit got seven times as much. The question is why? And I think answer is because The New York Times — you might love The New York Times, but the difference between The New York Times, the Wall Street Journal, the FT, the Washington Post, and the Boston Globe from the perspective of an LLM is actually not that much. The facts are the facts are the facts. And so yes, there’s some color and there’s different slants, and the opinion pages might be slightly different, but it’s kind of the same.

So I actually think the thing that’s really encouraging is we already see that where scarcity exists and where content is unique, like if you don’t have Reddit, you don’t have Reddit, so you need to have that content. I think that if we can create scarcity in these cases, what the market will show is that it’s actually that local, unique, differentiated content, which is the most valuable and as this market for content exists and comes on to being, again, regardless of Cloudflare, if Cloudflare disappears, I still think that it is inevitable that publishers will say, “Okay, we’re going to shut down the access of the bots to have access to our stuff, and then we’re going to create a market for the most unique stuff”. And the more unique it is, the more you’ll get paid for it, I think that that’s just inevitable.

I think that this is one of those seminal moments where we’re shifting the business model of the web and I’m actually super encouraged that the future of content is going to be much more like early Internet than, and again, with all due respect to my friend Ben Smith, BuzzFeed.

#### Cloudflare’s Power

**If it’s inevitable though, then why does Cloudflare need to be so aggressive? You’re instituting these policies of doing your best to block bots, putting together protocols for recognizing what it’s worth, payments, etc., all very nascent to be sure, a lot to be figured out. But you are not taking the posture of a company that this is inevitable and it’s going to be great, you are being pretty forceful in trying to make something happen.**

**MP:** Well, I think if we weren’t doing it, someone else would. But what I think we have a unique ability to do is we’re really good at stopping things like bots because we do it every day.

So again, it wasn’t like we were sitting around being like, “Hey, what should we do next? Let’s go change the business model of the web”, it was our customers who were publishers were coming to us being like, “We’re dying and we don’t have the technical wherewithal to step in front of it, but we need to stop this, please help”. And honestly, when [Neil [Vogel\]](https://www.iac.com/business-management/neil-vogel) at Dotdash Meredith was telling me this, I rolled my eyes and I was like, “Publishers, they’re such Luddites, they’re always complaining about the new technology, they’re always complaining about the next thing, this isn’t a big deal”. And Neil and a bunch of others finally said, “Just go pull the data”, and it was only when we actually saw the data, when we saw that over the course of the last 10 years, it’s become 10 times harder to get a click from Google for the same amount of content on that same kind of basis, it’s now 750 times harder with OpenAI, it’s 30,000 times harder with Anthropic.

The business of traffic on the Internet as being the currency is going away and so something either again, either content creation is going to die, it’s going to become futile, or we’ve got to create a new business model. Again, if our mission is to help build a better Internet, this seems squarely in the line with what we should be working on.

**So [why does Garry Tan say](https://x.com/garrytan/status/1961115612996145381) that you are an axis of evil with Browserbase and you should legalize AI agents?**

**MP:** I really don’t understand. I mean, I’m confused by Garry, I think part of it might be that he’s an investor in Perplexity.

Every story needs four characters, you need to have a victim, you need to have a villain, you need to have a hero, and you need to have the village idiot or the stooge. And if you think about it, any news story has those four characters. Right now, the people who have most been the villains [have been Perplexity](https://blog.cloudflare.com/perplexity-is-using-stealth-undeclared-crawlers-to-evade-website-no-crawl-directives/), where they’re doing just actively nefarious things in order to try and get around content company.

I’ll give you an example of something that we’ve seen them do, which is that if they’re blocked from getting the content of an article, they’ll actually, they’ll query against services like Trade Desk, which is an ad serving service and Trade Desk will provide them the headline of the article and they’ll provide them a rough description of what the article is about. They will take those two things and they will then make up the content of the article and publish it as if it was fact for, “This was published by this author at this time”.

So you can imagine if Perplexity couldn’t get to Stratechery content, they would say, “Oh, Ben Thompson wrote about this”, and then they would just make something up about it and they put your name along it. Forget copyright, that’s fraud, just straight up and that’s the sort of bad behavior of some tech companies that again, I think needs to be called out and punished.

#### The Google Problem

**You go back to 2024 though, so I think, I don’t remember if it was you or someone else calling out Perplexity for ignoring do not crawl things on websites, and people got really up in arms about it. [My pushback at the time was](https://stratechery.com/2024/perplexity-and-robots-txt-perplexitys-defense-google-and-competition/), “Do we want Google competition or not?”, because what Google is doing is yes, OpenAI pioneered their own web bot and protocol and you can ask OpenAI to not crawl your website and they respect it. Google did the same, but the Google same one is Googlebot-extended, the actual Googlebot is like, “Well, you want to be in search”, and it turns out AI Overviews is governed by Googlebot. So why does Google get to do it and Perplexity can’t?**

**MP:** I think we should have an open bet, and I’ll take whatever bet you want. It can be a basketball game at some point. It can be whatever is is, that 12 months from now, will Google provide publishers a way of opting out of AI Overviews? And I’ll say yes, and I think you’ll say no.

**I don’t know, I have to think about this, I’m not committed, I’m not committing yet.**

**MP:** Okay. But I agree, and you wrote about it, is Google is the problem. We were very careful when we did this. We actually blocked training, which is what they use Gemini for and so we can block that and we’ve blocked that across the place. We have not blocked RAG, we’ve not blocked search for anyone including Perplexity, even though they are doing naughty, naughty things. And by the way, OpenAI is evidence that you can do the right thing, you can be good actors and you can still have actually a better product as a result, and be a viable competitor to Google.

So I think that Google will, and I am hopeful that they will, because they are a company that has really believed in the ecosystem, and they see what’s happening in the ecosystem and they understand that if something doesn’t change, that the ecosystem is going to suffer. I think that they will voluntarily give publishers a way out of AI Overviews within the next 12 months. I’m actually optimistic it will happen faster than that and if they don’t do it voluntarily, I know that there are a number of regulators that would be happy to force them to do it.

**Do you feel encouraged or discouraged by [the Google case this week](https://stratechery.com/2025/google-remedy-decision-reasonable-remedies-the-google-patronage-network/)?**

**MP:** Well, so first of all, it’s like 280 pages long.

**It was really long, I was up until 5:00 AM, I had to read through the whole thing.**

**MP:** I’ve read summaries and I’ve had my team who are going through it. So I’m not sure that forcing them to sell Chrome is the right remedy.

**No, I thought that it was dumb. I think the big thing is, my whole thing with Google is [they’ve basically paid off everyone](https://stratechery.com/2024/friendly-google-and-enemy-remedies/), and then the justification for allowing them to continue payments is that everyone’s dependent on Google payments. It’s like this circular justification, which isn’t wrong, but that’s basically the nut of it.**

**MP:** Again, I’m not an expert on the — as we’re recording this, that came out yesterday, and so I’m not an expert on it yet — the thing that worries me the most is that some of what the judge suggested was that Google had to share data with the rest of the ecosystem, and so a horrible outcome for this that will actually destroy the ecosystem, and the most depressing sentence in the ruling that I’ve seen so far was the one that was like, “It seems like this might have bad impacts on the publishing industry, but not a single publisher testified”, and I’m like, “Oh guys, you got to get on the field if you’re going to win the game”.

So I do think that regardless of what Cloudflare is doing, the economics of the Internet are changing, they’re changing because there is a better interface that is coming along that is being provided by a lot of people including Google and if the ecosystem is going to thrive and survive, we have to change the economics of the compensation to content creators. We can do that in a way which I think is very reasonable, and my prediction will be that over time, AI companies will look more like YouTube or Netflix, where they compete on what original content they have access to as opposed to the algorithms. We already see the algorithms are largely commodities, they’re leapfrogging each other all the time, I think that’s going to continue for quite some time. I am in the there will be thousands of different AI interfaces we’ll pay for or they’ll be ad supported in various ways, but what will cause you to choose one versus the other is whether or not they have access to whatever it is that you particularly care about.

**I do think, by the way, I think I agree with your bet just because the core Google strategy that paid off massively in this case is basically paying everyone off.**

**MP:** That’s right.

**So that’s why this is a scheme to pay everyone off.**

**MP:** Exactly. Now I think it’s really important — so again, I don’t know exactly what this is — but we have to make sure that whatever the payment scheme is that it takes into account that if you’re smaller, you pay less, and if you’re bigger, you pay more. So again, my straw man on this is $1 per year per MAU, and you pay it into a pool and then how we distribute it out to the content creators, that’s a whole other can of worms.

**Potentially a very profitable can of worms if someone can sit in the middle and be the market maker.**

**MP:** Potentially.

**Could that be Cloudflare?**

**MP:** Whether we’re that or not, again, if it’s profitable, there’ll be a lot of people that will be competing to do that and if we can add value, then we’ll capture some of the value. But I think that the most important thing is how do we figure out some way where the vast majority of that is actually going to the people that are doing the work of creating content? And ideally, do it in a way where what we’re rewarding is not who is rage baiting people, but who is actually filling in human knowledge.

#### Cloudflare’s Motivation

**[I wrote this a few months ago](https://stratechery.com/2025/the-agentic-web-and-original-sin/), you know we’re on the same page, it’s just what’s striking about you doing this is like, well, maybe someone has to actually break some eggs to make it happen.**

**MP:** Of course.

**But this leads to my final question that ties together when we talked about content moderation and now we’re talking about this, is there a dichotomy between the power you’re seeking to leverage to push through this new standard for content, and then the power you claim to not want when it comes to content moderation? “We don’t want to be involved”, well, do you want to have power or do you not?**

**MP:** We still want to stop when cyber attacks are coming. That’s where we built a fairly valuable business in having the power of doing that. I think where we get nervous is in saying, is this content good or bad?

And so for instance, I think that the right way to figure out who gets paid as content providers is not Cloudflare coming up with an algorithm that says, “This is important content, you should have it and this is unimportant content, you shouldn’t”. Each AI company should come up with their own algorithm and be able to plug it into, whether it’s our system or our competitors, whoever it is, and say, “Here’s our system, you see content, rank it for us based on our algorithm and OpenAI’s rank will be different than Anthropic’s, will be different than Perplexity’s”. You should have that ability and that ranking should I think be scored on two different axis, which is how reputable is this, and then how novel is it? How much does it actually further that? If you do that correctly and you have a real variety of different — if you have tons of different AI companies and you have tons of different content creators, then again, I think sure, we’re playing the role of being the technical facilitator there, but the actual decisions on what content matters and doesn’t, that still is something that, again, I don’t think it’s the right place for us to be making those decisions.

**You want to be a market maker, not a picker.**

**MP:** We’re not an editor.

**That’s the word.**

**MP:** Yeah. Again, I think the one thing that we are very good at right now is creating scarcity and every market depends on that. Whether we or someone else then figures out how to monetize that scarcity, it’s fine and by the way, I think the vast majority of this will be large publishers doing direct deals with large AI companies.

**But that sucks for all the small guys.**

**MP:** That’s where again, we can be kind of a piece of it where we say, “Okay, we now on behalf of the small guys, let’s work with you to figure out how you can participate in this market as well”. Small guys on both sides — small AI companies, startups. We have to make sure that as we design this, that it’s working in this way, and then small content creators.

I guess my ask would be, since you have a relatively influential list, if you’re an academic and you’re reading Ben’s stuff and you think that this is really interesting, the reason Google ended up figuring this out, and I think being a great force of good in the world, Google did a lot of good in the world. The reason they did that is they thought very carefully about how these market dynamics are and so we’re trying to work with the leading academic economists and market theoricians to figure out, “How should this market look like going forward?”, and so reach out if you’re reading this because we’d love to talk to you about it.

**We spent a lot of time at the beginning — we’ve gone a little long, I appreciate you sticking on — talking about your background and how you got to where you were, it’s really interesting. It’s not the traditional tech path, is that actually integral to this final topic? If you are just a traditional, came up through computers, went to Stanford, started a company, would you be picking this fight, or is this downstream from teaching out of law school that doesn’t even exist anymore, just to get away from Hooters?**

**MP:** Just to get away from running my dad’s Hooters.

I don’t know. I think that the most valuable — so I have an English major, I have a computer science minor, I’ve got a law degree, I’ve got a business school degree, there were lots of times where I thought those three years at law school where completely wasted. It is incredibly valuable. I can sit and read the Google judge’s ruling and understand what’s important and what’s not in that, those things have been incredibly important. I think that the ability to speak and write and communicate and appreciate how much work creating great content is, I think that that’s been really helpful. I think I’ve tried, when we’ve talked about content moderation in the past, we struggle with those issues and struggle with what the right thing to do is.

**It went back to what side of the bed did you wake up on, right?**

**MP:** Yeah, totally.

**(laughing) [That’s a reference](https://gizmodo.com/cloudflare-ceo-on-terminating-service-to-neo-nazi-site-1797915295) for people that don’t remember previous controversies.**

**MP:** How angry your girlfriend at the time, wife now, is at you at that particular moment.

But going back and literally reading Aristotle, I think that that’s great. And so today I get invited, there’s a handful of AI companies that have invited me to be on their board and things, and I always say no. But I talk to them and I must be one of the top non-academic buyers of Aristotle’s [Politics](https://en.wikipedia.org/wiki/Politics_(Aristotle)), because almost every AI CEO has gotten a signed copy from me saying, “What you’re doing is incredibly important, but you’ve got to think about the ethics and how to create trust”.

I do think that, I hope that the people that are building these incredible systems that are going to be incredibly transformative, that they do spend some time in the liberal arts, that they do spend some time actually stopping and reading and thinking about like, “Okay, if we’re successful” — I think there’s been a lot of, “If we’re successful, it’s some version of James Cameron and Terminator”. But I think there’s another version which is, “Okay, if we build these powerful systems, how do we make sure that they’re trustworthy over time?”, and those are issues that people have struggled with for a long time and I feel very lucky to have gotten the time to sit and think about them and I do think that that’s been helpful and instructive to the company that Cloudflare is today.

**Well, I feel happy and lucky to have had your time in this interview. Thanks for coming on.**

**MP:** Ben, thanks for having me.

------

This Daily Update Interview is also available as a podcast. To receive it in your podcast player, [visit Stratechery](https://stratechery.passport.online/member).

The Daily Update is intended for a single recipient, but occasional forwarding is totally fine! If you would like to order multiple subscriptions for your team with a group discount (minimum 5), please contact me directly.

Thanks for being a supporter, and have a great day!