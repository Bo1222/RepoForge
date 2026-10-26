# Agent 实习简历调研：阶段记录（受阻，未完成）

截至 2026-10-21，**满足“小红书博主公开提供、Agent 实习相关、项目栏可读取”的独立简历为 0/10 份**。下面是候选与项目线索，不是已完成的十份简历集合。不将其他平台简历、招聘信息、工具项目、教程或二次整理冒充原帖简历。

本轮沿用用户目标：先搜集至少十份参考简历并逐份提取项目介绍，再研究本项目表达。尚未进入最终简历撰写阶段。

第三个连续目标回合已重新核验同一来源访问障碍：公开搜索页仍是 HTTP 200、结果为空，当前没有小红书连接工具，也没有新增可读种子来源。任务需要可读分享链接、原帖截图或可用数据连接才能继续补齐样本，故记为受阻；两份开源模板保留为补充，不替代十份目标。见[第三轮访问检查](third-turn-access-check.json)与[阻塞审计](blocked-audit.json)。

## 来源与环境

- 工作区：`/mnt/data2/home/geqb/LLM/moatless-tools`。
- HEAD：`2268485edb612a54e74f9c64cccdf72c84b4c4ea`。仓库已有未提交修改，本轮没有修改业务实现，也没有提交或推送。
- 来源仓库：origin `https://github.com/Bo1222/IssueAgent.git`；upstream `https://github.com/aorwall/moatless-tools.git`。
- 主要工具：公开网页检索、GitHub connector、Python urllib 公开页面读取。没有登录小红书、购买内容、私信作者或发布内容。

## 已发现且可继续跟进的来源

| ID | 作者/署名 | 内容与链接 | 当前能读到什么 | 不计入十份的原因 |
|---|---|---|---|---|
| C01 | 确实👍 | [原帖](https://www.xiaohongshu.com/explore/695533c3000000002103cef4)；[二次整理](https://post.smzdm.com/p/adog635d/) | LangGraph 简历生成工具的项目描述 | 是“生成简历的 Agent”，没有取得“投 Agent 岗的简历”；原帖当前无法浏览 |
| C02 | 确实👍 | [原帖](https://www.xiaohongshu.com/explore/699b5e6a00000000150398d4)；[二次整理](https://post.smzdm.com/p/aggkne0m/) | 医疗合规审计 RAG/Skill 项目介绍 | 未取得简历；与 C01 同一作者，不能据此计算两份独立简历 |
| C03 | 哦耶大师（转载署名）/ WufeiHalf | [原帖](https://www.xiaohongshu.com/explore/699fe5600000000015033347)；[二次整理](https://post.smzdm.com/p/a7090mz5/)；[开源项目](https://github.com/WufeiHalf/cn-resume-optimizer) | 简历优化工具链 README | 没有面向 Agent 实习的简历样本；账号关联仅据转载 |
| C04 | 淮南 / HuaiNan54321 | [作者公开项目](https://github.com/HuaiNan54321/ecom-service-agent) | 电商客服 Agent 的功能与架构介绍，README 自述在小红书同步分享 | 没有作者简历的项目栏 |
| C05 | 一只小阿鱼🐟 | [原帖](https://www.xiaohongshu.com/explore/69be8d73000000021a02907c)；[二次整理](https://post.smzdm.com/p/arz4ede7/) | 转行与项目准备建议 | 没有具体简历，示例含 xx 占位内容 |
| C06 | sxhmzl | [原帖](https://www.xiaohongshu.com/explore/697312df000000001a02a85a)；[二次整理](https://post.smzdm.com/p/akovx69e/) | 大模型实习学习、面试路线 | 没有可提取的具体项目栏 |
| C07 | 原帖作者未知 | [存档工具 README](https://github.com/firmamento73/xhs-note-archiver) | 目录示例里有“保姆级零基础找Agent开发实习怎么优化简历项目”标题 | 仓库没有该笔记正文或图片，完整原帖链接未知 |

小红书链接来自实际读取到的公开转载页 href，不是用猜测的 note ID 合成。没有附带有效分享参数，能否在用户 App 中打开需要另行核实。转载页标注 AI 生成的内容只作为低等级线索。

## 已能提取的项目介绍

以下均为依据标明来源所做的概括，**不是简历原文摘录**，也没有验证作者实际工程结果。

### C01：岗位定制简历生成工具

据[二次整理](https://post.smzdm.com/p/adog635d/)，用岗位描述与个人资料驱动 LangGraph 流程，通过 RAG 读取 JSON 资料，输出 YAML 后由 RenderCV 生成 PDF。描述了 MVP 跑通和虚构技能、夸大工龄的问题；审核 Agent、PDF 和代码库接入属于后续计划。

可参考的表达：输入、编排、资料读取、交付物、已知失败和后续计划。不能把“计划引入审核 Agent”改写成“已实现多 Agent 审核”。

### C02：医疗合规审计 RAG/Skill

据[二次整理](https://post.smzdm.com/p/aggkne0m/)，针对审计长文档、跨页表格和决策树图设计检索处理，用父页面索引与模型重排改善检索，并采用 Skill 批处理，让非技术人员能够维护规则。没有取得可核查的性能指标。

可参考的表达：文档难点、针对性机制、架构取舍。该描述选择的是 Skill，不能为迎合岗位而写成多 Agent 系统。

### C03：中文技术简历优化工具链

[仓库 README](https://github.com/WufeiHalf/cn-resume-optimizer)描述 PDF 提取、简历/JD 结构化、Rubric 评估，输出 Markdown 优化稿和报告，可选导出 PDF。强调本地处理、规则模板和可审计流程。未运行仓库代码，未验证处理效果或隐私边界。

可参考的表达：流程和可检查产物。该工具服务于简历制作，本身不是求职者的 Agent 实习简历样本。

### C04：电商客服 Agent

[作者 README](https://github.com/HuaiNan54321/ecom-service-agent)描述订单、物流、推荐、退换货等场景，提供 ReAct 工具调用、MCP、政策 FAQ 检索、售前售后分流、短长期记忆、按需 Skill，以及沙箱重跑和过程/结果双层评估。

可参考的表达：业务场景、工具协议、记忆机制、评测方式。这里的依据是 README 功能说明，不代表已独立验证生产使用或效果。不能把 README 的生产篇规划全部写成已交付能力。

## 第二轮新增：博主公开的完整简历模板

发现“阿东玩AI”公开维护的 [LLM-Resume-Template](https://github.com/adongwanai/LLM-Resume-Template)，取得两份完整 LaTeX 模板并提取五个项目段，见[来源、许可、模板与项目介绍](supplemental/README.md)。同账号的求职工具 README 自述小红书账号名，但对应发布帖尚未核实，因此该补充不计入十份小红书原帖简历。

两份模板分别是 Agent / LLM 匿名示例和大模型算法占位模板。五个项目段是长程 Agent Harness、SFT/RL 轨迹流水线、业务分析 Agent、自纠错 RAG、分层记忆对话 Agent。它们来自同一博主，不能宣称是五份简历或两名真实求职者的履历。模板原文件按 Git blob 校验通过，已记录固定源 commit 和 CC BY 4.0 署名。

## 工具证据与访问边界

完整结构化候选见 [sources.json](sources.json)，页面探测见 [access-probes.json](access-probes.json)，关键动作见 [tool-trace.md](tool-trace.md)。

已验证的访问结果：

1. 内置网页工具无法直接读取小红书搜索页；插件目录以“小红书”和“Xiaohongshu RedNote”检索均未返回可用插件。
2. 默认沙箱内 urllib 读取公开页面报 `Operation not permitted`。之后经工具自动审核放行的公开网络读取可以获取 HTML；这只是解决网络访问，不等于取得站内结果。
3. 小红书搜索页返回 HTTP 200，但页面初始状态 `search.feeds=[]`、`loggedIn=false`。**空结果不能证明小红书上没有相关简历。**
4. C01 原帖被跳转到页面不可浏览提示，`error_code=300031`，`noteDetailMap` 为空。不能据此断言作者删帖。
5. 五条转载页均可读取，已保存页面标题、原帖 href、读取时间和 HTML SHA-256；没有保存账号凭据或整页内容。
6. 第二轮直接读取 C02、C03、C05、C06 四条原帖，均返回 `error_code=300031` 和“当前笔记暂时无法浏览”，`noteDetailMap` 为空，详见 [followup-probes.json](followup-probes.json)。访问障碍仍存在；不能据此认定笔记被删除或不存在。

网页工具曾对部分转载页返回 Cache miss/Timeout；后续 urllib 成功读取，保留两种结果，不把初次失败改写为一次成功。

## 完成条件与后续工作

- 至少十份独立参考简历；同一简历的不同截图、改稿和转载不重复计数。
- 每份至少记录发布者、原帖链接、可读的简历项目栏及证据形式。
- 提取项目背景、职责、关键技术机制、结果/指标；看不清或没有写的内容标记未知。
- 区分作者表述、二次转述和分析者归纳；不补造指标。
- 样本凑齐后，再依据当前代码与个人实际贡献，研究 moatless-tools 的项目命名、介绍与 bullet。

当前需要补充可读的站内来源：用户已有收藏中的少量完整分享链接（保留原分享参数），或原帖/简历截图即可作为种子继续核验。无需提供账号密码或 Cookie。仍保留“至少十份”的目标，未将候选线索、项目概括或补充模板当作十份原帖简历。
