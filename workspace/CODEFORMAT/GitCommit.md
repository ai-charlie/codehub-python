
#### 提交原则：

- 应该把握好提交的频率，尽量按照**原子功能**提交，并在每次commit时给与清晰简洁的注释。
- 在完成一个较完整的功能后，进行一次push动作。
- 如果多人在同一个分支开发，用fetch + rebase命令替代pull, 避免不必要的分支merge。
- Merge代码时需要两人以上进行代码Review。

#### 格式化的Commit message好处：

- 提供更多的历史信息，方便快速浏览。

- 可以过滤某些commit（比如文档改动），便于快速查找信息。

- 可以直接从commit生成Change log。

#### Commit message 格式：

每次提交Commit message应符合以下规范，包含三个部分：header, body和footer，其中 header 部分必选，body和footer可选，如下：

```
<type>: <subject> // header【必填】
// 空一行
<body> // body 【可选】
// 空一行
<footer> // footer【可选】
// 任何一行都不得超过72个字符
```

##### type

type代表某次提交的类型，比如是修复一个bug还是增加一个新的feature。

所有的type类型如下：

 

| **类型** | **意义**                                                 |
| -------- | -------------------------------------------------------- |
| feat     | 新增 feature,[v版本号][需求简短说明][提交说明]           |
| fix      | 修复 bug，有jira编号，附上jira编号                       |
| docs     | 仅修改了文档，比如README,CHANGELOG,CONTRIBUTE等等        |
| style    | 只修改了代码格式：空格、格式缩进、换行等等，没改代码逻辑 |
| refactor | 代码重构，没有加新功能或者修复bug                        |
| perf     | 优化相关，比如提升性能、体验                             |
| test     | 测试用例，包括单元测试、集成测试等                       |
| chore    | 改变构建流程、或者增加依赖库、工具、变更版本号等         |
| revert   | 版本回滚                                                 |

 

##### 详细说明

```
<header> 50个字符以内，描述主要变更内容
<body> 更详细的说明文本，建议72个字符以内。 需要描述的信息包括:
* 为什么这个变更是必须的? 它可能是用来修复一个bug，增加一个feature，提升性能、可靠性、稳定性等等
* 他如何解决这个问题? 具体描述解决问题的步骤
* 是否存在副作用、风险?
<footer> 如果需要的化可以添加一个链接到issue地址或者其它文档，或者关闭某个issue。
```

#### 示例

```
// 需求开发过程中
feat: [G2 4.5.0][String UID][完成UserInfo功能开发]
// 文档更新，简明的细节说明
docs: 更新README.md
- 项目目录说明
- 新增提交例子
```

#### Merge request的MR message

和代码提交的 Commit message 类似，MR message 也需要遵循上面提到的格式规范，唯一不同的是：MR 的 message 里还需要增加一个 reviewer 的部分，即包含四个部分：reviewer 、header、body和footer，其中 reviewer 和 header 部分必选，body和footer可选。如下所示：

```
@Evan
// 空一行
<type>: <subject> // header【必填】
// 空一行
<body> // body 【可选】
// 空一行
<footer> // footer【可选】
// 任何一行都不得超过72个字符
```

示例如下：

```
@Evan
feat: [G2 4.5.0][String UID][完成UserInfo功能开发]
```