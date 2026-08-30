# coding-agent

本项目的目标是从零实现一个简化的编程 Agent。模型通信可以使用普通的 OpenAI 兼容客户端，
但 Agent Loop、工具调度、对话历史、上下文压缩、终止条件、错误处理和结果解析必须由项目自身实现。
项目不允许使用 LangChain/LangGraph Memory、LlamaIndex、OpenAI/Claude Agents SDK、
AutoGen、CrewAI，也不允许包装 Claude Code、Codex、OpenCode 等现成 Agent 产品。
真实 API Key 只能保存在环境变量或未提交的 `.env` 中，禁止写入代码、README 或 Git 历史。

## 当前已实现能力

- 从环境变量读取 DeepSeek 或通义千问（Qianwen）API 配置；
- 调用 OpenAI 兼容的 Chat Completions API，支持 DeepSeek 与 Qianwen 双提供商；
- 解析模型原生 Tool Calling 响应；
- 自主管理“模型 → 工具 → 模型”的 Agent Loop；
- 自主管理完整对话历史、输入预算和确定性上下文压缩，不依赖 LangChain Memory 等现成记忆模块；
- 将工具调用和对应工具结果作为原子块保留，避免压缩后形成无效消息序列；
- 通过工作区内的版本化 JSON 文件保存原始完整会话，恢复时可切换 DeepSeek 与 Qianwen；
- 每次任务前自动创建内容寻址的工作区检查点，支持 `/diff`、`/undo` 和 `/checkpoints`；
- 通过 `read_file` 分段读取工作区内的 UTF-8 文本文件；
- 通过 `list_files` 递归发现工作区内的文件和目录；
- 通过 `search_text` 在陌生仓库中定位代码、配置和符号；
- 通过 `write_file` 创建或显式覆盖工作区内的 UTF-8 文本文件；
- 通过 `rename_file` 安全重命名或移动工作区内的普通文件；
- 通过 `apply_patch` 对已有文件应用精确上下文补丁；
- 通过 `run_command` 在工作区内执行程序、测试和静态检查；
- 使用 Pydantic 校验工具参数；
- 限制可读写文件大小和工具返回文本长度；
- 使用最大模型轮次防止无限循环；
- 分类处理模型超时、连接、鉴权、限流、HTTP 状态和非法响应错误；
- 通过 Typer 和 Rich 提供单次任务与 `>>` 持续交互模式，支持 `--provider` 切换模型提供商；

## 当前项目目录

```text
coding-agent/
├── coding_agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── cli.py
│   ├── config.py
│   ├── checkpoints/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── models.py
│   │   ├── scanner.py
│   │   └── store.py
│   ├── context/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── history.py
│   │   ├── manager.py
│   │   ├── summary.py
│   │   └── token_counter.py
│   ├── llm_client.py
│   ├── protocol.py
│   ├── sessions/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── models.py
│   │   └── store.py
│   └── tools/
│       ├── __init__.py
│       ├── base.py
│       ├── command.py
│       ├── filesystem.py
│       ├── patch.py
│       ├── registry.py
│       ├── rename.py
│       ├── search.py
│       └── workspace.py
├── tests/
│   ├── test_agent.py
│   ├── test_checkpoints.py
│   ├── test_config.py
│   ├── test_context.py
│   ├── test_cli.py
│   ├── test_llm_client.py
│   ├── test_patch.py
│   ├── test_rename.py
│   ├── test_search.py
│   ├── test_sessions.py
│   └── test_tools.py
├── examples/
│   └── demo/
│       └── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirement.txt
├── run-agent.ps1
└── README.md
```


## 文件功能说明

### 核心代码

#### `coding_agent/__init__.py`

定义 `coding_agent` Python 包，并保存当前项目版本号 `0.1.0`。

#### `coding_agent/__main__.py`

提供 `python -m coding_agent` 运行入口。该文件将命令行参数交给 `cli.py` 中的 Typer 应用处理。

#### `coding_agent/agent.py`

实现项目的核心 Agent Loop，主要职责包括：

1. 创建一次性历史，或根据会话 ID 恢复已有完整历史并追加用户任务；
2. 在任务执行前创建工作区检查点；
3. 在每次模型请求前根据上下文预算生成请求视图；
4. 判断模型返回的是工具调用还是最终答案；
5. 将工具调用交给 `ToolRegistry` 执行；
6. 把工具调用与工具结果作为完整原子块加入对话历史；
7. 在完整工具交互和最终回答产生后原子保存持久会话；
8. 完成后计算相对检查点的文件变化并返回；
9. 达到最大步数或收到异常响应时安全终止。

该文件还定义：

- `SYSTEM_PROMPT`：约束模型必须使用工具读取工作区，不得虚构工具结果；
- `AgentResult`：保存最终答案、模型轮次、工具调用次数、完整消息历史、检查点 ID 和文件变化；
- `AgentError`：表示 Agent 无法安全完成任务。

#### `coding_agent/cli.py`

实现命令行界面。当前提供 `run` 命令，支持：

- 接收自然语言任务；
- 省略任务参数时进入持续交互模式，复用同一个 Agent 和会话；
- 使用 `--workspace` 或 `-w` 指定 Agent 可访问的工作区；
- 使用 `--max-steps` 限制最大模型轮次；
- 使用 `--provider` 或 `-p` 选择模型提供商（`deepseek` 或 `qianwen`）；
- 兼容旧的 `--model` 或 `-m` 参数别名；
- 使用 `--session` 或 `-s` 创建或恢复工作区内的 JSON 会话；
- 支持用 `quit`、`exit`、`q` 或 `退出` 结束交互模式；
- 支持 `/diff [id]`、`/undo [id]` 和 `/checkpoints` 本地命令；
- 展示提供商、模型名称、工作区、最终答案和运行统计；
- 将配置、模型 API、Agent、会话、文件错误和用户中断转换为清晰的终端提示。

#### `coding_agent/config.py`

负责从进程环境变量读取提供商配置。当前支持两套参数：

- `DeepseekConfig`：读取 DeepSeek 的 API Key、地址、模型、推理、请求超时和重试次数；
- `QianwenConfig`：读取 Qianwen 的 API Key、地址、模型、推理、请求超时和重试次数；
- 请求超时默认每次尝试 180 秒，可配置为 1～3,600 秒；SDK 重试默认 2 次，可配置为 0～10 次。

如果对应的 API Key 未设置，程序会抛出 `ConfigurationError`，不会在代码中使用默认密钥。

#### `coding_agent/checkpoints/`

实现不依赖 Git 的第一版工作区检查点：

- `models.py`：定义版本化 manifest、文件记录、变更集合、恢复结果和 `CheckpointError`；
- `scanner.py`：稳定扫描普通文件，排除受保护目录，拒绝符号链接并计算 SHA-256；
- `store.py`：将文件原始字节按 SHA-256 去重保存，并原子写入检查点 manifest；
- `manager.py`：创建和列出检查点，识别新增、修改、删除及重命名，生成 Unified Diff，并安全恢复；
- `__init__.py`：提供检查点包的稳定公开接口。

检查点保存在 `<workspace>/.coding-agent/checkpoints/`，模型工具无法读取。任务开始前保存完整
允许范围，因此 `run_command` 在工作区内产生的文件变化也能被发现和撤销。Undo 前会创建
`pre_undo` 安全检查点，恢复完成后重新扫描并验证哈希。

第一版最多扫描 20,000 个文件，单文件最大 32 MiB，总快照最大 256 MiB；超过限制或遇到
非忽略路径中的符号链接时拒绝执行没有完整 Undo 保护的任务。`.env`、`.git`、`.coding-agent`、
虚拟环境、依赖、缓存和构建目录不进入检查点，因此 Undo 不会改写这些状态。当前版本尚未实现
并发恢复锁和自动清理旧检查点，检查点也可能包含项目源码，不应提交 `.coding-agent`。

#### `coding_agent/context/`

实现不依赖第三方 Agent 框架的第一版对话历史与上下文管理：

- `config.py`：定义输入窗口、输出预留、安全余量、近期块数量和摘要长度配置；
- `history.py`：维护仅追加的完整历史，校验工具调用 ID，并把 assistant tool call 与全部 tool result 组成不可拆分块；
- `token_counter.py`：使用 UTF-8 字节数进行偏保守的无依赖 Token 估算，同时计算消息和工具 Schema 成本；
- `summary.py`：从旧工具事件生成确定性摘要，不调用模型，不把旧文件正文整段复制进上下文；
- `manager.py`：在预算内优先返回完整历史，超预算时保留系统消息、最新用户任务和近期原子块，并压缩更早的会话与工具事件；
- `__init__.py`：提供上下文包的稳定公开接口。

这里刻意区分两份数据：`ConversationHistory` 始终保存当前会话的完整审计历史，
`ContextManager` 只为单次模型请求构建受预算约束的临时视图，因此压缩不会破坏原始记录。
摘要被明确标记为不可信数据；模型需要精确内容时仍应重新读取文件。

#### `coding_agent/sessions/`

实现不依赖数据库的完整会话持久化：

- `models.py`：定义版本化 `SessionDocument`、提供商分段、工作区事件、完整消息和 `SessionError`；
- `adapter.py`：根据目标提供商生成请求副本，移除其他提供商专属的 `reasoning_content`；
- `store.py`：校验会话 ID，加载并验证 JSON，通过同目录临时文件、`fsync` 和 `os.replace` 原子保存；
- `__init__.py`：提供会话包的稳定公开接口。

每个 ID 对应 `<workspace>/.coding-agent/sessions/session-<id>.json`。文件保存完整的
system、user、assistant、tool 和 tool_calls 消息，而不是压缩后的模型请求。加载时会校验
格式版本、工作区、提供商分段和工具消息配对；同一 ID 的新对话追加到原历史后再整体原子更新。

#### `coding_agent/llm_client.py`

实现 OpenAI 兼容 API 的适配层，提供 `DeepSeekClient` 和 `QianwenClient` 两个客户端。该模块只负责模型通信，不负责 Agent 决策：

- 使用 `openai` Python 客户端连接各自配置的 OpenAI 兼容地址；
- 构造包含模型、消息、工具和推理配置的请求；
- 解析模型返回的文本、`finish_reason` 和 Tool Calls；
- 将超时、连接、鉴权、权限、限流、错误请求和服务端状态分类转换为安全的 `ModelClientError`；
- 校验 `choices`、message、正文、结束原因和每个 tool call 的必需字段，拒绝结构损坏的响应；
- 将 SDK 返回对象统一转换为项目内部的 `ModelTurn` 和 `ToolCall`。

这种隔离设计使 Agent 核心逻辑不直接依赖厂商 SDK 的数据结构，也方便在测试中替换为 Fake Client。

#### `coding_agent/protocol.py`

定义模型层和 Agent 层之间的内部协议：

- `ToolCall`：保存工具调用 ID、工具名称和原始 JSON 参数，并提供 `as_api_dict()` 生成可加入 assistant 消息的 tool_calls 数据；
- `ModelTurn`：保存一次模型响应的正文、工具调用、结束原因和推理内容；
- `ModelClientError`：保存安全错误消息、错误类别、是否可重试和可选 HTTP 状态码；
- `ChatClient`：通过 Python `Protocol` 描述模型客户端必须提供的 `complete` 接口。

`ModelTurn.as_assistant_message()` 会将内部对象重新转换成可加入 API 对话历史的 assistant 消息。

#### `coding_agent/tools/__init__.py`

作为工具包的稳定公开入口，向 Agent 暴露 `ToolRegistry`。因此调用方仍然使用
`from coding_agent.tools import ToolRegistry`，无需了解工具包内部如何拆分。

#### `coding_agent/tools/base.py`

定义工具系统的公共抽象：

- `ToolResult`：工具结构化返回值类型；
- `ToolHandler`：本地工具处理函数类型；
- `ToolSpec`：集中保存工具名称、说明、Pydantic 输入模型和处理函数，并统一生成发送给模型的 JSON Schema。

#### `coding_agent/tools/workspace.py`

集中处理多个本地工具共用的工作区安全规则：

- 解析工作区相对路径；
- 拒绝绝对路径、路径逃逸和 `.env`；
- 判断缓存、依赖、构建产物和版本控制目录是否应被忽略；
- 将本地路径转换为稳定的工作区相对路径。

#### `coding_agent/tools/filesystem.py`

实现文件系统类工具及其参数模型：

- `ReadFileInput` 与 `read_file()`：分段读取 UTF-8 文件并添加行号；
- `ListFilesInput` 与 `list_files()`：按稳定顺序递归发现文件和目录；
- `WriteFileInput` 与 `write_file()`：通过同目录临时文件和原子替换创建或覆盖文件；
- `FILE_TOOLS`：按 `read_file`、`list_files`、`write_file` 的顺序声明文件工具规格。

#### `coding_agent/tools/patch.py`

实现 `apply_patch` 精确补丁工具模块，针对已有文件做基于上下文的精准修改，不必重写整个文件：

- `ApplyPatchInput`：校验目标路径和补丁文本，拒绝包含空字节的补丁；
- 解析 `@@` 开头的 hunk，支持裸 `@@` 或统一 diff 形式 `@@ -start,count +start,count @@`；
- 每个 hunk 至少包含一条上下文或 `-` 行，禁止纯插入 hunk，并强制 `+`/`-` 行数量与头部计数一致；
- 按上下文精确匹配，匹配位置不唯一或 hunk 相互重叠时拒绝应用；
- 通过同目录临时文件写入并保留原文件权限后原子替换，返回前后 SHA-256、增删行数与命中位置；
- 文件与补丁文本上限均为 256 KiB，拒绝被忽略路径、敏感路径、符号链接、非 UTF-8 文件和目录。



#### `coding_agent/tools/rename.py`

实现 `rename_file` 文件重命名工具：

- `RenameFileInput`：校验源路径、目标路径、覆盖选项和父目录创建选项；
- `rename_file()`：使用 `os.replace` 在工作区内原子重命名或移动普通文件；
- 默认拒绝覆盖已有目标文件，只有显式传入 `overwrite=true` 才允许替换；
- 可通过 `create_parent_dirs=true` 显式创建缺失的目标父目录；
- 拒绝目录、路径逃逸、敏感路径、被忽略路径以及经过符号链接的路径；
- 返回源路径、目标路径、操作类型、移动字节数以及被替换文件的字节数。

重命名会移除旧路径，而不是复制出第二份文件。Agent 在重命名后会使用 `search_text`
查找旧文件名或模块名引用，并在需要时继续更新引用。

#### `coding_agent/tools/search.py`

实现 `search_text` 文本搜索工具：

- `SearchTextInput`：校验查询文本、搜索路径、大小写选项、文件通配符和数量限制；
- `search_text()`：搜索单个文件或按稳定路径顺序递归搜索目录；
- 每个匹配结果返回工作区相对路径、行号、首个匹配列、匹配行文本和单行截断状态；
- 支持 `*.py`、`src/*.py` 等可选文件名通配符；
- 跳过被忽略目录、符号链接、超大文件、非 UTF-8 文件和包含空字节的文件；
- `SEARCH_TEXT_TOOL`：声明搜索工具规格并由注册器统一接入。

当前查询按普通文本匹配，不把用户输入解释为正则表达式，避免无效或高开销的正则表达式影响 Agent Loop。

#### `coding_agent/tools/command.py`

实现命令执行类工具：

- `RunCommandInput`：校验参数数组、工作目录、超时和输出长度限制；
- `run_command()`：使用 `shell=False` 执行程序，并返回退出码、stdout、stderr、耗时和超时状态；
- 命令拦截、子进程环境清理和输出截断等辅助逻辑；
- `RUN_COMMAND_TOOL`：声明命令工具规格。

#### `coding_agent/tools/registry.py`

实现 `ToolRegistry`。注册器组合各模块声明的 `ToolSpec`，统一完成：

1. 向模型提供全部工具的 JSON Schema；
2. 根据工具名称查找对应规格；
3. 解析模型返回的 JSON 参数；
4. 使用对应 Pydantic 模型校验参数；
5. 调用处理函数并捕获可预期的校验、路径和文件系统错误。

新增工具时只需在所属模块声明新的 `ToolSpec` 并加入工具集合，不需要继续扩展注册器中的条件分支。

当前 `DEFAULT_TOOLS` 注册的工具为 `read_file`、`list_files`、`write_file`、`rename_file`、
`search_text`、`run_command`、`apply_patch`。

当前安全限制包括：

- 不允许绝对路径；
- 不允许通过 `..` 或符号链接逃出工作区；
- 不允许读取 `.env`；
- 不允许通过模型工具访问 `.env`、`.coding-agent`、`.git`、依赖目录、缓存或构建产物；
- 目录发现会忽略版本控制目录、虚拟环境、依赖目录、缓存和构建产物；
- `list_files` 最大递归深度为 10，单次最多返回 1,000 项；
- `search_text` 只搜索工作区内允许访问的普通 UTF-8 文件，不跟随符号链接；
- `search_text` 默认最多检查 1,000 个候选文件、返回 100 个匹配行，上限分别为 10,000 和 1,000；
- 搜索时单文件最大为 256 KiB，单行最多返回 1,000 个字符，总结果限制为 40,000 个字符；
- `rename_file` 只处理普通文件，不允许重命名目录或通过符号链接移动文件；
- `rename_file` 默认拒绝覆盖目标文件，覆盖必须显式设置 `overwrite=true`；
- 单个文件不能超过 256 KiB；
- 写入现有文件必须显式设置 `overwrite=true`；
- 写入内容最大为 256 KiB，并按 UTF-8 实际字节数计算；
- `write_file` 返回写前和写后的 SHA-256，便于审计变更；
- `run_command` 始终使用参数数组和 `shell=False`；
- 命令工作目录必须位于工作区，默认超时 30 秒，最大超时 120 秒；
- 子进程 `PATH` 优先使用当前 Agent 所在虚拟环境的 Python；
- stdout 和 stderr 默认各保留 12,000 个字符，超长输出保留头尾；
- 子进程环境会移除 API Key、Token、Password、Secret 等敏感变量；
- 明显危险的系统命令和破坏性 Git 子命令会被拒绝；
- 工具返回文本最多保留 40,000 个字符；
- 只读取 UTF-8 文本文件。

### 自动化测试

#### `tests/test_agent.py`

使用确定性的 Fake Client 测试完整 Agent Loop：

- 模型第一次请求读取文件；
- 本地执行 `read_file`；
- 工具结果返回模型；
- 模型第二次生成最终答案；
- Agent 正确统计模型轮次和工具调用次数；
- 无限请求工具时，Agent 能按最大步数终止；
- 模型能够调用 `write_file` 创建文件，再使用 `read_file` 回读验证；
- 模型能够完成 `write_file → run_command → 根据输出结束` 的执行闭环。
- 模型能够调用 `rename_file` 重命名文件，并根据工具成功结果结束任务；
- 模型能够调用 `apply_patch` 完成局部修改；
- 超出上下文预算时，请求会压缩旧事件，同时 `AgentResult.messages` 保留完整历史；
- Agent 执行工具前创建检查点，并在完成后返回变更列表；
- 检查点能够撤销 Agent 工具产生的新文件。

#### `tests/test_config.py`

测试 DeepSeek 与 Qianwen 的配置读取逻辑，包括缺少 API Key 时拒绝启动、环境变量覆盖、
超时和重试范围校验、Qianwen 默认兼容地址，以及旧 `Config` 名称对 `DeepseekConfig` 的兼容别名。

#### `tests/test_context.py`

测试自研上下文管理，包括：

- 预算充足时请求消息保持不变，且请求副本不会修改完整历史；
- 超预算时旧工具事件被确定性压缩，近期工具调用和结果仍完整保留；
- 同一轮多个工具结果组成一个不可拆分的原子块；
- 拒绝孤立工具结果和缺失工具结果；
- 工具 Schema 成本会计入输入预算；
- 必需消息无法放入窗口时返回清晰错误；
- 上下文环境变量读取、非法配置和默认 Token 估算。
- 从完整消息恢复历史，并在多轮压缩时始终保留最新用户任务。

#### `tests/test_cli.py`

测试命令行客户端工厂能够根据提供商分别构造 DeepSeek 与 Qianwen 客户端，并验证单次错误输出、
交互循环、Agent 复用、退出命令、检查点斜杠命令和任务失败后的继续提问。

#### `tests/test_checkpoints.py`

测试工作区检查点，包括新增、修改、删除和重命名检测、Unified Diff、完整恢复、Undo 前安全
检查点、受保护状态排除、损坏对象拒绝恢复和符号链接拒绝。

#### `tests/test_llm_client.py`

通过 Mock 对象测试 DeepSeek 与 Qianwen 适配层是否：

- 构造正确的模型请求；
- 启用 Tool Calling，并按各提供商要求传递推理配置；
- 正确解析模型返回的工具名称、参数和结束原因；
- 将超时、连接、鉴权、限流和服务端错误转换成分类的安全错误；
- 拒绝空 choices 和字段不完整的 tool call。

测试不会发起真实网络请求。

#### `tests/test_sessions.py`

测试 JSON 会话持久化，包括：

- 完整历史往返加载，并在同一 ID 下追加新 user/assistant 而不丢失旧消息；
- 会话 ID 路径安全、格式版本、损坏 JSON 和工作区绑定校验；
- 原子替换失败时保留原会话文件；
- 两次独立 Agent 运行使用同一 ID 恢复历史；
- Undo 工作区事件独立持久化，并在请求副本中提示模型重新读取文件；
- 模型文件工具无法读取、写入或发现 `.coding-agent` 会话状态。

#### `tests/test_tools.py`

测试 `read_file`、`list_files`、`write_file`、`run_command` 和工具注册器，包括：

- 分段读取并添加正确行号；
- 拒绝工作区路径逃逸；
- 拒绝非法 JSON；
- 拒绝 Schema 之外的额外参数；
- 目录递归、稳定排序和文件元数据；
- 最大深度和数量截断；
- 忽略 `.env`、`.git`、虚拟环境、缓存、依赖和构建目录；
- 拒绝越界、被忽略目录和非目录目标；
- 创建 UTF-8 文件并返回写入元数据；
- 拒绝未授权覆盖且保持原文件不变；
- 显式覆盖、相同内容覆盖和 SHA-256 变化判断；
- 可选创建父目录；
- 拒绝敏感路径、目录目标和超出 UTF-8 字节上限的内容；
- 成功命令的 stdout、stderr、退出码、工作目录和耗时；
- 非零退出、命令不存在、执行超时和输出截断；
- 子进程无法继承 `DEEPSEEK_API_KEY`；
- 拒绝危险命令、越界工作目录和非法参数。

#### `tests/test_search.py`

测试 `search_text` 的独立搜索行为，包括：

- 递归搜索、稳定结果顺序、路径、行号和列号；
- 默认区分大小写以及可选的不区分大小写搜索；
- 文件名通配符和单文件搜索；
- 匹配数量、候选文件数量和长行截断；
- 跳过被忽略目录、非 UTF-8 文件和超出大小限制的文件；
- 拒绝路径逃逸、被忽略路径、多行查询和非法参数。

#### `tests/test_rename.py`

测试 `rename_file` 的文件移动行为，包括：

- 重命名后旧路径消失且文件内容保持不变；
- 默认拒绝覆盖并保留源文件和目标文件；
- 显式覆盖目标文件并返回被替换文件元数据；
- 可选创建目标父目录；
- 拒绝相同路径、缺失源文件、目录、路径逃逸和被忽略路径；
- 原子移动失败时保留原始文件；
- 拒绝缺失参数和 Schema 之外的额外参数。

#### `tests/test_patch.py`

测试 `apply_patch` 的独立补丁行为，包括裸 hunk、包装格式、CRLF 保留、多 hunk、
行号消歧、模糊匹配拒绝、计数与重叠校验、路径校验、原子失败和文件权限保留。

### 配置与工程文件

#### `examples/demo/README.md`

最小演示工作区。文件中说明计算器实现应放在 `calculator.py`，用于验证真实模型能否主动调用 `read_file` 并根据工具结果回答问题。

#### `.env.example`

提供 DeepSeek、Qianwen、请求超时、重试和上下文预算的环境变量模板，只包含占位符，不包含真实 API Key。
真实配置应放在未提交的 `.env` 中。

#### `.gitignore`

忽略虚拟环境、`.env`、`.coding-agent` 会话状态、Python 缓存、测试缓存、覆盖率文件、
构建产物和常见编辑器临时文件，防止密钥和本地生成内容被提交。

#### `pyproject.toml`

定义项目元数据、Python 版本、运行依赖、开发依赖、`coding-agent` 命令入口，以及 pytest 和 Ruff 配置。

#### `requirement.txt`

列出运行和开发所需的全部 Python 依赖，可用于一次性安装 DeepSeek/Qianwen 客户端、Pydantic、Typer、Rich、pytest、覆盖率工具和 Ruff。

#### `run-agent.ps1`

PowerShell 启动脚本。自动读取 `.env`，根据 `-Provider` 校验对应 API Key，优先使用项目
`.venv` 中的 Python，并把可选任务、工作区、最大轮次、提供商和会话 ID 传递给 `coding_agent`。
省略任务时直接进入持续交互模式。
`-CheckConfig` 模式只验证配置，不调用模型或显示密钥。

#### `README.md`

当前文件，负责说明项目定位、已实现能力、目录结构和各文件职责。

## 安装依赖

在项目根目录创建并激活虚拟环境后执行：

```powershell
pip install -r requirement.txt
```

## 运行 Agent

项目提供 `run-agent.ps1`，可以自动读取项目根目录的 `.env`、根据提供商校验 API Key、
选择 `.venv` 中的 Python 并启动 Agent。默认提供商为 DeepSeek，也可通过
`-Provider qianwen` 切换到 Qianwen。

### 持续交互模式

启动时不提供任务，就会进入 `>>` 循环：

```powershell
.\run-agent.ps1 `
  -Workspace C:\Users\FRAN\Desktop\test `
  -MaxSteps 20 `
  -Provider qianwen `
  -Session 1
```

启动后可以连续输入任务：

```text
coding-agent  provider=qianwen  model=...  workspace=...  session=1
Interactive mode  输入编程任务，输入 quit、exit 或退出结束。

>> 查看当前项目结构并总结
...
Completed

>> 修改刚才分析的代码并运行测试
...
Completed

>> /diff
Changes since cp-...:
  M coding_agent/agent.py

>> /undo
Restore cp-...? [y/N] y
Restored cp-...

>> quit
会话已结束。
```

这里的 `-Session 1` 会让每个问题共享并持久化完整对话历史；关闭程序后，再用相同工作区和
Session ID 启动仍可继续原会话。不传 `-Session` 也能交互，但每个问题的历史彼此独立。

每个普通任务开始前都会自动创建检查点。`/diff [checkpoint-id]` 查看差异，
`/undo [checkpoint-id]` 在确认后恢复，`/checkpoints` 列出最近的检查点。省略 ID 时使用当前进程
最近的检查点；重启后使用工作区中创建时间最新的检查点。Undo 只恢复工作区文件，不删除
原始对话消息，并会先建立 `pre_undo` 安全检查点。恢复成功后自动向当前会话的
`workspace_events` 写入本地事件；下一次请求会收到“文件内容可能过期，必须重新读取”的提示。
未使用 `-Session` 时事件只保存在当前交互进程内，退出后不持久化。

只检查 `.env` 是否能够正确加载（不会显示密钥）：

```powershell
.\run-agent.ps1 -CheckConfig
```

直接运行任务：

```powershell
.\run-agent.ps1 `
  "请阅读 README.md，并总结这个项目" `
  -Workspace .
```

也可以指定最大模型轮次：

```powershell
.\run-agent.ps1 `
  "请分析 coding_agent 目录" `
  -Workspace . `
  -MaxSteps 20
```

如果不使用启动脚本，也可以在 PowerShell 中手动加载 `.env`：

```powershell
Get-Content .env |
Where-Object { $_ -match '^\s*[^#][^=]*=' } |
ForEach-Object {
    $name, $value = $_ -split '=', 2
    $value = $value.Trim().Trim([char]34).Trim([char]39)
    Set-Item -Path "Env:$($name.Trim())" -Value $value
}
```

运行真实 DeepSeek Tool Calling 演示：

```powershell
python -m coding_agent run `
  "请阅读 README.md，并告诉我计算器代码应该放在哪里" `
  --workspace .\examples\demo
```

成功的执行应包含一次 `read_file` 工具调用，并输出类似统计：

```text
steps=2 tool_calls=1
```

使用千问提供商运行（需要设置 `QIANWEN_API_KEY`）：

```powershell
python -m coding_agent run `
  "请阅读 README.md，并告诉我计算器代码应该放在哪里" `
  --workspace .\examples\demo `
  --provider qianwen
```

也可以通过启动脚本运行 Qianwen：

```powershell
.\run-agent.ps1 `
  "请阅读 README.md，并总结这个项目" `
  -Workspace . `
  -Provider qianwen
```

模型请求的超时和 SDK 自动重试可在 `.env` 中分别配置：

```dotenv
DEEPSEEK_REQUEST_TIMEOUT_SECONDS=180
DEEPSEEK_MAX_RETRIES=2
QIANWEN_REQUEST_TIMEOUT_SECONDS=180
QIANWEN_MAX_RETRIES=2
```

发生超时、连接失败、鉴权失败、限流、错误请求或服务端异常时，CLI 会输出一条分类后的
简洁错误，不再展示底层 HTTP Traceback。超时是每次请求尝试的上限；开启 thinking 的复杂
任务可适当提高该值，限流和服务端错误则应结合提示等待后重试。

验证写入和执行闭环：

```powershell
.\run-agent.ps1 `
  "创建 hello.py，使它输出 Hello, Agent!，运行程序并确认输出正确" `
  -Workspace .\examples\demo
```

理想工具轨迹为：

```text
list_files
→ write_file("hello.py")
→ run_command(["python", "hello.py"])
→ 检查 exit_code 和 stdout
→ 返回完成报告
```

该任务会在演示工作区创建并执行 `hello.py`。如果文件已经存在，模型必须显式
传入 `overwrite=true` 才能覆盖。

## 持久会话

使用 `--session`（启动脚本中为 `-Session`）可以把完整对话历史保存在当前工作区。
第一次使用某个 ID 时创建会话：

```powershell
.\run-agent.ps1 `
  "创建 calculator.py，实现加减乘除并运行验证" `
  -Workspace .\examples\demo `
  -Session calculator
```

再次使用同一个 ID 时恢复全部旧消息，再追加当前任务：

```powershell
.\run-agent.ps1 `
  "修改上一次写的代码，让除数为零时抛出异常，并重新验证" `
  -Workspace .\examples\demo `
  -Session calculator
```

也可以直接使用模块入口：

```powershell
python -m coding_agent run `
  "继续上一次任务" `
  --workspace .\examples\demo `
  --session calculator
```

会话文件位于：

```text
examples/demo/.coding-agent/sessions/session-calculator.json
```

同一 ID 会加载旧 JSON，将本次 user、assistant 和 tool 消息追加到内存历史后，再通过
临时文件整体原子更新；文件替换不会删除之前的消息。不同 ID 使用不同文件。不传
`--session` 时保持原来的一次性模式，不创建 `.coding-agent`。会话与规范化工作区绑定，
但不再与单一提供商绑定；ID 只允许 1～64 位字母、数字、下划线和连字符，并且必须以字母
或数字开头。

同一会话可以从 DeepSeek 切换到 Qianwen，也可以再切换回来。JSON 中的 assistant 原始消息
（包括 `reasoning_content`）始终原样保存，同时用 `provider_segments` 记录各段消息的来源。
真正请求模型时才创建深拷贝：目标提供商自己的推理字段保持不变，其他提供商产生的
`reasoning_content` 会从请求副本中移除。转换和上下文压缩都不会反写原始会话历史。
`workspace_events` 独立保存 Undo 等可信本地事件，并只在构建模型请求时追加到临时系统提示。
旧版 `format_version: 1` 和版本 2 会话会在内存中迁移，并在下一次保存时写成版本 3。

`.coding-agent` 已加入 Git 和模型工具忽略规则。会话文件可能包含读取过的代码片段和命令输出，
不应提交到仓库。当前第一版不支持并发写入同一会话；工具执行产生副作用后、会话保存前若进程
异常终止，恢复时仍应要求 Agent 重新读取文件并核对工作区状态。

## 对话历史与上下文预算

第一版上下文管理完全由项目代码实现，不使用 LangChain Memory、LlamaIndex、
Agents SDK、AutoGen、CrewAI 等现成 Agent 或记忆模块。默认配置为：

```dotenv
CODING_AGENT_CONTEXT_TOKENS=65536
CODING_AGENT_OUTPUT_RESERVE=8192
CODING_AGENT_CONTEXT_SAFETY_MARGIN=2048
CODING_AGENT_RECENT_BLOCKS=6
CODING_AGENT_SUMMARY_MAX_CHARS=8000
```

可用于模型输入的预算为：

```text
模型上下文窗口 - 输出预留 - 安全余量 - 工具 Schema 估算
```

当完整历史能够放入预算时，模型会收到原始消息；超出预算后，系统消息和最新用户任务始终保留，
最近的当前轮工具交互按原子块保留，更早的用户、助手和工具事件会生成确定性摘要。摘要只记录
任务概要、文件路径、操作结果、哈希和命令退出码等必要证据，不通过额外模型调用生成。默认
Token 估算不依赖特定厂商 tokenizer，而是按 UTF-8 字节数偏保守估计，实际部署时应根据所用
模型窗口调整上述参数。

启用 `--session` 后，原始完整历史会跨进程保存在 JSON 中，但每次发送给模型的仍然只是经过
提供商转换和上下文预算处理的请求视图。
当前第一版不提供长期语义记忆、向量检索、会话分支、并发会话锁或云端同步；后续能力仍应基于
自有数据结构实现，而不是接入现成 Agent Memory。

## 命令执行安全边界

`run_command` 是带路径、参数、超时、输出和环境变量限制的本地执行器，但不是
操作系统级安全沙箱。`shell=False` 和危险命令策略可以降低误操作风险，却无法阻止
一个获准执行的解释器脚本主动访问当前用户有权限访问的其他资源。因此只应在可信的
本地项目或隔离环境中使用，不应直接运行来源未知的恶意仓库。

## 运行测试

```powershell
pytest
```

静态检查：

```powershell
ruff check .
```

