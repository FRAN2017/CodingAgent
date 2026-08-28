# coding-agent

## 当前已实现能力

- 从环境变量读取 DeepSeek 或通义千问（Qianwen）API 配置；
- 调用 OpenAI 兼容的 Chat Completions API，支持 DeepSeek 与 Qianwen 双提供商；
- 解析模型原生 Tool Calling 响应；
- 自主管理“模型 → 工具 → 模型”的 Agent Loop；
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
- 通过 Typer 和 Rich 提供命令行界面，支持 `--model` 切换模型提供商；

## 当前项目目录

```text
coding-agent/
├── coding_agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── cli.py
│   ├── config.py
│   ├── llm_client.py
│   ├── protocol.py
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
│   ├── test_config.py
│   ├── test_llm_client.py
│   ├── test_rename.py
│   ├── test_search.py
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

1. 构造系统消息和用户任务消息；
2. 调用模型客户端获取下一轮响应；
3. 判断模型返回的是工具调用还是最终答案；
4. 将工具调用交给 `ToolRegistry` 执行；
5. 把工具结果作为 `role: tool` 消息加入对话历史；
6. 持续循环，直到模型生成最终答案；
7. 达到最大步数或收到异常响应时安全终止。

该文件还定义：

- `SYSTEM_PROMPT`：约束模型必须使用工具读取工作区，不得虚构工具结果；
- `AgentResult`：保存最终答案、模型轮次、工具调用次数和完整消息历史；
- `AgentError`：表示 Agent 无法安全完成任务。

#### `coding_agent/cli.py`

实现命令行界面。当前提供 `run` 命令，支持：

- 接收自然语言任务；
- 使用 `--workspace` 或 `-w` 指定 Agent 可访问的工作区；
- 使用 `--max-steps` 限制最大模型轮次；
- 使用 `--model` 或 `-m` 选择模型提供商（`deepseek` 或 `qianwen`）；
- 展示提供商、模型名称、工作区、最终答案和运行统计；
- 将配置错误、Agent 错误、文件错误和用户中断转换为清晰的终端提示。

#### `coding_agent/config.py`

负责从进程环境变量读取提供商配置。当前支持两套参数：

- `DeepseekConfig`：读取 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`DEEPSEEK_THINKING`、`DEEPSEEK_REASONING_EFFORT`；
- `QianwenConfig`：读取 `QIANWEN_API_KEY`、`QIANWEN_BASE_URL`、`QIANWEN_MODEL`、`QIANWEN_THINKING`、`QIANWEN_REASONING_EFFORT`；
- 两者共用 `thinking_enabled`、`reasoning_effort` 推理配置和 `request_timeout_seconds` 请求超时（默认 60 秒）。

如果对应的 API Key 未设置，程序会抛出 `ConfigurationError`，不会在代码中使用默认密钥。

#### `coding_agent/llm_client.py`

实现 OpenAI 兼容 API 的适配层，提供 `DeepSeekClient` 和 `QianwenClient` 两个客户端。该模块只负责模型通信，不负责 Agent 决策：

- 使用 `openai` Python 客户端连接各自配置的 OpenAI 兼容地址；
- 构造包含模型、消息、工具和推理配置的请求；
- 解析模型返回的文本、`finish_reason` 和 Tool Calls；
- 将 SDK 返回对象统一转换为项目内部的 `ModelTurn` 和 `ToolCall`。

这种隔离设计使 Agent 核心逻辑不直接依赖厂商 SDK 的数据结构，也方便在测试中替换为 Fake Client。

#### `coding_agent/protocol.py`

定义模型层和 Agent 层之间的内部协议：

- `ToolCall`：保存工具调用 ID、工具名称和原始 JSON 参数，并提供 `as_api_dict()` 生成可加入 assistant 消息的 tool_calls 数据；
- `ModelTurn`：保存一次模型响应的正文、工具调用、结束原因和推理内容；
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
- 不允许写入 `.env`、`.git`、依赖目录、缓存或构建产物；
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
- 模型能够调用 `rename_file` 重命名文件，并根据工具成功结果结束任务。

#### `tests/test_config.py`

测试配置读取逻辑，包括缺少 DeepSeek API Key 时拒绝启动，以及正确读取模型名称和推理开关。
该文件仍按多提供商拆分前的旧 `Config` 接口编写，正在随配置改造同步更新（见“当前边界与下一阶段”）。

#### `tests/test_llm_client.py`

通过 Mock 对象测试 DeepSeek 适配层是否：

- 构造正确的模型请求；
- 启用 Tool Calling 和推理配置；
- 正确解析模型返回的工具名称、参数和结束原因。

测试不会发起真实网络请求。该文件同样仍引用旧的 `Config` 构造方式，正在随多提供商改造同步更新。

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

### 配置与工程文件

#### `examples/demo/README.md`

最小演示工作区。文件中说明计算器实现应放在 `calculator.py`，用于验证真实模型能否主动调用 `read_file` 并根据工具结果回答问题。

#### `.env.example`

提供 DeepSeek 环境变量模板，只包含占位符，不包含真实 API Key。真实配置应放在未提交的 `.env` 中。

#### `.gitignore`

忽略虚拟环境、`.env`、Python 缓存、测试缓存、覆盖率文件、构建产物和常见编辑器临时文件，防止密钥和本地生成内容被提交。

#### `pyproject.toml`

定义项目元数据、Python 版本、运行依赖、开发依赖、`coding-agent` 命令入口，以及 pytest 和 Ruff 配置。

#### `requirement.txt`

列出运行和开发所需的全部 Python 依赖，可用于一次性安装 DeepSeek/Qianwen 客户端、Pydantic、Typer、Rich、pytest、覆盖率工具和 Ruff。

#### `run-agent.ps1`

PowerShell 启动脚本。自动读取 `.env`、校验 `DEEPSEEK_API_KEY`、优先使用项目
`.venv` 中的 Python，并把任务、工作区和最大轮次参数传递给 `coding_agent`。
`-CheckConfig` 模式只验证配置，不调用模型或显示密钥。

#### `README.md`

当前文件，负责说明项目定位、已实现能力、目录结构和各文件职责。

## 安装依赖

在项目根目录创建并激活虚拟环境后执行：

```powershell
pip install -r requirement.txt
```

## 运行 Agent

项目提供 `run-agent.ps1`，可以自动读取项目根目录的 `.env`、校验 API Key、
选择 `.venv` 中的 Python 并启动 Agent。该脚本当前校验的是 DeepSeek 配置；
使用 Qianwen 提供商请直接以 `python -m coding_agent run` 方式运行。

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
  --model qianwen
```

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

## 当前边界与下一阶段

当前版本已经具备“发现文件 → 搜索文本 → 读取文件 → 写入或重命名文件 → 执行并观察结果”的最小编程闭环，并支持 DeepSeek 与通义千问（Qianwen）双提供商。

当前进度中的未完成部分：

- `apply_patch` 已接入 `ToolRegistry` 默认工具集；其自动化测试尚未补全；
- `test_config.py` 与 `test_llm_client.py` 仍按多提供商拆分前的旧 `Config` 接口编写，`pytest` 目前会在收集阶段失败，需要随配置改造同步更新。

下一阶段计划依次实现：

```text
补全 apply_patch 的自动化测试
→ 更新 test_config.py / test_llm_client.py 以覆盖多提供商配置与客户端
→ 测试失败后的自动修复
→ 基于验证证据的完成判定
```