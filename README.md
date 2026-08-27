# coding-agent

## 当前已实现能力

- 从环境变量读取 DeepSeek API 配置；
- 调用 DeepSeek Chat Completions API；
- 解析模型原生 Tool Calling 响应；
- 自主管理“模型 → 工具 → 模型”的 Agent Loop；
- 通过 `read_file` 分段读取工作区内的 UTF-8 文本文件；
- 通过 `list_files` 递归发现工作区内的文件和目录；
- 通过 `write_file` 创建或显式覆盖工作区内的 UTF-8 文本文件；
- 使用 Pydantic 校验工具参数；
- 限制可读写文件大小和工具返回文本长度；
- 使用最大模型轮次防止无限循环；
- 通过 Typer 和 Rich 提供命令行界面；

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
│   └── tools.py
├── tests/
│   ├── test_agent.py
│   ├── test_config.py
│   ├── test_llm_client.py
│   └── test_tools.py
├── examples/
│   └── demo/
│       └── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirement.txt
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
- 展示模型名称、工作区、最终答案和运行统计；
- 将配置错误、Agent 错误、文件错误和用户中断转换为清晰的终端提示。

#### `coding_agent/config.py`

负责从进程环境变量读取 DeepSeek 配置。当前支持：

- `DEEPSEEK_API_KEY`：必填的 API 密钥；
- `DEEPSEEK_BASE_URL`：API 地址；
- `DEEPSEEK_MODEL`：使用的模型名称；
- `DEEPSEEK_THINKING`：是否启用推理模式；
- `DEEPSEEK_REASONING_EFFORT`：推理强度。

如果没有设置 API Key，程序会抛出 `ConfigurationError`，不会在代码中使用默认密钥。

#### `coding_agent/llm_client.py`

实现 DeepSeek API 适配层。该模块只负责模型通信，不负责 Agent 决策：

- 使用 `openai` Python 客户端连接 DeepSeek 的兼容接口；
- 构造包含模型、消息、工具和推理配置的请求；
- 解析模型返回的文本、`finish_reason` 和 Tool Calls；
- 将 SDK 返回对象转换为项目内部的 `ModelTurn` 和 `ToolCall`。

这种隔离设计使 Agent 核心逻辑不直接依赖厂商 SDK 的数据结构，也方便在测试中替换为 Fake Client。

#### `coding_agent/protocol.py`

定义模型层和 Agent 层之间的内部协议：

- `ToolCall`：保存工具调用 ID、工具名称和原始 JSON 参数；
- `ModelTurn`：保存一次模型响应的正文、工具调用、结束原因和推理内容；
- `ChatClient`：通过 Python `Protocol` 描述模型客户端必须提供的 `complete` 接口。

`ModelTurn.as_assistant_message()` 会将内部对象重新转换成可加入 API 对话历史的 assistant 消息。

#### `coding_agent/tools.py`

实现本地工具系统以及 `read_file`、`list_files`、`write_file` 三个实际工具。

主要内容包括：

- `ReadFileInput`：使用 Pydantic 校验路径、开始行和结束行；
- `ListFilesInput`：校验目录路径、最大递归深度和返回数量限制；
- `WriteFileInput`：校验文件路径、完整内容、覆盖和父目录创建选项；
- `read_file_schema()`：生成发送给模型的 JSON Schema；
- `list_files_schema()`：生成目录发现工具的 JSON Schema；
- `write_file_schema()`：生成本地文本文件写入工具的 JSON Schema；
- `resolve_workspace_path()`：解析并检查工作区路径；
- `read_file()`：读取指定行、添加行号并返回结构化结果；
- `list_files()`：按稳定顺序递归列出目录，返回路径、类型和文件大小；
- `write_file()`：通过同目录临时文件和原子替换创建或覆盖文件；
- `ToolRegistry`：解析 JSON 参数、校验参数并分发工具调用。

当前安全限制包括：

- 不允许绝对路径；
- 不允许通过 `..` 或符号链接逃出工作区；
- 不允许读取 `.env`；
- 不允许写入 `.env`、`.git`、依赖目录、缓存或构建产物；
- 目录发现会忽略版本控制目录、虚拟环境、依赖目录、缓存和构建产物；
- `list_files` 最大递归深度为 10，单次最多返回 1,000 项；
- 单个文件不能超过 256 KiB；
- 写入现有文件必须显式设置 `overwrite=true`；
- 写入内容最大为 256 KiB，并按 UTF-8 实际字节数计算；
- `write_file` 返回写前和写后的 SHA-256，便于审计变更；
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
- 模型能够调用 `write_file` 创建文件，再使用 `read_file` 回读验证。

#### `tests/test_config.py`

测试配置读取逻辑，包括缺少 API Key 时拒绝启动，以及正确读取模型名称和推理开关。

#### `tests/test_llm_client.py`

通过 Mock 对象测试 DeepSeek 适配层是否：

- 构造正确的模型请求；
- 启用 Tool Calling 和推理配置；
- 正确解析模型返回的工具名称、参数和结束原因。

测试不会发起真实网络请求。

#### `tests/test_tools.py`

测试 `read_file`、`list_files`、`write_file` 和工具注册器，包括：

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
- 拒绝敏感路径、目录目标和超出 UTF-8 字节上限的内容。

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

列出运行和开发所需的全部 Python 依赖，可用于一次性安装 DeepSeek 客户端、Pydantic、Typer、Rich、pytest、覆盖率工具和 Ruff。

#### `README.md`

当前文件，负责说明项目定位、已实现能力、目录结构和各文件职责。

## 安装依赖

在项目根目录创建并激活虚拟环境后执行：

```powershell
pip install -r requirement.txt
```

## 运行 Agent

当前实现只从进程环境变量读取配置，不会自动解析 `.env`。在 PowerShell 中加载 `.env`：

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

验证写入和回读闭环：

```powershell
python -m coding_agent run `
  "创建 hello.py，使文件内容为 print('Hello, Agent!')，然后重新读取并确认内容正确" `
  --workspace .\examples\demo
```

该任务会在演示工作区创建 `hello.py`。如果文件已经存在，模型必须显式传入
`overwrite=true` 才能覆盖。

## 运行测试

```powershell
pytest
```

静态检查：

```powershell
ruff check .
```
