"""Command-line entry point."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from coding_agent.agent import Agent, AgentError, AgentResult
from coding_agent.checkpoints import (
    ChangeSet,
    CheckpointError,
    CheckpointManager,
    RestoreResult,
)
from coding_agent.config import ConfigurationError, DeepseekConfig, QianwenConfig
from coding_agent.context import ContextConfig
from coding_agent.llm_client import DeepSeekClient, QianwenClient
from coding_agent.planning import PlanError, PlanExecuteAgent
from coding_agent.protocol import ChatClient
from coding_agent.sessions import JsonSessionStore, SessionError

app = typer.Typer(
    name="coding-agent",
    help="A minimal coding agent powered by model-native tool calling.",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"quit", "exit", "q", "退出"}


class Provider(str, Enum):
    deepseek = "deepseek"
    qianwen = "qianwen"


class AgentMode(str, Enum):
    react = "react"
    plan_execute = "plan-execute"


def create_client(provider: Provider) -> tuple[ChatClient, str]:
    if provider == Provider.deepseek:
        config = DeepseekConfig.from_env()
        return DeepSeekClient(config), config.model

    if provider == Provider.qianwen:
        config = QianwenConfig.from_env()
        return QianwenClient(config), config.model

    raise ConfigurationError(f"Unsupported provider: {provider}")


@app.callback()
def main() -> None:
    """coding-agent command group."""


@app.command()
def run(
    task: Annotated[
        str | None,
        typer.Argument(
            help="Programming task; omit it to enter interactive mode"
        ),
    ] = None,
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace the agent may inspect",
        ),
    ] = Path("."),
    max_steps: Annotated[
        int,
        typer.Option(
            min=1,
            max=100,
            help="Maximum number of model turns",
        ),
    ] = 20,
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            "-p",
            "--model",
            "-m",
            help="Choose LLM provider (the --model alias is kept for compatibility)",
        ),
    ] = Provider.deepseek,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            "-s",
            help="Create or resume a JSON session in the workspace",
        ),
    ] = None,
    agent_mode: Annotated[
        AgentMode,
        typer.Option(
            "--agent-mode",
            "-a",
            help="Choose react or plan-execute orchestration",
        ),
    ] = AgentMode.react,
) -> None:
    """Run one task or start an interactive coding-agent session."""
    try:
        resolved_workspace = workspace.resolve(strict=True)
        if not resolved_workspace.is_dir():
            raise ValueError(f"Not a directory: {workspace}")

        client, model_name = create_client(provider)
        session_store = (
            JsonSessionStore(resolved_workspace) if session is not None else None
        )
        checkpoint_manager = CheckpointManager(resolved_workspace)
        agent_class = Agent if agent_mode == AgentMode.react else PlanExecuteAgent
        agent = agent_class(
            client,
            resolved_workspace,
            max_steps=max_steps,
            context_config=ContextConfig.from_env(),
            session_store=session_store,
            session_id=session,
            provider=provider.value if session is not None else None,
            model=model_name if session is not None else None,
            checkpoint_manager=checkpoint_manager,
        )

        console.print(
            f"[bold cyan]coding-agent[/bold cyan]  "
            f"provider={provider.value}  "
            f"model={model_name}  "
            f"mode={agent_mode.value}  "
            f"workspace={resolved_workspace}"
            + (f"  session={session}" if session is not None else "")
        )

        if task is None:
            _run_interactive(
                agent,
                checkpoint_manager,
                persistent=session is not None,
            )
            return

        result = agent.run(task)

    except (
        ConfigurationError,
        AgentError,
        CheckpointError,
        PlanError,
        SessionError,
        OSError,
        ValueError,
    ) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=130) from exc

    _print_result(result)


def _run_interactive(
    agent: Agent,
    checkpoint_manager: CheckpointManager,
    *,
    persistent: bool,
) -> None:
    console.print(
        "[bold green]Interactive mode[/bold green]  "
        "输入编程任务，输入 [bold]quit[/bold]、[bold]exit[/bold] 或"
        " [bold]退出[/bold] 结束。"
    )
    console.print(
        "[dim]Checkpoint commands: /diff, /undo, /checkpoints[/dim]"
    )
    if not persistent:
        console.print(
            "[yellow]未提供 --session；每个问题将使用独立的对话历史。[/yellow]"
        )

    while True:
        try:
            task = console.input("\n[bold cyan]>>[/bold cyan] ").strip()
        except EOFError:
            console.print("\n[yellow]输入已结束。[/yellow]")
            return

        if task.casefold() in EXIT_COMMANDS:
            console.print("[yellow]会话已结束。[/yellow]")
            return
        if not task:
            continue
        if task.startswith("/"):
            try:
                _handle_checkpoint_command(task, checkpoint_manager, agent)
            except (CheckpointError, OSError, ValueError) as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
            continue

        try:
            result = agent.run(task)
        except (
            AgentError,
            CheckpointError,
            PlanError,
            SessionError,
            OSError,
            ValueError,
        ) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            continue

        _print_result(result)


def _print_result(result: AgentResult) -> None:
    console.print("\n[bold green]Completed[/bold green]")
    console.print(result.final_answer)
    console.print(
        f"\n[dim]steps={result.steps} tool_calls={result.tool_calls}[/dim]"
    )
    if result.checkpoint_id is not None:
        console.print(
            f"[dim]checkpoint={result.checkpoint_id} "
            f"changed_files={len(result.changes)}[/dim]"
        )
    if result.plan_id is not None:
        console.print(f"[dim]plan={result.plan_id}[/dim]")


def _handle_checkpoint_command(
    command: str,
    checkpoint_manager: CheckpointManager,
    agent: Agent,
) -> None:
    parts = command.split()
    name = parts[0].casefold()
    if len(parts) > 2:
        raise ValueError(f"Too many arguments for {parts[0]}")
    checkpoint_id = parts[1] if len(parts) == 2 else None

    if name == "/diff":
        _print_changes(checkpoint_manager.diff(checkpoint_id))
        return
    if name == "/checkpoints":
        if checkpoint_id is not None:
            raise ValueError("/checkpoints does not accept an id")
        checkpoints = checkpoint_manager.list()
        if not checkpoints:
            console.print("[yellow]No checkpoints are available.[/yellow]")
            return
        for checkpoint in checkpoints[:20]:
            console.print(
                f"[cyan]{checkpoint.checkpoint_id}[/cyan]  "
                f"{checkpoint.kind}  {checkpoint.created_at}  "
                f"{checkpoint.task}"
            )
        return
    if name == "/undo":
        target = (
            checkpoint_manager.get(checkpoint_id)
            if checkpoint_id is not None
            else checkpoint_manager.latest()
        )
        changes = checkpoint_manager.diff(target.checkpoint_id)
        _print_changes(changes)
        answer = console.input(
            f"[yellow]Restore {target.checkpoint_id}? [y/N][/yellow] "
        ).strip()
        if answer.casefold() not in {"y", "yes"}:
            console.print("[yellow]Undo cancelled.[/yellow]")
            return
        result = checkpoint_manager.restore(target.checkpoint_id)
        _print_restore(result)
        agent.record_workspace_restore(result)
        console.print("[dim]Workspace restore event recorded.[/dim]")
        return
    raise ValueError(
        "Unknown command. Available commands: /diff, /undo, /checkpoints"
    )


def _print_changes(change_set: ChangeSet) -> None:
    if not change_set.changes:
        console.print(
            f"[green]No changes since {change_set.checkpoint_id}.[/green]"
        )
        return
    labels = {
        "added": "A",
        "modified": "M",
        "deleted": "D",
        "renamed": "R",
    }
    console.print(f"[bold]Changes since {change_set.checkpoint_id}:[/bold]")
    for change in change_set.changes:
        if change.status == "renamed":
            console.print(f"  R {change.old_path} -> {change.path}")
        else:
            console.print(f"  {labels[change.status]} {change.path}")
        if change.patch:
            console.print(change.patch, markup=False, highlight=False)


def _print_restore(result: RestoreResult) -> None:
    console.print(
        f"[bold green]Restored {result.checkpoint_id}[/bold green]  "
        f"restored_files={result.restored_files}  "
        f"removed_files={result.removed_files}"
    )
    console.print(
        f"[dim]Safety checkpoint: {result.safety_checkpoint_id}. "
        "Conversation messages were retained.[/dim]"
    )


if __name__ == "__main__":
    app()
