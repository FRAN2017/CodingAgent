"""Command-line entry point."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from coding_agent.agent import Agent, AgentError
from coding_agent.config import ConfigurationError, DeepseekConfig, QianwenConfig
from coding_agent.context import ContextConfig
from coding_agent.llm_client import DeepSeekClient, QianwenClient
from coding_agent.protocol import ChatClient
from coding_agent.sessions import JsonSessionStore, SessionError

app = typer.Typer(
    name="coding-agent",
    help="A minimal coding agent powered by model-native tool calling.",
    no_args_is_help=True,
)

console = Console()


class Provider(str, Enum):
    deepseek = "deepseek"
    qianwen = "qianwen"


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
        str, typer.Argument(help="Programming task for the agent")
    ],
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
) -> None:
    """Run one agent task."""
    try:
        resolved_workspace = workspace.resolve(strict=True)
        if not resolved_workspace.is_dir():
            raise ValueError(f"Not a directory: {workspace}")

        client, model_name = create_client(provider)
        session_store = (
            JsonSessionStore(resolved_workspace) if session is not None else None
        )
        agent = Agent(
            client,
            resolved_workspace,
            max_steps=max_steps,
            context_config=ContextConfig.from_env(),
            session_store=session_store,
            session_id=session,
            provider=provider.value if session is not None else None,
            model=model_name if session is not None else None,
        )

        console.print(
            f"[bold cyan]coding-agent[/bold cyan]  "
            f"provider={provider.value}  "
            f"model={model_name}  "
            f"workspace={resolved_workspace}"
            + (f"  session={session}" if session is not None else "")
        )

        result = agent.run(task)

    except (
        ConfigurationError,
        AgentError,
        SessionError,
        OSError,
        ValueError,
    ) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=130) from exc

    console.print("\n[bold green]Completed[/bold green]")
    console.print(result.final_answer)
    console.print(
        f"\n[dim]steps={result.steps} tool_calls={result.tool_calls}[/dim]"
    )


if __name__ == "__main__":
    app()
