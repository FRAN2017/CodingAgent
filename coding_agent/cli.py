"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from coding_agent.agent import Agent, AgentError
from coding_agent.config import Config, ConfigurationError
from coding_agent.llm_client import DeepSeekClient

app = typer.Typer(
    name="coding-agent",
    help="A minimal coding agent powered by DeepSeek tool calling.",
    no_args_is_help=True,
)
console = Console()


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
    ] = 10,
) -> None:
    """Run one agent task."""
    try:
        resolved_workspace = workspace.resolve(strict=True)
        if not resolved_workspace.is_dir():
            raise ValueError(f"Not a directory: {workspace}")

        config = Config.from_env()
        client = DeepSeekClient(config)
        agent = Agent(client, resolved_workspace, max_steps=max_steps)

        console.print(
            f"[bold cyan]coding-agent[/bold cyan]  "
            f"model={config.model}  workspace={resolved_workspace}"
        )
        result = agent.run(task)
    except (ConfigurationError, AgentError, OSError, ValueError) as exc:
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
