"""Safe file renaming inside the workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from coding_agent.tools.base import ToolSpec
from coding_agent.tools.workspace import (
    is_ignored_name,
    path_uses_symlink,
    resolve_workspace_path,
    workspace_relative_path,
)


class RenameFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=1,
        description="Workspace-relative path of the existing file",
    )
    destination: str = Field(
        min_length=1,
        description="Workspace-relative new path for the file",
    )
    overwrite: bool = Field(
        default=False,
        description="Must be true to replace an existing destination file",
    )
    create_parent_dirs: bool = Field(
        default=False,
        description="Create missing destination parent directories when true",
    )


def rename_file(workspace: Path, arguments: RenameFileInput) -> dict[str, Any]:
    workspace = workspace.resolve()
    source = resolve_workspace_path(workspace, arguments.source)
    destination = resolve_workspace_path(workspace, arguments.destination)
    relative_source = source.relative_to(workspace)
    relative_destination = destination.relative_to(workspace)

    if source == destination:
        return {
            "ok": False,
            "error": "Source and destination must be different paths",
        }
    if relative_source == Path(".") or relative_destination == Path("."):
        return {"ok": False, "error": "Workspace directory cannot be renamed"}
    if any(is_ignored_name(part) for part in relative_source.parts):
        return {
            "ok": False,
            "error": f"Renaming an ignored or sensitive path is not allowed: {arguments.source}",
        }
    if any(is_ignored_name(part) for part in relative_destination.parts):
        return {
            "ok": False,
            "error": (
                "Renaming into an ignored or sensitive path is not allowed: "
                f"{arguments.destination}"
            ),
        }
    if path_uses_symlink(workspace, arguments.source):
        return {
            "ok": False,
            "error": f"Renaming through a symbolic link is not allowed: {arguments.source}",
        }
    if path_uses_symlink(workspace, arguments.destination):
        return {
            "ok": False,
            "error": (
                "Renaming through a symbolic link is not allowed: "
                f"{arguments.destination}"
            ),
        }

    if not source.exists():
        return {"ok": False, "error": f"Source file does not exist: {arguments.source}"}
    if not source.is_file():
        return {"ok": False, "error": f"Source is not a file: {arguments.source}"}

    destination_existed = destination.exists()
    if destination_existed and not destination.is_file():
        return {
            "ok": False,
            "error": f"Destination is not a file: {arguments.destination}",
        }
    if destination_existed and not arguments.overwrite:
        return {
            "ok": False,
            "error": (
                f"Destination file already exists: {arguments.destination}. "
                "Set overwrite=true to replace it."
            ),
        }

    parent = destination.parent
    parent_dirs_created = False
    if not parent.exists():
        if not arguments.create_parent_dirs:
            return {
                "ok": False,
                "error": (
                    "Destination parent directory does not exist: "
                    f"{workspace_relative_path(workspace, parent)}"
                ),
            }
        parent.mkdir(parents=True)
        parent_dirs_created = True
    elif not parent.is_dir():
        return {
            "ok": False,
            "error": (
                "Destination parent path is not a directory: "
                f"{workspace_relative_path(workspace, parent)}"
            ),
        }

    bytes_moved = source.stat().st_size
    replaced_bytes = destination.stat().st_size if destination_existed else None
    os.replace(source, destination)

    return {
        "ok": True,
        "source": workspace_relative_path(workspace, source),
        "destination": workspace_relative_path(workspace, destination),
        "action": "replaced" if destination_existed else "renamed",
        "bytes_moved": bytes_moved,
        "replaced_bytes": replaced_bytes,
        "parent_dirs_created": parent_dirs_created,
    }


RENAME_FILE_TOOL = ToolSpec(
    name="rename_file",
    description=(
        "Rename or move one file inside the workspace. The source file is removed "
        "from its old path. Replacing an existing destination requires explicit "
        "overwrite=true."
    ),
    input_model=RenameFileInput,
    handler=rename_file,
)
