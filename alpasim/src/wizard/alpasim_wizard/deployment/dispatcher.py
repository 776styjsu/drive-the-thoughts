# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Command dispatcher with logging capabilities for the wizard."""

import logging
import subprocess
from pathlib import Path
from typing import Literal, Union, overload

logger = logging.getLogger("alpasim_wizard_dispatcher")


class OsDispatchError(RuntimeError):
    """Error raised when command execution fails."""

    pass


@overload
def dispatch_command(
    cmd: str,
    log_dir: Union[Path, str],
    dry_run: bool = False,
    blocking: Literal[True] = True,
    start_new_session: bool = False,
) -> str: ...


@overload
def dispatch_command(
    cmd: str,
    log_dir: Union[Path, str],
    dry_run: bool = False,
    blocking: Literal[False] = False,
    start_new_session: bool = False,
) -> subprocess.Popen[str] | None: ...


def dispatch_command(
    cmd: str,
    log_dir: Union[Path, str],
    dry_run: bool = False,
    blocking: bool = True,
    start_new_session: bool = False,
) -> str | subprocess.Popen[str] | None:
    """Execute a command with logging.

    Args:
        cmd: The command to execute
        log_dir: Directory for logging command output
        dry_run: If True, commands are logged but not executed
        blocking: If True, wait for completion and return output; if False, return immediately
        start_new_session: Start a non-blocking command in a dedicated process
            group so the caller can terminate it and all descendants. Ignored
            for blocking commands.

    Returns:
        Command output if blocking=True. If blocking=False, the running process
        handle, or None for a dry run.
    """
    log_dir = Path(log_dir)

    # Ensure log directories exist
    txt_logs_dir = log_dir / "txt-logs"
    txt_logs_dir.mkdir(parents=True, exist_ok=True)

    cmd_log_file = txt_logs_dir / "os_dispatch_log.txt"
    output_log_file = txt_logs_dir / "os_dispatch_output.txt"

    # Log the command
    with open(cmd_log_file, "a") as f:
        f.write(f"{cmd}\n")

    # Handle dry-run mode
    if dry_run:
        logger.info(f"[DRY-RUN] Would execute: {cmd}")
        return "" if blocking else None

    logger.info(f"Executing: {cmd}")

    # Execute the command with output logging
    with open(output_log_file, "a") as log_file:
        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"Command: {cmd}\n")
        log_file.write(f"{'=' * 60}\n")
        # Run the command and capture output
        try:
            # Blocking: pipe stdout so we can read it line-by-line.
            # Non-blocking: write directly to log file so srun/pyxis
            # errors aren't silently lost (the child inherits its own
            # fd, so closing log_file in the parent is safe).
            stdout_dest = subprocess.PIPE if blocking else log_file
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=stdout_dest,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=start_new_session if not blocking else False,
            )

            if blocking:
                output_lines = []
                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        log_file.write(line)
                        logger.debug(line.rstrip())
                        output_lines.append(line)
                    process.stdout.close()

                # Wait for completion and check return code
                return_code = process.wait()
                if return_code != 0:
                    raise OsDispatchError(
                        f"Command failed with return code {return_code}: {cmd}"
                    )

                return "".join(output_lines)
            else:
                logger.info(
                    "Started non-blocking process pid=%d "
                    f"(output → {output_log_file}): {cmd}",
                    process.pid,
                )
                return process

        except subprocess.SubprocessError as e:
            raise OsDispatchError(f"Failed to execute command '{cmd}': {e}")
