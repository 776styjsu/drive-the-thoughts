# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Apptainer deployment strategy."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from ..context import WizardContext
from ..services import ContainerDefinition, build_container_set
from ..utils import image_url_to_apptainer_image
from .dispatcher import dispatch_command

logger = logging.getLogger(__name__)


class ApptainerDeployment:
    """Deployment strategy using Apptainer (formerly Singularity).

    Generates ``apptainer exec`` commands for each container, with
    ``--nv`` for GPU passthrough, ``--bind`` for volumes, ``--env`` for
    environment variables, and ``--pwd`` for the working directory.

    Images are resolved by searching pre-built ``.sif`` files in
    ``wizard.sif_caches`` or falling back to a ``docker://`` URI so that
    Apptainer can pull and convert the image at runtime.
    """

    def __init__(self, context: WizardContext):
        """Initialize with context and build container set.

        Args:
            context: The wizard context
        """
        self.context = context
        self.container_set = build_container_set(context, use_address_string="0.0.0.0")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy_all_services(self) -> None:
        """Deploy all services (simulation, evaluation, aggregation)."""
        self.deploy_simulation()
        self.deploy_evaluation()
        self.deploy_aggregation()

    def generate_run_script(self) -> None:
        """Generate a ``run.sh`` script that runs all phases sequentially."""
        sim_cmds = self._build_phase_commands(
            self.container_set.sim, self.container_set.runtime
        )
        eval_cmds = self._build_phase_commands(self.container_set.eval)
        agg_cmds = self._build_phase_commands(self.container_set.agg)

        script_lines = [
            "#!/bin/bash",
            "set -e",
            "",
            "# --- Simulation phase ---",
        ]
        script_lines.extend(sim_cmds)
        script_lines += ["", "# --- Evaluation phase ---"]
        script_lines.extend(eval_cmds)
        script_lines += ["", "# --- Aggregation phase ---"]
        script_lines.extend(agg_cmds)
        script_lines += ["", 'echo "All phases completed successfully!"']

        run_script_path = Path(self.context.cfg.wizard.log_dir) / "run.sh"
        run_script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(run_script_path, "w") as f:
            f.write("\n".join(script_lines) + "\n")
        os.chmod(run_script_path, 0o755)
        logger.info("Generated Apptainer run script: %s", run_script_path)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def deploy_simulation(self) -> None:
        """Deploy simulation services (including runtime)."""
        logger.info("Running simulation services via Apptainer")
        containers = list(self.container_set.sim or [])
        containers_last = list(self.container_set.runtime or [])

        for c in containers:
            if c.command == "noop":
                continue
            cmd = self._to_apptainer_exec(c)
            dispatch_command(
                cmd,
                log_dir=Path(self.context.cfg.wizard.log_dir),
                dry_run=self.context.cfg.wizard.dry_run,
                blocking=False,
            )

        # Wait for services to come up before launching runtime
        if containers_last and not self.context.cfg.wizard.dry_run:
            self._wait_for_containers(containers)

        for c in containers_last:
            cmd = self._to_apptainer_exec(c)
            dispatch_command(
                cmd,
                log_dir=Path(self.context.cfg.wizard.log_dir),
                dry_run=self.context.cfg.wizard.dry_run,
                blocking=True,
            )

    def deploy_evaluation(self) -> None:
        """Deploy evaluation services."""
        logger.info("Running evaluation services via Apptainer")
        for c in self.container_set.eval or []:
            cmd = self._to_apptainer_exec(c)
            dispatch_command(
                cmd,
                log_dir=Path(self.context.cfg.wizard.log_dir),
                dry_run=self.context.cfg.wizard.dry_run,
                blocking=True,
            )

    def deploy_aggregation(self) -> None:
        """Deploy aggregation services."""
        logger.info("Running aggregation services via Apptainer")
        for c in self.container_set.agg or []:
            cmd = self._to_apptainer_exec(c)
            dispatch_command(
                cmd,
                log_dir=Path(self.context.cfg.wizard.log_dir),
                dry_run=self.context.cfg.wizard.dry_run,
                blocking=True,
            )

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _resolve_image(self, container: ContainerDefinition) -> str:
        """Resolve the container image to a ``.sif`` path or ``docker://`` URI.

        Tries pre-built Apptainer images in ``wizard.sif_caches`` first. Cached
        images may be either ``.sif`` files or sandbox directories. Falls back
        to a ``docker://`` URI so Apptainer pulls at runtime.
        """
        sif_caches: list[str] = getattr(self.context.cfg.wizard, "sif_caches", []) or []
        if sif_caches:
            try:
                return image_url_to_apptainer_image(
                    container.service_config.image, sif_caches
                )
            except ValueError:
                pass  # fall through to docker:// URI

        # Fallback: let Apptainer pull from Docker registry
        return f"docker://{container.service_config.image}"

    def _to_apptainer_exec(self, container: ContainerDefinition) -> str:
        """Generate an ``apptainer exec`` command for a container.

        Args:
            container: ContainerDefinition instance

        Returns:
            Full apptainer exec command string
        """
        image = self._resolve_image(container)

        # Prepend module load so apptainer is available on HPC modules
        parts: list[str] = [
            "module load apptainer >/dev/null 2>&1 || module load apptainer/1.3.4 >/dev/null 2>&1 || true;",
            "apptainer exec"
        ]

        # GPU support
        if container.gpu is not None:
            parts.append("--nv")
            parts.append(f"--env CUDA_VISIBLE_DEVICES={container.gpu}")

        # Volume mounts
        for v in container.volumes:
            parts.append(f"--bind {v.to_str()}")

        # Set the working directory based on the container image
        is_old_nurec = "nvidia_nurec" in image
        is_new_nre = any(s in image for s in ("nre_ga", "nre-ga", "/nre/"))

        if container.workdir:
            parts.append(f"--pwd {container.workdir}")
        elif is_old_nurec:
            parts.append("--pwd /app/pycena_run.runfiles/nre_repo")
        elif is_new_nre:
            parts.append("--pwd /app/internal/scripts/pycena/runtime/pycena_run.runfiles/_main")
        else:
            parts.append("--pwd /repo")

        # Environment variables
        for e in container.environments or []:
            if "=" in e:
                parts.append(f"--env {e}")
            else:
                # Pass-through from host: VAR → --env VAR=$VAR
                parts.append(f"--env {e}=${{{e}}}")

        if not (is_old_nurec or is_new_nre):
            # Force uv to use the container's environment instead of the host's $PWD/.venv
            parts.append("--env UV_PROJECT_ENVIRONMENT=/repo/.venv")
            # Prevent Python from writing __pycache__ files over the slow FUSE/network mount
            parts.append("--env PYTHONDONTWRITEBYTECODE=1")

        # Writable scratch for runtime writes outside bind-mounts.
        # NRE containers need a large writable area because the entrypoint
        # creates a Bazel-runfiles venv with thousands of symlinks; this
        # overflows --writable-tmpfs (kernel tmpfs is small; symlink returns
        # ENOMEM). A bind mount at the venv root also fails because the venv
        # tool wipes its own root before recreating, and you can't rmdir a
        # mount point (EBUSY). A directory --overlay tends to fail on GPFS
        # since overlayfs uppers need xattr support GPFS lacks.
        # File-backed --overlay (ext3 in a regular file on GPFS) sidesteps
        # all of these: full POSIX semantics regardless of underlying FS,
        # and the venv root inside the overlay is a plain directory the tool
        # can remove and recreate freely. We create the .img lazily in the
        # dispatched bash command so re-runs are idempotent.
        if is_old_nurec or is_new_nre:
            port = container.service_instances[0].address.port
            overlay_img = (
                Path(self.context.cfg.wizard.log_dir)
                / "nre_overlays"
                / f"{container.name}_{port}.img"
            )
            overlay_img.parent.mkdir(parents=True, exist_ok=True)
            create_step = (
                f"[ -f {overlay_img} ] || "
                f"apptainer overlay create --size 4096 {overlay_img};"
            )
            parts.insert(1, create_step)
            parts.append(f"--overlay {overlay_img}")
        else:
            parts.append("--writable-tmpfs")

        # Image
        parts.append(image)

        # Command
        if container.command and container.command != "noop":
            escaped_command = container.command.replace(r"\$", "$")
            parts.append(f'bash -c "{escaped_command}"')

        return " \\\n  ".join(parts)

    # ------------------------------------------------------------------
    # Script generation helpers
    # ------------------------------------------------------------------

    def _build_phase_commands(
        self,
        containers: Optional[List[Any]],
        extra_containers: Optional[List[Any]] = None,
    ) -> list[str]:
        """Build shell command lines for a deployment phase.

        Background services are launched with ``&`` and the last container
        (or the extra containers) run in the foreground.
        """
        lines: list[str] = []
        all_containers = list(containers or [])
        extras = list(extra_containers or [])

        # Background services
        for c in all_containers:
            if c.command == "noop":
                continue
            cmd = self._to_apptainer_exec(c)
            lines.append(f"{cmd} &")

        if extras:
            lines.append("")
            lines.append("# Wait for background services to become ready")
            lines.append("sleep 10")
            lines.append("")

        # Foreground / final services
        for c in extras:
            cmd = self._to_apptainer_exec(c)
            lines.append(cmd)

        return lines

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def _wait_for_containers(
        self,
        containers: List[ContainerDefinition],
        timeout: Optional[int] = None,
    ) -> None:
        """Wait for container service addresses to become reachable."""
        import time

        if timeout is None:
            timeout = self.context.cfg.wizard.timeout

        logger.info("Waiting for Apptainer services to become ready...")
        waited = 0
        for container in containers:
            for si in container.service_instances:
                if si.address is None:
                    continue
                while not si.address.is_open():
                    time.sleep(1)
                    waited += 1
                    if timeout is not None and waited > timeout:
                        raise TimeoutError(
                            f"Service {container.name} at {si.address} "
                            "did not become ready in time"
                        )
                logger.info("  %s ready.", si.address)
        logger.info("All Apptainer services ready.")
