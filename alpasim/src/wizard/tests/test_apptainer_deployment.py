# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from alpasim_wizard.deployment.apptainer import ApptainerDeployment
from alpasim_wizard.deployment import dispatcher


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.wait_timeouts: list[float] = []

    def wait(self, timeout: float) -> int:
        self.wait_timeouts.append(timeout)
        return 0


def _lifecycle_deployment(tmp_path: Path) -> ApptainerDeployment:
    deployment = ApptainerDeployment.__new__(ApptainerDeployment)
    deployment.context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                log_dir=str(tmp_path),
                dry_run=False,
                timeout=1,
            )
        )
    )
    deployment.container_set = SimpleNamespace(
        sim=[SimpleNamespace(uuid="driver-0", name="driver", command="serve")],
        runtime=SimpleNamespace(uuid="runtime-0", name="runtime", command="run"),
    )
    return deployment


def test_nre_command_uses_current_workdir_and_apptainer_home(tmp_path: Path) -> None:
    image = tmp_path / "nre_ga_26.04.sif"
    image.touch()

    deployment = ApptainerDeployment.__new__(ApptainerDeployment)
    deployment.context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                log_dir=str(tmp_path),
                sif_caches=[str(tmp_path)],
            )
        )
    )
    container = SimpleNamespace(
        name="renderer",
        service_config=SimpleNamespace(image="nvcr.io/nvidia/nre/nre-ga:26.04"),
        service_instances=[SimpleNamespace(address=SimpleNamespace(port=6014))],
        gpu=0,
        volumes=[],
        workdir=None,
        environments=["HOME=/tmp", "XDG_CACHE_HOME=/tmp/.cache"],
        command="/app/run serve-grpc --port=6014",
    )

    command = deployment._to_apptainer_exec(container)

    assert "--pwd /app" in command
    assert "--home /tmp" in command
    assert "--env HOME=/tmp" not in command
    assert "pycena_run.runfiles/_main" not in command
    assert str(image) in command


def test_nonblocking_dispatch_returns_process_in_dedicated_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(pid=1234)
    popen_kwargs = {}

    def fake_popen(command, **kwargs):
        del command
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)

    result = dispatcher.dispatch_command(
        "serve-forever",
        log_dir=tmp_path,
        blocking=False,
        start_new_session=True,
    )

    assert result is process
    assert popen_kwargs["start_new_session"] is True


def test_apptainer_cleans_up_services_after_runtime_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _lifecycle_deployment(tmp_path)
    process = _FakeProcess(pid=1234)
    events = []

    monkeypatch.setattr(
        deployment, "_to_apptainer_exec", lambda container: container.uuid
    )
    monkeypatch.setattr(
        deployment,
        "_wait_for_containers",
        lambda containers: events.append(("ready", [c.uuid for c in containers])),
    )

    def fake_dispatch(command, *, log_dir, dry_run, blocking, start_new_session=False):
        del log_dir, dry_run
        events.append(("dispatch", command, blocking, start_new_session))
        return "" if blocking else process

    monkeypatch.setattr(
        "alpasim_wizard.deployment.apptainer.dispatch_command", fake_dispatch
    )
    monkeypatch.setattr(
        deployment,
        "_cleanup_background_processes",
        lambda processes: events.append(
            ("cleanup", [(name, item.pid) for name, item in processes])
        ),
    )

    deployment.deploy_simulation()

    assert events == [
        ("dispatch", "driver-0", False, True),
        ("ready", ["driver-0"]),
        ("dispatch", "runtime-0", True, False),
        ("cleanup", [("driver-0", 1234)]),
    ]


def test_apptainer_cleans_up_services_when_startup_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _lifecycle_deployment(tmp_path)
    process = _FakeProcess(pid=1234)
    cleaned_up = []

    monkeypatch.setattr(
        deployment, "_to_apptainer_exec", lambda container: container.uuid
    )
    monkeypatch.setattr(
        deployment,
        "_wait_for_containers",
        lambda containers: (_ for _ in ()).throw(TimeoutError("not ready")),
    )

    def fake_dispatch(command, *, log_dir, dry_run, blocking, start_new_session=False):
        del command, log_dir, dry_run
        assert not blocking
        assert start_new_session
        return process

    monkeypatch.setattr(
        "alpasim_wizard.deployment.apptainer.dispatch_command", fake_dispatch
    )
    monkeypatch.setattr(
        deployment,
        "_cleanup_background_processes",
        lambda processes: cleaned_up.extend(processes),
    )

    with pytest.raises(TimeoutError, match="not ready"):
        deployment.deploy_simulation()

    assert cleaned_up == [("driver-0", process)]


def test_apptainer_cleanup_error_does_not_mask_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    deployment = _lifecycle_deployment(tmp_path)
    process = _FakeProcess(pid=1234)

    monkeypatch.setattr(
        deployment, "_to_apptainer_exec", lambda container: container.uuid
    )
    monkeypatch.setattr(deployment, "_wait_for_containers", lambda containers: None)

    def fake_dispatch(command, *, log_dir, dry_run, blocking, start_new_session=False):
        del log_dir, dry_run
        if blocking:
            raise RuntimeError("runtime failed")
        assert start_new_session
        return process

    monkeypatch.setattr(
        "alpasim_wizard.deployment.apptainer.dispatch_command", fake_dispatch
    )
    monkeypatch.setattr(
        deployment,
        "_cleanup_background_processes",
        lambda processes: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with caplog.at_level(logging.ERROR, logger="alpasim_wizard.deployment.apptainer"):
        with pytest.raises(RuntimeError, match="runtime failed"):
            deployment.deploy_simulation()

    assert "Unexpected error while cleaning up Apptainer services" in caplog.text
    assert "cleanup failed" in caplog.text


def test_apptainer_cleanup_escalates_stuck_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _lifecycle_deployment(tmp_path)
    process = _FakeProcess(pid=1234)
    signals = []

    monkeypatch.setattr(
        "alpasim_wizard.deployment.apptainer.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    monkeypatch.setattr(deployment, "_process_group_exists", lambda pid: True)

    deployment._cleanup_background_processes(
        [("driver-0", process)],
        timeout_s=0,
    )

    assert signals == [
        (1234, signal.SIGTERM),
        (1234, signal.SIGKILL),
    ]
    assert process.wait_timeouts == [1.0]
