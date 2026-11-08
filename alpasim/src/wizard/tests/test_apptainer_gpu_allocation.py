from types import SimpleNamespace

import pytest

from alpasim_wizard.deployment.apptainer import ApptainerDeployment


@pytest.mark.parametrize("allocation,index,expected", [("3", 0, "3"), ("2,6", 1, "6"), ("GPU-abc", 0, "GPU-abc")])
def test_container_uses_slurm_assigned_device(monkeypatch, allocation, index, expected):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", allocation)
    deployment = ApptainerDeployment.__new__(ApptainerDeployment)
    monkeypatch.setattr(deployment, "_resolve_image", lambda _: "base.sif")
    container = SimpleNamespace(gpu=index, volumes=[], workdir="/repo", environments=[], command="run")
    assert f"--env CUDA_VISIBLE_DEVICES={expected}" in deployment._to_apptainer_exec(container)


def test_container_rejects_gpu_outside_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    deployment = ApptainerDeployment.__new__(ApptainerDeployment)
    monkeypatch.setattr(deployment, "_resolve_image", lambda _: "base.sif")
    container = SimpleNamespace(gpu=1)
    with pytest.raises(ValueError, match="exceeds"):
        deployment._to_apptainer_exec(container)
