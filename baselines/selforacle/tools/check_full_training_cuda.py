"""CUDA-only numerical check using synthetic inputs, never benchmark frames."""

import argparse
import os
import tempfile
from pathlib import Path

import torch

from selforacle_baseline.data import sha256, write_json
from selforacle_baseline.full_training import augment
from selforacle_baseline.models import Reconstructor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("New output and Slurm CUDA allocation required")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    inputs = torch.rand((32, 80 * 160 * 3), device="cuda")
    sizes = torch.tensor([[1920, 1080]] * len(inputs), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(100000)
    augmented = augment(inputs, sizes, generator, 0.6)
    assert augmented.shape == inputs.shape and torch.isfinite(augmented).all()
    assert augmented.min() >= 0 and augmented.max() <= 1
    records = []
    with tempfile.TemporaryDirectory(prefix="selforacle-cuda-check-") as directory:
        for kind in ("sae", "vae"):
            config = {"model": kind, "height": 80, "width": 160}
            model = Reconstructor(config).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            reconstructed, penalty = model(augmented)
            loss = (reconstructed - augmented).square().mean() + penalty
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                expected = model(inputs)[0]
            path = Path(directory) / f"{kind}.pt"
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
            reloaded = Reconstructor(config).cuda()
            reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            reloaded.eval()
            with torch.inference_mode():
                actual = reloaded(inputs)[0]
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            records.append({"model": kind, "finite_loss": float(loss), "gradient_norm": float(norm),
                            "checkpoint_reload_identical": True})
            print(f"{kind}: CUDA augmentation, finite gradients and checkpoint reload passed", flush=True)
            del model, optimizer, reloaded, expected, actual, reconstructed, penalty, loss
            torch.cuda.empty_cache()
    write_json(args.output, {"status": "passed", "input_source": "synthetic_only_no_benchmark_access",
               "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "source_sha256": sha256(Path(__file__)), "checks": records})


if __name__ == "__main__":
    main()
