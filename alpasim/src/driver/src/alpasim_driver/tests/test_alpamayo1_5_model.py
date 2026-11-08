# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from ..models.alpamayo1_5_model import Alpamayo15Model


def test_alpamayo1_5_loads_with_supported_eager_attention(monkeypatch) -> None:
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model
    from_pretrained = MagicMock(return_value=loaded_model)

    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo1_5_model.Alpamayo1_5.from_pretrained",
        from_pretrained,
    )
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo1_5_model.helper.get_processor",
        MagicMock(),
    )
    monkeypatch.setattr(Alpamayo15Model, "_init_common", MagicMock())

    Alpamayo15Model(
        checkpoint_path="nvidia/Alpamayo-1.5-10B",
        device=torch.device("cpu"),
        camera_ids=["camera_front_wide_120fov"],
    )

    assert from_pretrained.call_args.kwargs["attn_implementation"] == "eager"
