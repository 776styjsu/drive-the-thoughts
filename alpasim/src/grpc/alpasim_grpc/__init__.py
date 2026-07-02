# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import importlib.metadata

from alpasim_grpc.v0.common_pb2 import VersionId

version_str = importlib.metadata.version("alpasim_grpc")

if version_str is None:
    raise RuntimeError("Could not find the version of the alpasim_grpc package")

__version__ = tuple(int(v) for v in version_str.split("."))  # (0, 0, 0)

API_VERSION_MESSAGE = VersionId.APIVersion(
    major=__version__[0],
    minor=__version__[1],
    patch=__version__[2],
)
