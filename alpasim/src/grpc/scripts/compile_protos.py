#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Compile proto files to Python modules."""
import os

from grpc_tools import command


def clean_proto_files() -> None:
    """Delete all generated proto files (*.py except __init__.py and *.pyi)."""
    # Clean files in alpasim_grpc/v0 directory
    for root, unused_dirs, files in os.walk("alpasim_grpc/v0"):
        for file in files:
            if (file.endswith(".py") and file != "__init__.py") or file.endswith(
                ".pyi"
            ):
                file_path = os.path.join(root, file)
                print(f"Deleting {file_path}")
                os.remove(file_path)


def compile_protos() -> None:
    # First clean old proto files
    print("Cleaning old proto files...")
    clean_proto_files()

    # Use the same grpc_tools.command API for exact compatibility
    command.build_package_protos(".", strict_mode=True)
    print("Proto compilation completed successfully!")
