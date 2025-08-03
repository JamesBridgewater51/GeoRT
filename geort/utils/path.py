# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

# GeoRT Package Path Helper.
# No more pain configuring paths!
def get_package_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve()

def to_package_root(path):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve() / path

def get_data_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "data").resolve()

def get_checkpoint_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "checkpoint").resolve()

def get_human_data_output_path(human_data):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "data" / human_data).resolve()

def get_human_data(name):
    data_root = Path(get_data_root())
    # `os.listdir` returns a list of files (with extensions) in the directory.
    all_data_names = os.listdir(data_root)
    # Filter out only *.npy, and remove exts.
    all_data_names = [f for f in all_data_names if f.endswith('.npy')]
    all_data_names = [f[:-4] for f in all_data_names]  # Remove '.npy' extension

    for data_name in all_data_names:
        if name == data_name:
            return data_root / (data_name + '.npy')


if __name__ == '__main__':
    print(get_package_root())
    print(get_human_data_output_path("human"))

