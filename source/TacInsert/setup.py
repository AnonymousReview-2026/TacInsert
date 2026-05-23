# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'TacInsert' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(
    os.path.join(EXTENSION_PATH, "config", "extension.toml")
)

# Runtime dependencies are provided by Isaac Lab extension dependencies.
INSTALL_REQUIRES = []

# Installation operation
setup(
    name="TacInsert",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={
        "TacInsert": [
            "tasks/direct/tacinsert/README.md",
            "tasks/direct/tacinsert/agents/*.yaml",
            "tasks/direct/tacinsert/assets/*/*.usd",
            "tasks/direct/tacinsert/assets/*/*/*.usd",
            "tasks/direct/tacinsert/checkpoints/README.md",
            "tasks/direct/tacinsert/figures/*.png",
            "tasks/direct/tacinsert/sim2real/*.py",
            "tasks/direct/tacinsert/sim2real/README.md",
            "tasks/direct/tacinsert/sim2real/configs/*.yaml",
            "tasks/direct/tacinsert/sim2real/checkpoints/README.md",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
