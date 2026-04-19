# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import attrs
import wandb
import wandb.util
from omegaconf import DictConfig

from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed, log
from imaginaire.utils.easy_io import easy_io

if TYPE_CHECKING:
    from imaginaire.config import CheckpointConfig, Config, JobConfig
    from imaginaire.model import ImaginaireModel


@distributed.rank0_only
def init_wandb(config: Config, model: ImaginaireModel) -> None:
    """Initialize Weights & Biases (wandb) logger.

    Args:
        config (Config): The config object for the Imaginaire codebase.
        model (ImaginaireModel): The PyTorch model.
    """
    if isinstance(config.job, DictConfig):
        from imaginaire.config import JobConfig

        config_job = JobConfig(**config.job)
    else:
        config_job = config.job
    config_checkpoint = config.checkpoint
    # Try to fetch the W&B job ID for resuming training.
    wandb_id = _read_wandb_id(config_job, config_checkpoint)
    if wandb_id is None:
        # Generate a new W&B job ID.
        wandb_id = wandb.util.generate_id()
        _write_wandb_id(config_job, config_checkpoint, wandb_id=wandb_id)
        log.info(f"Generating new wandb ID: {wandb_id}")
    else:
        log.info(f"Resuming with existing wandb ID: {wandb_id}")
    # refactor config so that wandb better understands it
    local_safe_yaml_fp = LazyConfig.save_yaml(config, os.path.join(config_job.path_local, "config.yaml"))
    if os.path.exists(local_safe_yaml_fp):
        config_resolved = easy_io.load(local_safe_yaml_fp)
    else:
        config_resolved = attrs.asdict(config)
    # Initialize the wandb library.
    wandb.init(
        force=True,
        id=wandb_id,
        project=config_job.project,
        group=config_job.group,
        name=config_job.name,
        config=config_resolved,
        dir=config_job.path_local,
        resume="allow",
        mode=config_job.wandb_mode,
    )


def _read_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig) -> str | None:
    """Read the W&B job ID. If it doesn't exist, return None.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.

    Returns:
        wandb_id (str | None): W&B job ID.
    """
    wandb_id = None
    wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
    if os.path.isfile(wandb_id_path):
        with open(wandb_id_path) as f:
            wandb_id = f.read().strip()
    return wandb_id


def _write_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig, wandb_id: str) -> None:
    """Write the generated W&B job ID.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        wandb_id (str): The W&B job ID.
    """
    content = f"{wandb_id}\n"
    wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
    with open(wandb_id_path, "w") as file:
        file.write(content)
