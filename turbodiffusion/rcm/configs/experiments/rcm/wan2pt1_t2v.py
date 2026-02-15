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

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict


def build_debug_run(job):
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=6,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=6,
                    num_samples=5,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                tangent_warmup=8,
            )
        ),
    )


"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=rcm/configs/registry.py -- experiment="wan2pt1_1pt3B_res480p_t2v_rCM_debug"
"""
WAN2PT1_1PT3B_RES480P_T2V: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_distill_rcm"},
            {"override /net": "wan2pt1_1pt3B_t2v_jvp"},
            {"override /net_teacher": "wan2pt1_1pt3B_t2v"},
            {"override /net_fake_score": "wan2pt1_1pt3B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {"override /optimizer_fake_score": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan",
            name="wan2pt1_1pt3B_res480p_t2v_rCM",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                loss_scale_dmd=1.0,
                fsdp_shard_size=4,
                resolution="480p",
                timestep_shift=5,
                max_simulation_steps_fake=4,
                state_t=20,
                sigma_max=80,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=5,
                optimizer_fake_score=dict(
                    lr=4e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                teacher_ckpt="assets/checkpoints/Wan2.1-T2V-1.3B.dcp",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                teacher_guidance=5.0,
                tangent_warmup=1000,
                precision="bfloat16",
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                    num_samples=5,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    num_samples=5,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT1_14B_RES480P_T2V: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT1_1PT3B_RES480P_T2V['job']['name']}",
            {"override /net": "wan2pt1_14B_t2v_jvp"},
            {"override /net_teacher": "wan2pt1_14B_t2v"},
            {"override /net_fake_score": "wan2pt1_14B_t2v"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan",
            name="wan2pt1_14B_res480p_t2v_rCM",
        ),
        optimizer=dict(
            lr=1e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100,
                state_t=21,
                fsdp_shard_size=32,
                student_update_freq=10,
                optimizer_fake_score=dict(
                    lr=1e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                tangent_warmup=0,
                scm_timeshift=True,
                dmd_fix_timesteps=True,
                teacher_ckpt="assets/checkpoints/Wan2.1-T2V-14B.dcp",
                net=dict(
                    sac_config=dict(
                        mode="mm_only",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=250,
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=250,
                ),
                every_n_sample_ema=dict(
                    every_n=250,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

job_list = [WAN2PT1_1PT3B_RES480P_T2V, WAN2PT1_14B_RES480P_T2V]

for job in job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))
