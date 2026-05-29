<div align="center">

# TurboT2AV

Fast text-to-audio-video generation distilled from LTX-2 19B.

</div>

## Overview

TurboT2AV generates synchronized audio-video from text prompts in 4 steps.
The demo compares the 40-step teacher with the 4-step student.
This repository provides single-GPU inference for the distilled checkpoint.
On an NVIDIA H20, single-sample inference takes about 50 seconds for the
40-step teacher and about 2.5 seconds for the 4-step student.
Training code and the full data-processing pipeline are coming soon.

Main contributions:

- Combines the diversity of consistency models (DCM/SCM) with the high
  perceptual quality of score-model distillation (DMD), taking advantage of both
  families of methods by using CM as a forward-divergence offline method that
  complements DMD as a reverse-KL on-policy method.
- First extends this combined distillation strategy to a large-scale joint
  audio-video generation model at the 14B-video + 5B-audio scale.

<table>
  <thead>
    <tr>
      <th align="center" width="50%">Teacher (40 steps)</th>
      <th align="center" width="50%">Student (4 steps)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/116d8f07-96a3-4a68-b0bd-fb03eb385269" alt="teacher_p5" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/2b0509e0-216c-4575-ad03-d183a673a9ec" alt="student_p5" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/e5330419-2147-48a9-9225-03113c6d1488" alt="teacher_p6" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/9bbbf2ad-b16a-42ea-b63d-2faf1ef0abb2" alt="student_p6" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/3a7e87b7-5c72-496c-9e6c-205e1ad31bb5" alt="teacher_p73" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/eb21d65c-3bf9-4999-b73d-6c1f825f1549" alt="student_p73" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/f04e2e34-b503-4557-8700-18ce5a058ff9" alt="teacher_p79" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/24ba0beb-f362-4e13-8b03-cac9f63a5410" alt="student_p79" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/820cd365-1737-4126-89aa-af9b120a0c22" alt="teacher_p92" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/de460bd2-c41a-4888-ad0c-735d0358787b" alt="student_p92" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/32bba4ef-ebd2-4a58-a574-352f22ba64b9" alt="teacher_p99" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/1cb3b685-b4cf-4455-b421-623a7ccb39c0" alt="student_p99" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/59a77c01-e237-4cdc-8099-c8cd2e364b70" alt="teacher_p165" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/f692a251-0f63-48eb-a51e-cdb0deaedcd7" alt="student_p165" width="100%" controls></video></td>
    </tr>
  </tbody>
</table>

## 1. Setup

```bash
cd TurboDiffusion/TurboT2AV/LTX-2
pixi install
pixi run install-pytorch
pixi run pip install -e packages/ltx-core
pixi run pip install -e packages/ltx-pipelines
pixi run pip install -e packages/ltx-distillation
```

## 2. Download Weights

| Model Name | Checkpoint Link |
| --- | --- |
| TurboT2AV-14BVideo-5BAudio | [Hugging Face Model](https://huggingface.co/luyu1021/turbo-t2av) |
| LTX-2-19B | [Hugging Face Model](https://huggingface.co/Lightricks/LTX-2) |
| Gemma-3-12B-IT-QAT-Q4_0 | [Hugging Face Model](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

Gemma is a gated Hugging Face model. Before downloading, visit the model page,
accept the access terms, and export a Hugging Face token with access permission:

```bash
export HF_TOKEN=your_huggingface_token
```

Base model weights:

```bash
hf download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir /path/to/checkpoints/LTX-2
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir /path/to/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

Distilled checkpoint:

```bash
hf download luyu1021/turbo-t2av checkpoints/scm_dmd_checkpoint_001000/model.pth --local-dir /path/to/checkpoints
```

## 3. Run Inference

Set environment variables:

```bash
export TURBO_CHECKPOINT_PATH=/path/to/ltx-2-19b-dev.safetensors
export TURBO_GEMMA_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

### Teacher (40 steps)

```bash
cd LTX-2
PYTHONPATH=packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/prompts.csv \
  --output_dir /path/to/teacher_output \
  --model_kind teacher \
  --teacher_mode native_rf \
  --teacher_steps 40 \
  --num_prompts 8
```

### Student (4 steps)

```bash
cd LTX-2
PYTHONPATH=packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/prompts.csv \
  --output_dir /path/to/student_output \
  --model_kind student \
  --student_checkpoint /path/to/checkpoint.pt \
  --student_param auto \
  --num_prompts 8
```

`--prompts_file` supports CSV (`video_id,prompt`) or plain text (one prompt per line).

Outputs are saved under the requested `--output_dir` with separate subfolders:
`video/` for MP4 files, `audio/` for WAV files, and `json/` for prompt metadata.
