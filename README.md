# WALA Core Source

> **Project Page:** **[liujiahao2077.github.io/WALA](https://liujiahao2077.github.io/WALA/)**

This repository contains the compact source release for WALA, including the core model architecture built around Qwen3-VL, DINOv3, and Depth Anything V2.

## Layout

- `src/WALA/model/framework/VLM4A/WALA.py`: main WALA framework.
- `src/WALA/model/modules/world_model/LAM.py`: latent action model.
- `src/WALA/model/modules/action_model/ActionHead.py`: action prediction heads.
- `src/WALA/model/modules/vlm/QWen3.py`: Qwen3-VL interface.
- `src/WALA/model/modules/dino_model/dinov3.py`: DINOv3 feature wrapper.
- `configs/wala.yaml`: minimal configuration example.

## Weights

Hugging Face: https://huggingface.co/liujiahao2077/WALA
