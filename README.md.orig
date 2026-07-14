# WALA

Project page: [WALA Project Page](https://liujiahao2077.github.io/WALA.github.io/)

Paper: [WALA: Learning Executable Latent Actions from Action-Labeled Robot Demonstrations and Action-Free Videos](https://arxiv.org/abs/2607.11397)

**WALA** stands for **World- and Action-supervised Latent Actions**. This repository contains the project code for **WALA: Learning Executable Latent Actions from Action-Labeled Robot Demonstrations and Action-Free Videos**.

This repository includes:

- semantic-geometric latent action model pretraining with DINOv3 features and dense depth supervision;
- policy training with robot action prediction, latent action target matching, and future dynamics supervision;
- RoboCasa-GR1 tabletop, RoboTwin, and pretraining examples;
- policy-server utilities for simulation evaluation.

## Repository Layout

```text
wala/                         Core Python package
  model/framework/policies/   WALA policy framework
  model/modules/latent_action_model/  Latent action model modules
  model/modules/vision_encoder/       DINOv3 wrapper
  training/                   Training entrypoints
deployment/model_server/      WebSocket policy server used by simulation evaluation
examples/
  Pretrain/                   Latent action model pretraining
  Robocasa_tabletop/          RoboCasa-GR1 tabletop training and evaluation
  Robotwin/                   RoboTwin training and evaluation
checkpoints/                  Local directory for third-party pretrained weights
results/                      Training checkpoints, logs, and rollout videos
```

Datasets, simulator assets, third-party pretrained weights, and training outputs are not included.

## Environment Setup

WALA uses one Python environment for model training and policy serving. RoboCasa and RoboTwin evaluation should be installed in separate simulator environments.

### 1. Create the WALA environment

Install PyTorch first with the CUDA build that matches your machine. For CUDA 12.1, for example:

```bash
conda create -n wala python=3.10 -y
conda activate wala

pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install WALA and the core Python dependencies:

```bash
git clone <this-repository-url> WALA
cd WALA

pip install -r requirements.txt
pip install -e .
```

### 2. Install Depth Anything V2 dependencies

WALA uses the bundled Depth Anything V2 implementation for dense depth
supervision. Install its Python dependencies in the same `wala` environment:

```bash
conda activate wala
pip install -r wala/model/third_party/Depth-Anything-V2/requirements.txt
```

For the original Depth Anything V2 usage notes and checkpoint links, see
`wala/model/third_party/Depth-Anything-V2/README.md`.

### 3. Prepare third-party pretrained weights

Download the required third-party checkpoints according to their original
licenses and place them under `checkpoints/`. The example configurations use
the following local layout:

```text
checkpoints/
  qwen/Qwen3-VL-4B-Instruct/
  dinov3/dinov3-vitb16-pretrain-lvd1689m/
  depth_anything/depth_anything_v2_vitl.pth
```

If your checkpoints are stored elsewhere, update the paths in the corresponding
example YAML files before training or evaluation.

### 4. Prepare datasets

WALA uses LeRobot-format datasets in the released training scripts. Prepare the
datasets required by each benchmark and update the dataset paths in the example
YAML files:

```text
examples/Pretrain/train_files/wala_pretrain.yaml
examples/Robocasa_tabletop/train_files/wala_robocasa.yaml
examples/Robotwin/train_files/wala_robotwin.yaml
```

The pretraining examples may mix action-free videos and action-labeled robot
demonstrations. The policy-training examples expect the corresponding RoboCasa
or RoboTwin datasets.

### 5. RoboCasa-GR1 tabletop simulator environment

RoboCasa evaluation uses two processes:

- the WALA policy server, launched in the `wala` environment;
- the RoboCasa simulator, launched in a separate `robocasa` environment.

First install the simulator following the official
[RoboCasa-GR1 tabletop task setup](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).
WALA serves the policy in the `wala` environment, while the RoboCasa simulator runs in its own
`robocasa` environment.

After the simulator is installed, install the small set of evaluation-side
Python packages used by the WALA bridge:

```bash
conda activate robocasa
pip install tyro websockets msgpack rich omegaconf av imageio imageio-ffmpeg
```

See `examples/Robocasa_tabletop/README.md` for the full details.

### 6. RoboTwin simulator environment

RoboTwin evaluation also uses two processes:

- the WALA policy server, launched in the `wala` environment;
- the RoboTwin evaluator, launched in a separate `robotwin` environment.

Install RoboTwin following the official
[RoboTwin 2.0 installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
The policy server runs in
the `wala` environment, and RoboTwin evaluation runs in the `robotwin`
environment. After installing RoboTwin, point WALA to that checkout:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
export ROBOTWIN_PATH=/absolute/path/to/third_party/RoboTwin
```

Create or activate the RoboTwin environment required by the official
installation, then install WALA's eval-side bridge dependencies:

```bash
conda activate robotwin
pip install -r examples/Robotwin/eval_files/requirements.txt
```

See `examples/Robotwin/README.md` for the full details.

## Training

Run commands from the repository root. The scripts write checkpoints and logs under `results/Checkpoints/`.

Pretrain the latent action model:

```bash
conda activate wala
bash examples/Pretrain/train_files/run_pretrain_wala.sh
```

Train on RoboCasa-GR1 tabletop:

```bash
conda activate wala
bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wala.sh
```

Train on RoboTwin:

```bash
conda activate wala
bash examples/Robotwin/train_files/run_robotwin_train_wala.sh
```

Before launching distributed training, check GPU count, batch size, NCCL interface, and `accelerate` settings in each script. The default scripts use the DeepSpeed Zero-2 accelerate config under `wala/config/deepseeds/`.

## Evaluation

### RoboCasa-GR1 tabletop

Edit checkpoint paths and Python paths directly in the shell script, then run:

```bash
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh
```

For the provided PPU runtime launcher:

```bash
bash examples/Robocasa_tabletop/eval_files/run_robocasa_eval_ppu.sh
```

Both scripts start WALA policy servers, dispatch the RoboCasa tasks, and save logs/videos under the corresponding `results/Checkpoints/<run_id>/` directory.

### RoboTwin

Set `ROBOTWIN_PATH`, then launch:

```bash
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin

bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean \
    -n full_eval \
    -s 42 \
    -j 2 \
    -c results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt \
    -p 6666 \
    all
```

Use `demo_clean` for the Easy setting and `demo_randomized` for the Hard setting. See `examples/Robotwin/README.md` for task lists, logging paths, and optional flags.

## Notes

- Training outputs are written under `results/`.
- Third-party pretrained model files can be placed under `checkpoints/`.
- RoboCasa and RoboTwin are separate projects; install and update them independently from WALA.

## Acknowledgement

We sincerely thank the great open-source [StarVLA](https://github.com/starVLA/starVLA) project. If you encounter environment installation issues, you may also refer to the StarVLA environment setup and benchmark-specific installation instructions. The environments used in this project are built on top of the StarVLA setup. We thank StarVLA again for its valuable open-source contribution.
