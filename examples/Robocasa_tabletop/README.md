# RoboCasa-GR1 Tabletop Example

This folder contains WALA training and evaluation files for RoboCasa-GR1
tabletop tasks. The training directory provides the example configuration and
launch script for RoboCasa policy training.

## Training

```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wala.sh
```

## Environment Setup

Follow the official [RoboCasa-GR1 tabletop task setup](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
to create the `robocasa` simulator environment and prepare the required assets.
WALA uses a two-environment workflow: the policy server runs in the `wala`
environment, while the simulator runs in the `robocasa` environment.

Install the evaluation-side bridge dependencies in the `robocasa` environment:

```bash
conda activate robocasa
pip install tyro websockets msgpack rich omegaconf av imageio imageio-ffmpeg
```

## Evaluation

Run the RoboCasa batch evaluator:

```bash
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh
```

For the PPU runtime launcher:

```bash
bash examples/Robocasa_tabletop/eval_files/run_robocasa_eval_ppu.sh
```

Modify the checkpoint path, Python paths, GPU count, episode count, and rollout
settings directly in the corresponding shell script before running. Both
scripts start policy servers, dispatch all RoboCasa-GR1 tabletop tasks, and save
logs and rollout videos under the corresponding `results/Checkpoints/<run_id>/`
directory.
