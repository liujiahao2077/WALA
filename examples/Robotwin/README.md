# RoboTwin Example

This folder contains WALA training and evaluation files for RoboTwin 2.0.
The training directory provides the example configuration and launch script for
RoboTwin policy training.

## Training

```bash
bash examples/Robotwin/train_files/run_robotwin_train_wala.sh
```

## Environment Setup

Follow the official [RoboTwin 2.0 installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
to create the base `robotwin` environment and download the simulator assets.
WALA uses a two-environment workflow: the policy server runs in the `wala`
environment, and the RoboTwin evaluator runs in the `robotwin` environment.

After RoboTwin is installed, prepare the evaluation-side dependencies:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
pip install -r examples/Robotwin/eval_files/requirements.txt
```

Following the [StarVLA](https://github.com/starVLA/starVLA) RoboTwin setup,
patch your local RoboTwin checkout so `script/eval_policy.py` accepts
`--policy_ckpt_path`. WALA passes this checkpoint path at evaluation time so
the RoboTwin-side policy interface can load the corresponding WALA
configuration and normalization statistics.

Apply the following change inside your RoboTwin repository:

```diff
diff --git a/script/eval_policy.py b/script/eval_policy.py
--- a/script/eval_policy.py
+++ b/script/eval_policy.py
@@ -69,6 +69,7 @@ def main(usr_args):
     # checkpoint_num = usr_args['checkpoint_num']
     policy_name = usr_args["policy_name"]
     instruction_type = usr_args["instruction_type"]
+    policy_ckpt_path = usr_args["policy_ckpt_path"]
     save_dir = None
     video_save_dir = None
     video_size = None
@@ -81,6 +82,7 @@ def main(usr_args):
     args['task_name'] = task_name
     args["task_config"] = task_config
     args["ckpt_setting"] = ckpt_setting
+    args["policy_ckpt_path"] = policy_ckpt_path
 
     embodiment_type = args.get("embodiment")
     embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
@@ -327,11 +329,13 @@ def eval_policy(task_name,
 def parse_args_and_config():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, required=True)
     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
     args = parser.parse_args()
 
     with open(args.config, "r", encoding="utf-8") as f:
         config = yaml.safe_load(f)
+    config["policy_ckpt_path"] = args.policy_ckpt_path
 
     # Parse overrides
     def parse_override_pairs(pairs):
```

## Evaluation

Use `start_eval.sh` as the main RoboTwin evaluation launcher. It starts the
WALA policy server, waits for the server to become ready, launches RoboTwin
evaluation jobs, writes logs, and cleans up child processes on exit.

Example for evaluating all 50 tasks in the Easy setting:

```bash
bash examples/Robotwin/eval_files/start_eval.sh -m demo_clean -n full_eval -s 42 -j 2 -c results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt -p 6666 all
```

Example for evaluating one task:

```bash
bash examples/Robotwin/eval_files/start_eval.sh -m demo_clean -n single_task -s 42 -j 1 -c results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt -p 6666 adjust_bottle
```

Use `demo_clean` for the Easy setting and `demo_randomized` for the Hard
setting. `start_eval.sh` schedules jobs over the GPUs visible to the current
process. Set `CUDA_VISIBLE_DEVICES` before launching if you want to restrict
evaluation to a subset of GPUs; for example, the command below exposes only
GPUs 0, 1, 2, and 3 to the launcher:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Useful options:

```text
-s, --seed              Evaluation seed
-j, --jobs-per-gpu      Concurrent tasks per visible GPU
-p, --base-port         First policy-server port
--server-timeout        Policy-server startup timeout in seconds
```

Logs are written under the checkpoint directory by default:

```text
<ckpt_dir>/robotwin_eval_logs/<name>_<mode>_<ckpt_stem>_<timestamp>/
```

The low-level scripts `run_policy_server.sh` and `eval.sh` are kept for manual
inspection, but normal evaluation should go through `start_eval.sh`.
