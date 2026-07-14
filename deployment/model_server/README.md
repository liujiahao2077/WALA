# WALA Policy Server

Start a websocket policy server from a trained WALA checkpoint:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path results/Checkpoints/<run_id>/checkpoints/<checkpoint>.pt \
  --port 10093 \
  --use_bf16
```

The RoboCasa evaluation scripts start policy servers automatically, so this
entry point is mainly useful for custom evaluation or deployment clients.
