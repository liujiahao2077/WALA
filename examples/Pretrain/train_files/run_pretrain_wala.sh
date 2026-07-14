export NCCL_SOCKET_IFNAME=net0
export PYTHONMALLOC=malloc
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

Framework_name=WALA
config_yaml=./examples/Pretrain/train_files/wala_pretrain.yaml
run_root_dir=./results/Checkpoints
run_id=wala_pretrain_$(date +%Y_%m_%d)

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file wala/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 2 \
  --num_processes 32 \
  --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port 29500 \
  wala/training/train_wala.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wala_Pretrain
