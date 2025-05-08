source activate
conda activate sdm
# cd 
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

tar -xvzf mujoco210.tar.gz

#!/bin/bash
export WANDB_API_KEY=''

# wandb
python -c "import wandb; wandb.login()"


NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$NUM_GPUS" -lt 1 ]; then
  echo "No GPUs detected!"
  exit 1
fi

echo "Detected $NUM_GPUS GPUs"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dingpengxiang/mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl

# input your path
# bash scripts/train_policy_sdm.sh dp3_sdm metaworld_basketball 1108_sdm 0 0