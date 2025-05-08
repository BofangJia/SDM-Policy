# bash scripts/gen_demonstration_metaworld.sh handle-pull
# bash scripts/gen_demonstration_metaworld.sh handle-pull-side


cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../SDM/data/" \
