note=""
n_rollouts="24"
num_batch_envs="1"
export MAX_NUM_EMBODIMENTS="35"
dataset_name="dexmg_mg30"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="/home/ubuntu/myh/expirement/ACG/DAVIAN-Robotics/GR00T-N1-2B-tuned-DexMG-MG100-CrossEmbodiments"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" algo_name=gr00t_guidance_dexmg


acg_options="
algo.guidance.name=acg
algo.guidance.scale=1.03
algo.guidance.skip_blocks=7,9,11
"

note="${note}_acg"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" algo_name=gr00t_guidance_dexmg ${acg_options}

