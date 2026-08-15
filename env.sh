conda activate acg

pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

export HF_ENDPOINT=https://hf-mirror.com

export WANDB_API_KEY=""


# === Quick path: no preprocessing (84x84 decoding) ===
n_mg="1000"
export DEXMG_VIDEO_RESOLUTION="84x84"   # fast decoding, lower resolution
note="_${DEXMG_VIDEO_RESOLUTION}"

# === Experiment setup ===
steps="6000"
ngpu="1"
bs="2"
ga="64"
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"
model_path="/mnt/2t/myh/experiment/checkpoints/nvidia/GR00T-N1-2B"

# === Logging (Weights & Biases) ===
export WANDB_ENTITY="Entity"
export WANDB_PROJECT="Robot Project"

# === Launch training ===
python libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py \
  --num-gpus ${ngpu} \
  --output-dir checkpoints/dexmg/${exp_name} \
  --data-configs dexmg_bimanual_panda_gripper dexmg_bimanual_panda_hand dexmg_gr1_arms_only dexmg_gr1_arms_only \
  --video-backend decord \
  --embodiment_tag single_panda_gripper \
  --exp_name ${exp_name} \
  --batch_size ${bs} \
  --robomimic_config_json libs/Isaac-GR00T-N1/robomimic_configs/dexmg_mg${n_mg}_6.json \
  --gradient_accumulation_steps ${ga} \
  --no-save-only-model \
  --dataloader_num_workers 16 \
  --max-steps ${steps} \
  --save_steps 1000 \
  --save_total_limit 3 \
  --dataset_cls=dexmg \
  --pin_memory \
  --training_seed ${training_seed} \
  --base_model_path ${model_path} \
  --lora-rank 16 \
  --lora-alpha 32




# === DexMG rollout: without guidance ===
note=""
n_rollouts="24"
num_batch_envs="1"
export MAX_NUM_EMBODIMENTS="35"
dataset_name="dexmg_mg1000"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}_6.json"
# model_path="/mnt/2t/myh/experiment/checkpoints/DAVIAN-Robotics/GR00T-N1-2B-tuned-DexMG-MG100-CrossEmbodiments"
model_path="/mnt/2t/myh/experiment/checkpoints/nvidia/GR00T-N1-2B"
lora_path="/home/ubuntu/myh/experiment/ACG/checkpoints/dexmg/MG1000/LR=1e-4_Bs=1x2x64_Steps=6000_Seed=42_84x84/checkpoint-6000/"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" ${lora_path} algo_name=gr00t_guidance_dexmg


# === DexMG rollout: with ACG ===
note=""
n_rollouts="24"
num_batch_envs="1"
export MAX_NUM_EMBODIMENTS="35"
dataset_name="dexmg_mg1000"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}_6.json"
model_path="/mnt/2t/myh/experiment/checkpoints/nvidia/GR00T-N1-2B"
lora_path="/home/ubuntu/myh/experiment/ACG/checkpoints/dexmg/MG1000/LR=1e-4_Bs=1x2x64_Steps=6000_Seed=42_84x84/checkpoint-6000/adapter_model.safetensors"

acg_options="
algo.guidance.name=acg
algo.guidance.scale=1.03
algo.guidance.skip_blocks=7,9,11
"

note="${note}_acg"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" algo_name=gr00t_guidance_dexmg ${acg_options}




# === Quick path: no preprocessing (84x84 decoding) ===
n_mg="1000"
export DEXMG_VIDEO_RESOLUTION="84x84"   # fast decoding, lower resolution
note="_${DEXMG_VIDEO_RESOLUTION}"

# === Experiment setup ===
steps="60000"
ngpu="3"
bs="64"
ga="2"
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"
model_path="/home/mayuhang/models/GR00T-N1-2B"

# === Logging (Weights & Biases) ===
export WANDB_ENTITY="skywalkerm-no"
export WANDB_PROJECT="ACG"

# === Launch training ===
CUDA_VISIBLE_DEVICES=1,2,3 python libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py \
  --num-gpus ${ngpu} \
  --output-dir checkpoints/dexmg/${exp_name} \
  --data-configs dexmg_bimanual_panda_gripper dexmg_bimanual_panda_hand dexmg_gr1_arms_only dexmg_gr1_arms_only \
  --video-backend decord \
  --embodiment_tag single_panda_gripper \
  --exp_name ${exp_name} \
  --batch_size ${bs} \
  --robomimic_config_json libs/Isaac-GR00T-N1/robomimic_configs/dexmg_mg${n_mg}_6.json \
  --gradient_accumulation_steps ${ga} \
  --dataloader_num_workers 2 \
  --max-steps ${steps} \
  --save_steps 1000 \
  --save_total_limit 1 \
  --dataset_cls=dexmg \
  --pin_memory \
  --training_seed ${training_seed} \
  --base_model_path ${model_path} \
  --no-tune-visual



# === Experiment setup (LoRA) ===
  --lora_rank 16 \
  --lora_alpha 32 \
  bs="8"    # 从 64 降低到 8 (如果还是 OOM，可以进一步降到 4)
  ga="16"   # 相应增加梯度累加，维持等效 Batch Size：3 * 8 * 16 = 384



# === DexMG rollout: without guidance ===
note=""
n_rollouts="8"
num_batch_envs="1"
export MAX_NUM_EMBODIMENTS="35"
dataset_name="dexmg_mg1000"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}_6.json"
model_path="/home/mayuhang/ACG/checkpoints/dexmg/MG1000/checkpoint-60000/"
seed="42"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" algo_name=gr00t_guidance_dexmg




# === 10_nvidia.json ===
# 1. 确保目录存在
mkdir -p /usr/share/glvnd/egl_vendor.d/

# 2. 写入配置文件（根据之前查到的实际版本号）
cat <<EOF > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.595.71.05"
    }
}
EOF




export DATASET_ROOT="/home/mayuhang/datasets"  # fill in your dataset root path

# Replay and save videos for each task and camera view
task_name="two_arm_box_cleanup"             ; render_image_name="agentview"             ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_box_cleanup"             ; render_image_name="robot0_eye_in_hand"    ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_box_cleanup"             ; render_image_name="robot1_eye_in_hand"    ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_drawer_cleanup"          ; render_image_name="agentview"             ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_drawer_cleanup"          ; render_image_name="robot0_eye_in_hand"    ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_drawer_cleanup"          ; render_image_name="robot1_eye_in_hand"    ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_lift_tray"               ; render_image_name="agentview"             ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_lift_tray"               ; render_image_name="robot0_eye_in_hand"    ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_lift_tray"               ; render_image_name="robot1_eye_in_hand"    ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_can_sort_random"         ; render_image_name="agentview"             ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_coffee"                  ; render_image_name="agentview"             ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_pouring"                 ; render_image_name="agentview"             ; python libs/dexmimicgen/scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos_256x256/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}

export DATASET_ROOT="/home/mayuhang/datasets"  # fill in your dataset root path

# Each index (task index) corresponds to one parallel conversion job.
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py


# === Recommended path: preprocessed videos (256x256) ===
n_mg="1000"
export DEXMG_VIDEO_RESOLUTION="256x256" # preprocessed, high-quality frames
note="_${DEXMG_VIDEO_RESOLUTION}"

# === Logging (Weights & Biases) ===
steps="60000"
ngpu="4"
bs="32"
ga="1"
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"
model_path="/home/mayuhang/models/GR00T-N1-2B"

# === Launch training ===
export WANDB_ENTITY="skywalkerm-no"
export WANDB_PROJECT="ACG"

CUDA_VISIBLE_DEVICES=0,1,2,3 python libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py \
  --num-gpus ${ngpu} \
  --output-dir checkpoints/dexmg/${exp_name} \
  --data-configs dexmg_bimanual_panda_gripper dexmg_bimanual_panda_hand dexmg_gr1_arms_only dexmg_gr1_arms_only \
  --video-backend decord \
  --embodiment_tag single_panda_gripper \
  --exp_name ${exp_name} \
  --batch_size ${bs} \
  --robomimic_config_json libs/Isaac-GR00T-N1/robomimic_configs/dexmg_mg${n_mg}_6.json \
  --gradient_accumulation_steps ${ga} \
  --save-only-model \
  --dataloader_num_workers 2 \
  --max-steps ${steps} \
  --save_steps 1000 \
  --save_total_limit 2 \
  --dataset_cls=dexmg \
  --pin_memory \
  --training_seed ${training_seed} \
  --base_model_path ${model_path} \
  --no-tune-visual
