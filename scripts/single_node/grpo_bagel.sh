# 1 GPU
module purge
module load miniconda/3
module load cuda/12.6.0
conda activate ~/scratch/.conda/flow_grpo

export PYTHONPATH="${PYTHONPATH}:$(pwd)/bagel"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/flow_grpo"
export SAVE_DIR="/home/mila/r/rabiul.awal/scratch/"

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1  
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_bagel.py --config config/bagel.py:bagel_1gpu


# 4 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_bagel.py --config config/bagel.py:bagel_4gpu

# 8 GPU
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_bagel.py --config config/bagel.py:bagel_8gpu
