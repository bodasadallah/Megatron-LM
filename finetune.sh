#!/bin/bash

#SBATCH --job-name=mixtral_ora # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=4                   # Run all processes on a single node    
#SBATCH --ntasks=4                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8 
#SBATCH -p gpu                      # Use the gpu partition
##SBATCH --nodelist=ws-l6-005

# conda activate llama_factory

# export TRANSFORMERS_CACHE=/l/users/abdelrahman.sadallah/hugging_face


pip install -r requirements.txt
pip install deepspeed wandb

WANDBKEY='628fc7dc1050e9a41e10a9dc7ad0390219369cae'
wandb login $WANDBKEY

WANDB_API_KEY=$WANDBKEY
# /home/abdelrahman/MBZUAI/LLaMA-Factory/examples/deepspeed/ds_z2_config.json
###################### RUN LLM Finetune ######################
# deepspeed --num_gpus 4 src/train_bash.py \
# --deepspeed examples/deepspeed/ds_z2_config.json \
# --ddp_timeout 180000000 \
# --stage sft \
# --do_train \
# --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
# --dataset alpaca_gpt4_en \
# --template default \
# --finetuning_type lora \
# --lora_target q_proj,v_proj \
# --output_dir mixtral \
# --overwrite_cache \
# --overwrite_output_dir \
# --per_device_train_batch_size 2 \
# --gradient_accumulation_steps 4 \
# --lr_scheduler_type cosine \
# --logging_steps 10 \
# --save_steps 100 \
# --learning_rate 5e-5 \
# --num_train_epochs 1.0 \
# --plot_loss \
# --fp16

CHECKPOINT_PATH="output" 
TENSORBOARD_LOGS_PATH="logs"
VOCAB_FILE="" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="" #<Specify path and file prefix>_text_document
bash examples/gpt3/train_gpt3_175b_distributed.sh \
$CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH "


