#!/bin/bash



pip install  wandb

WANDBKEY='628fc7dc1050e9a41e10a9dc7ad0390219369cae'
wandb login $WANDBKEY

WANDB_API_KEY=$WANDBKEY

WORLD_SIZE=1


GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

WANDB_PROJECT='ML710-prject-megatron'
WANDB_EXP_NAME='bert-3650-race-ft'
WANDB_SAVE_DIR='logs'

TRAIN_DATA="bert/RACE/train/middle"
VALID_DATA="bert/RACE/dev/middle \
            bert/RACE/dev/high"
VOCAB_FILE=bert/bert-large-uncased-vocab.txt
PRETRAINED_CHECKPOINT=bert/bert_345m_uncased
CHECKPOINT_PATH=checkpoints/bert_345m_race


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
LOGGING_ARGS="
    --log-throughput \
    --log-progress \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --log-world-size-to-tensorboard \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR
"


DISTRIBUTED_TRAINING_ARGS="
    --sequence-parallel \
    --tensor-model-parallel-size 1 \
    --pipeline_model_parallel_size 4 \
    --pipeline_model_parallel_split_rank 2 
"
# --rampup-batch-size
TRAINING_ARGS="
    --seed 1234 \
    --micro-batch-size 4 \
    --lr 1.0e-5 \
    --lr-decay-style linear \
    --lr-warmup-fraction 0.06 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --recompute-activations \
    --recompute-granularity selective \
    --log-interval 100 \
    --train-samples 10000 \
    --tensorboard-dir logs/tensor_board \
    --use-flash-attn \
    --weight-decay 1.0e-1 \
    --clip-grad 1.0 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --fp16 \
    --epochs 3 \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 50 \
    --finetune \
    --auto-detect-ckpt-format
"
MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16
"
DATA_ARGS="
    --valid-data $VALID_DATA \
    --train-data $TRAIN_DATA \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file $VOCAB_FILE \
    --task RACE \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --save $CHECKPOINT_PATH
"

######## To enable async all reduce #########
export CUDA_DEVICE_MAX_CONNECTIONS=1
# python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
torchrun $DISTRIBUTED_ARGS ./tasks/main.py \
        $MODEL_ARGS \
        $LOGGING_ARGS \
        $TRAINING_ARGS \
        $DATA_ARGS \
        $DISTRIBUTED_TRAINING_ARGS 
               
               


