#!/bin/bash



pip install  wandb flash-attn

WANDBKEY='628fc7dc1050e9a41e10a9dc7ad0390219369cae'
wandb login $WANDBKEY

WANDB_API_KEY=$WANDBKEY



WANDB_PROJECT='ML710-prject-megatron'
WANDB_EXP_NAME='bert-pretrain-2ddp-2pp'

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

WANDB_SAVE_DIR='logs'

DATA_PATH="bert/pretraining_data/bdata_text_sentence"
# VALID_DATA="bert/RACE/dev/middle \
#             bert/RACE/dev/high"
VOCAB_FILE=bert/bert-large-uncased-vocab.txt
PRETRAINED_CHECKPOINT=bert/bert_345m_uncased
CHECKPOINT_PATH=checkpoints/pretrain_mp


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


    # --DDP-impl local
    # --sequence-parallel \
DISTRIBUTED_TRAINING_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline_model_parallel_size 2 \
"
    # --pipeline_model_parallel_split_rank 1


# --rampup-batch-size
    # --train-iters 1000000 \
TRAINING_ARGS="
    --seed 1234 \
    --global-batch-size 32 \
    --micro-batch-size 16 \
    --train-samples 5000000
    --lr-decay-samples 4000000 \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --min-lr 1.0e-5 \
    --lr 0.0001 \
    --min-lr 1.0e-5 \
    --lr-warmup-fraction .01 
    --seq-length 512 \
    --max-position-embeddings 512 \
    --recompute-activations \
    --recompute-granularity selective \
    --tensorboard-dir logs/tensor_board \
    --use-flash-attn \
    --clip-grad 1.0 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --fp16 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --auto-detect-ckpt-format
"
MODEL_ARGS="
    --num-layers 48 \
    --hidden-size 2048 \
    --num-attention-heads 64
"
    # --num-layers 24 \
    # --hidden-size 1024 \
    # --num-attention-heads 16
DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file $VOCAB_FILE \
    --save $CHECKPOINT_PATH \
    --split 949,50,1
"
    # --load $CHECKPOINT_PATH \

######## To enable async all reduce #########
export CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
        $MODEL_ARGS \
        $LOGGING_ARGS \
        $TRAINING_ARGS \
        $DATA_ARGS \
        $DISTRIBUTED_TRAINING_ARGS 
               
               


