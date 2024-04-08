#!/bin/bash
DEFAULT_NAME="/checkpoints/nemotron3-8b_v0.2.0"
NAME="${1:-$DEFAULT_NAME}"

DEFAULT_QUANT_CFG="fp8"
QUANT_CFG="${2:-$DEFAULT_QUANT_CFG}"

# CHANGE THE FOLLOWING IF YOU MOUNT YOUR DATA AND CHECKPOINTS DIFFERENTLY IN THE CONTAINER.
TP="8"
INFERENCE_TP=${TP}
DECODER_TYPE="gptnext"
CHECKPOINT_LOAD_DIR="${NAME}"
TOKENIZER_MODEL="${CHECKPOINT_LOAD_DIR}/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"

if [ "$QUANT_CFG" = "int4_awq" ]; then
    INFERENCE_TP="1"
fi

additional_options=" \
    --ammo-quant-cfg ${QUANT_CFG} \
    --ammo-load-classic-megatron-to-mcore \
    --decoder ${DECODER_TYPE} \
    --engine-dir /tmp/ammo \
    --max-input-len 2048 \
    --max-output-len 512 \
    --max-batch-size 8 \
    --inference-tensor-parallel ${INFERENCE_TP} "

trtllm_options=" \
    --engine-dir /tmp/ammo \
    --tokenizer ${TOKENIZER_MODEL} \
    --max-output-len 512 "

# DO NOT CHANGE THE SETTING BELOW UNLESS YOU KNOW WHAT YOU ARE DOING!!!
export CUDA_DEVICE_MAX_CONNECTIONS=1

options=" \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline_model_parallel_size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --save-interval 1000000 \
    --load ${CHECKPOINT_LOAD_DIR} \
    --bf16 \
    --use-mcore-models "

set +x

# Precompile CUDA extentions
python -c "import ammo.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Acquire launch configuration where variable launch_config will be set
launch_config="--nproc_per_node=${TP}"

# Launch multi-process with torchrun
torchrun ${launch_config} examples/inference/text_generation_ptq.py ${options} ${additional_options} --load ${CHECKPOINT_LOAD_DIR}

# This script is using mpi4py which will fork multiple processes.
python examples/inference/trtllm_text_generation.py ${trtllm_options}

