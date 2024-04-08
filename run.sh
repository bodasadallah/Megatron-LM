#!/bin/bash
pip install -r requirements.txt
pip install deepspeed wandb

WANDBKEY='628fc7dc1050e9a41e10a9dc7ad0390219369cae'
wandb login $WANDBKEY

WANDB_API_KEY=$WANDBKEY
pip install flash-attn
