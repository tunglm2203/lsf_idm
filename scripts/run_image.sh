#!/usr/bin/env bash

EXP_GPU_IDS=(0 0 0 0 0)   # TODO: USER update this

ROOT_LOG_DIR='./logs'
AGENT='sac_ae'

NUM_STEPS=1000000
DOMAIN_NAME='walker'
TASK_NAME='walk'
BATCHSIZE=128
EVAL_FREQ=10000
EXP_NAME='state'
INPUT_TYPE='pixel'


# TODO: For state
CUDA_VISIBLE_DEVICES=${EXP_GPU_IDS[0]} python train_logging_stepenv.py \
    --domain_name $DOMAIN_NAME --task_name $TASK_NAME \
    --encoder_type $INPUT_TYPE --decoder_type $INPUT_TYPE --work_dir $ROOT_LOG_DIR \
    --action_repeat 1 --num_eval_episodes 10 --agent $AGENT --frame_stack 1  --exp $EXP_NAME \
    --eval_freq $EVAL_FREQ --batch_size $BATCHSIZE --num_train_steps $NUM_STEPS\
    --save_tb --seed -1 --save_model

CUDA_VISIBLE_DEVICES=${EXP_GPU_IDS[0]} python train_logging_stepenv.py \
    --domain_name $DOMAIN_NAME --task_name $TASK_NAME \
    --encoder_type $INPUT_TYPE --decoder_type $INPUT_TYPE --work_dir $ROOT_LOG_DIR \
    --action_repeat 1 --num_eval_episodes 10 --agent $AGENT --frame_stack 1  --exp $EXP_NAME \
    --eval_freq $EVAL_FREQ --batch_size $BATCHSIZE --num_train_steps $NUM_STEPS\
    --save_tb --seed -1

CUDA_VISIBLE_DEVICES=${EXP_GPU_IDS[0]} python train_logging_stepenv.py \
    --domain_name $DOMAIN_NAME --task_name $TASK_NAME \
    --encoder_type $INPUT_TYPE --decoder_type $INPUT_TYPE --work_dir $ROOT_LOG_DIR \
    --action_repeat 1 --num_eval_episodes 10 --agent $AGENT --frame_stack 1  --exp $EXP_NAME \
    --eval_freq $EVAL_FREQ --batch_size $BATCHSIZE --num_train_steps $NUM_STEPS\
    --save_tb --seed -1

CUDA_VISIBLE_DEVICES=${EXP_GPU_IDS[0]} python train_logging_stepenv.py \
    --domain_name $DOMAIN_NAME --task_name $TASK_NAME \
    --encoder_type $INPUT_TYPE --decoder_type $INPUT_TYPE --work_dir $ROOT_LOG_DIR \
    --action_repeat 1 --num_eval_episodes 10 --agent $AGENT --frame_stack 1  --exp $EXP_NAME \
    --eval_freq $EVAL_FREQ --batch_size $BATCHSIZE --num_train_steps $NUM_STEPS\
    --save_tb --seed -1

CUDA_VISIBLE_DEVICES=${EXP_GPU_IDS[0]} python train_logging_stepenv.py \
    --domain_name $DOMAIN_NAME --task_name $TASK_NAME \
    --encoder_type $INPUT_TYPE --decoder_type $INPUT_TYPE --work_dir $ROOT_LOG_DIR \
    --action_repeat 1 --num_eval_episodes 10 --agent $AGENT --frame_stack 1  --exp $EXP_NAME \
    --eval_freq $EVAL_FREQ --batch_size $BATCHSIZE --num_train_steps $NUM_STEPS\
    --save_tb --seed -1
