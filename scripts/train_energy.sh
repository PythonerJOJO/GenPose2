#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python runners/trainer.py \
--data_path /root/autodl-tmp/xyzibd \
--log_dir EnergyNet \
--agent_type energy \
--sampler_mode ode \
--sampling_steps 500 \
--eval_freq 1 \
--batch_size 64 \
--n_epochs 500 \
--percentage_data_for_train 1.0 \
--percentage_data_for_test 1.0 \
--percentage_data_for_val 1.0 \
--seed 0 \
--is_train \
--dino pointwise \
--num_worker 12 \
