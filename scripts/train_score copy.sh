#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python runners/trainer.py \
--data_path Omni6DPose_SOPE_PATH \
--pretrained_score_model_path None \
--log_dir ScoreNet \
--agent_type score \
--use_pretrain \
--sampler_mode ode \
--sampling_steps 500 \
--eval_freq 1 \
--batch_size 64 \
--n_epochs 100 \
--percentage_data_for_train 1.0 \
--percentage_data_for_test 1.0 \
--percentage_data_for_val 1.0 \
--seed 0 \
--is_train \
--dino pointwise \
--num_workers 12 

--pretrained_score_model_path /home/dai/ai/GenPose2/results/ckpts/ScoreNet/ckpt_epoch30.pth \
