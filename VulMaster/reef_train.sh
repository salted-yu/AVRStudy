#!/bin/bash
name=reef_fine_tuning_model
CUDA_VISIBLE_DEVICES=4
SLURM_NTASKS=1
python train_model.py \
        --train_data  data/reef/reef_train.json \
        --eval_data   data/reef/reef_val.json \
	      --model_size base \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 1 \
        --accumulation_steps 64 \
        --total_steps 160000 \
        --eval_freq 8000 \
        --save_freq 8000 \
        --n_context 10 \
	      --beam_size 1 \
	      --use_adapted_model \
	      --adapted_model_path ./bugfix_pretrain_with_ast/pytorch_model.bin \
	      --text_maxlength 512 \
	      --answer_maxlength 512 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint_final