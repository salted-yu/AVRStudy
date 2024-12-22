python run.py \
    --model_name=pretrained_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/gpt2-csrc \
    --model_name_or_path=../pretrain-models/gpt2-csrc \
    --do_train \
    --train_data_file=../data/vrepair_non_domain_data/processed_non_domain_train.csv \
    --eval_data_file=../data/vrepair_non_domain_data/processed_non_domain_val.csv \
    --epochs 30 \
    --decoder_block_size 1024 \
    --output_size 128 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 1 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee pretrain.log
