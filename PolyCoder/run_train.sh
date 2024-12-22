python run.py \
    --load_pretrained_model \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/polycoder-160m \
    --model_name_or_path=../pretrain-models/polycoder-160m \
    --do_train \
    --train_data_file=../data/reef/reef_train.csv \
    --eval_data_file=../data/reef/reef_val.csv \
    --test_data_file=../data/reef/reef_test.csv \
    --epochs 30 \
    --decoder_block_size 2048 \
    --output_size 256 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_batch_size 1 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log