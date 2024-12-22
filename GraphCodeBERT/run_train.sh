python run.py \
    --load_pretrained_model \
    --model_name fine_tuned_model.bin \
    --do_train \
    --model_type roberta \
    --source_lang c_sharp \
    --model_name_or_path ../pretrain-models/graphcodebert-base \
    --tokenizer_name ../pretrain-models/graphcodebert-base \
    --config_name ../pretrain-models/graphcodebert-base/config.json \
    --train_filename ../data/reef/reef_train.csv \
    --dev_filename ../data/reef/reef_val.csv \
    --output_dir ./saved_models \
    --max_source_length 512 \
    --max_target_length 256 \
    --beam_size 50 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 75 \
    --seed 123456  2>&1 | tee train.log