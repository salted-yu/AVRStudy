python codebert_main.py \
    --beam_size 1 \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/codebert-base \
    --model_name_or_path=../pretrain-models/codebert-base \
    --do_test \
    --test_data_file=../data/reef/reef_test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam1_test.log

python codebert_main.py \
    --beam_size 3 \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/codebert-base \
    --model_name_or_path=../pretrain-models/codebert-base \
    --do_test \
    --test_data_file=../data/reef/reef_test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam3_test.log

python codebert_main.py \
    --beam_size 5 \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/codebert-base \
    --model_name_or_path=../pretrain-models/codebert-base \
    --do_test \
    --test_data_file=../data/reef/reef_test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam5_test.log