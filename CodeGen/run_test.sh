python run.py \
    --num_beams 1 \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/codegen-350m-multi \
    --model_name_or_path=../pretrain-models/codegen-350m-multi \
    --do_test \
    --test_data_file=../data/reef/reef_test.csv \
    --decoder_block_size 2048 \
    --output_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam1test.log


python run.py \
    --num_beams 3 \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/codegen-350m-multi \
    --model_name_or_path=../pretrain-models/codegen-350m-multi \
    --do_test \
    --test_data_file=../data/reef/reef_test.csv \
    --decoder_block_size 2048 \
    --output_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam3test.log


python run.py \
    --num_beams 5 \
    --model_name=fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../pretrain-models/codegen-350m-multi \
    --model_name_or_path=../pretrain-models/codegen-350m-multi \
    --do_test \
    --test_data_file=../data/reef/reef_test.csv \
    --decoder_block_size 2048 \
    --output_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam5test.log
