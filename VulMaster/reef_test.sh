name=reef_fine_tuning_model
CUDA_VISIBLE_DEVICES=0,1
SLURM_NTASKS=1
python test_model.py \
        --eval_data  data/reef/reef_test.json \
        --write_results \
        --model_path ./checkpoint_final/reef_fine_tuning_model/test_results/ \
        --per_gpu_eval_batch_size 1 \
        --n_context 10 \
	      --beam_size 1 \
        --text_maxlength 512 \
        --answer_maxlength 512 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint_final \
        | tee ./checkpoint_final/reef_fine_tuning_model/test_results/beam1.log


python test_model.py \
        --eval_data  data/reef/reef_test.json \
        --write_results \
        --model_path ./checkpoint_final/reef_fine_tuning_model/test_results/ \
        --per_gpu_eval_batch_size 1 \
        --n_context 10 \
	      --beam_size 3 \
        --text_maxlength 512 \
        --answer_maxlength 512 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint_final \
        | tee ./checkpoint_final/reef_fine_tuning_model/test_results/beam3.log


python test_model.py \
        --eval_data  data/reef/reef_test.json \
        --write_results \
        --model_path ./checkpoint_final/reef_fine_tuning_model/test_results/ \
        --per_gpu_eval_batch_size 1 \
        --n_context 10 \
	      --beam_size 5 \
        --text_maxlength 512 \
        --answer_maxlength 512 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint_final \
        | tee ./checkpoint_final/reef_fine_tuning_model/test_results/beam5.log