export CUDA_VISIBLE_DEVICES=0,1


##################### arc-challenge
python -m eval.run_eval \
    --eval_dataset arc-challenge \
    --data_dir data/eval/arc_challenge \
    --save_dir results/arc_challenge/mistral_7b/ToW \
    --model_name_or_path /data/data/mshen16/fine_nwp/output/mistral_7b/ToW \
    --tokenizer_name_or_path /data/data/mshen16/fine_nwp/output/mistral_7b/ToW \
    --newline_stop \
    --stop_at_triple_newline \
    --clm_max_length 512


python -m eval.run_eval \
    --eval_dataset arc-challenge \
    --data_dir data/eval/arc_challenge \
    --save_dir results/arc_challenge/mistral_7b/ToW-NoDeN \
    --model_name_or_path /data/data/mshen16/fine_nwp/output/mistral_7b/ToW-NoDeN \
    --tokenizer_name_or_path /data/data/mshen16/fine_nwp/output/mistral_7b/ToW-NoDeN \
    --newline_stop \
    --stop_at_triple_newline \
    --clm_max_length 2048