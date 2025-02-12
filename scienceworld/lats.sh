python run.py \
    --backend gpt-35-turbo \
    --task_start_index 94 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 10 \
    --log logs/lats/lats_test3_10.log \
    --algorithm 'lats' \
    ${@}