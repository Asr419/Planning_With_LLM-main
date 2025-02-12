python run.py \
    --backend gpt-35-turbo \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 20 \
    --log logs/rap/rap_test_f1.log \
    --algorithm 'rap' \
    ${@}