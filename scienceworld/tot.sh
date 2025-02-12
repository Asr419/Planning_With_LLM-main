python run.py \
    --backend gpt-35-turbo \
    --task_start_index 88 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 50 \
    --log logs/tot/tot_test_f1.log \
    --algorithm 'tot' \
    ${@}