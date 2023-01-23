# 2.0 + fp32
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --torch_compile \
    --torch_compile_mode "default" \
    --torch_compile_backend "inductor" \
    --wandb_enable \
    --run_name "2.0_fp32"

# 2.0 + fp16
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --fp16 \
    --torch_compile \
    --torch_compile_mode "default" \
    --torch_compile_backend "inductor" \
    --wandb_enable \
    --run_name "2.0_fp16"

# 2.0 + fp16 + reduce-overhead
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --fp16 \
    --torch_compile \
    --torch_compile_mode "reduce-overhead" \
    --torch_compile_backend "inductor" \
    --wandb_enable \
    --run_name "2.0_fp16_reduce-overhead"