# 2.0 + fp32
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --mixed_precision "no" \
    --wandb_enable \
    --torch_compile \
    --torch_compile_mode "default" \
    --torch_compile_backend "inductor" \
    --run_name "2.0_fp32" \
    --debug

# 2.0 + fp16
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --mixed_precision "fp16" \
    --wandb_enable \
    --torch_compile \
    --torch_compile_mode "default" \
    --torch_compile_backend "inductor" \
    --run_name "2.0_fp16" \
    --debug

# 2.0 + fp16 + gradient_checkpointing
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 64 \
    --learning_rate 4e-5 \
    --gradient_checkpointing \
    --mixed_precision "fp16" \
    --wandb_enable \
    --torch_compile \
    --torch_compile_mode "default" \
    --torch_compile_backend "inductor" \
    --run_name "2.0_gradckpt" \
    --debug

# 2.0 + fp16 + dynamic_padding
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --mixed_precision "fp16" \
    --wandb_enable \
    --dynamic_padding \
    --torch_compile \
    --torch_compile_mode "default" \
    --torch_compile_backend "inductor" \
    --torch_compile_dynamic \
    --run_name "2.0_dynamic_padding" \
    --debug