# 1.13 + fp32
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --mixed_precision "no" \
    --wandb_enable \
    --run_name "1.13_fp32"

# 1.13 + fp16
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --mixed_precision "fp16" \
    --wandb_enable \
    --run_name "1.13_fp16"

# 1.13 + fp16 + gradient_checkpointing
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 48 \
    --learning_rate 3e-5 \
    --gradient_checkpointing \
    --mixed_precision "fp16" \
    --wandb_enable \
    --run_name "1.13_gradckpt"

# 1.13 + fp16 + dynamic_padding
accelerate launch src/train_accelerate.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --mixed_precision "fp16" \
    --wandb_enable \
    --dynamic_padding \
    --run_name "1.13_dynamic_padding"