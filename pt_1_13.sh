# 1.13 + fp32
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --wandb_enable \
    --run_name "1.13_fp32"

# 1.13 + fp16
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --fp16 \
    --wandb_enable \
    --run_name "1.13_fp16"

# 1.13 + fp16 + gradient_checkpointing
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 64 \
    --learning_rate 4e-5 \
    --gradient_checkpointing \
    --fp16 \
    --wandb_enable \
    --run_name "1.13_gradckpt"

# 1.13 + fp16 + dynamic_padding
python src/train_pt.py \
    --model_name_or_path "bert-large-uncased" \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --fp16 \
    --wandb_enable \
    --dynamic_padding \
    --run_name "1.13_dynamic_padding"