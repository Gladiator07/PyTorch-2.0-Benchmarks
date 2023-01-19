#!/bin/bash
PYTORCH_VERSION="1.13"
MODEL_NAME="bert-base-uncased"
python main.py \
        # data args
        --dataset_name "imdb" \
        --max_seq_length 512 \
        --pad_to_max_length True \
        --max_train_samples 1000 \
        --max_eval_samples 1000 \
        # model args
        --model_name_or_path $MODEL_NAME \
        # wandb args
        --wandb_enable True \
        --name "$MODEL_NAME_$PYTORCH_VERSION" \
        --wandb_project "PyTorch 2.0 Benchmarks" \
        --tags "$MODEL_NAME,$PYTORCH_VERSION"
        --save_arifacts True \
        # training args
        --output_dir "artifacts" \
        --evaluation_strategy "epoch" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --num_train_epochs 3 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.2 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --seed 42 \
        --fp16 False \
        --dataloader_num_workers 6 \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_accuracy" \
        --greater_is_better True \
        --group_by_length False \
        --report_to "wandb" \
        --dataloader_pin_memory True \
        --gradient_checkpointing False


