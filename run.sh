#!/bin/bash
python main.py \
        --config configs/1_0.yaml \
        model_name_or_path="bert-base-uncased" \
        per_device_train_batch_size=32 \
        per_device_eval_batch_size=32