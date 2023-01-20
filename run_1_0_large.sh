# !/bin/bash
python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="bert-large-uncased" \
        per_device_train_batch_size=24 \
        per_device_eval_batch_size=24 \
        learning_rate=3e-5 \
        save_artifacts=false

python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="roberta-large" \
        per_device_train_batch_size=24 \
        per_device_eval_batch_size=24 \
        learning_rate=3e-5 \
        save_artifacts=false
        
python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="microsoft/deberta-v2-xlarge" \
        per_device_train_batch_size=16 \
        per_device_eval_batch_size=16 \
        learning_rate=2e-5 \
        save_artifacts=false