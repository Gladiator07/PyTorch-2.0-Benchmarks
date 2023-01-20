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
        model_name_or_path="albert-large-v2" \
        per_device_train_batch_size=8 \
        per_device_eval_batch_size=8 \
        learning_rate=1e-5 \
        save_artifacts=false
