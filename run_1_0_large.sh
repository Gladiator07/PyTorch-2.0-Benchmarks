# !/bin/bash
python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="bert-large-uncased" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        save_artifacts=false

python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="roberta-large" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        save_artifacts=false
        
python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="microsoft/deberta-v3-large" \
        training_args.per_device_train_batch_size=16 \
        training_args.per_device_eval_batch_size=16 \
        training_args.learning_rate=2e-5 \
        save_artifacts=false

python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="google/electra-large-discriminator" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        save_artifacts=false

python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="funnel-transformer/xlarge" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        save_artifacts=false