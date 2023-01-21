# !/bin/bash
python src/train.py \
        --config configs/2_0.yaml \
        model_name_or_path="bert-large-uncased" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        training_args.torch_compile=true \
        training_args.torch_compile_mode="default" \
        training_args.torch_compile_backend="inductor" \
        save_artifacts=false

# default compile mode
python src/train.py \
        --config configs/2_0.yaml \
        model_name_or_path="google/electra-large-discriminator" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=2e-5 \
        training_args.torch_compile=true \
        training_args.torch_compile_mode="default" \
        training_args.torch_compile_backend="inductor" \
        save_artifacts=false


# reduce-overhead compile mode
python src/train.py \
        --config configs/2_0.yaml \
        model_name_or_path="bert-large-uncased" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        training_args.torch_compile=true \
        training_args.torch_compile_mode="reduce-overhead" \
        training_args.torch_compile_backend="inductor" \
        save_artifacts=false

python src/train.py \
        --config configs/2_0.yaml \
        model_name_or_path="google/electra-large-discriminator" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=2e-5 \
        training_args.torch_compile=true \
        training_args.torch_compile_mode="reduce-overhead" \
        training_args.torch_compile_backend="inductor" \
        save_artifacts=false
        
# max-autotune compile mode
python src/train.py \
        --config configs/2_0.yaml \
        model_name_or_path="bert-large-uncased" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=3e-5 \
        training_args.torch_compile=true \
        training_args.torch_compile_mode="max-autotune" \
        training_args.torch_compile_backend="inductor" \
        save_artifacts=false

python src/train.py \
        --config configs/2_0.yaml \
        model_name_or_path="google/electra-large-discriminator" \
        training_args.per_device_train_batch_size=24 \
        training_args.per_device_eval_batch_size=24 \
        training_args.learning_rate=2e-5 \
        training_args.torch_compile=true \
        training_args.torch_compile_mode="max-autotune" \
        training_args.torch_compile_backend="inductor" \
        save_artifacts=false