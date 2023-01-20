#!/bin/bash
python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="bert-large-uncased"

python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="roberta-large"

python src/train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="albert-large-v2"