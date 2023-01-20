#!/bin/bash
python train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="bert-large-uncased"

python train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="roberta-large"

python train.py \
        --config configs/1_0_large.yaml \
        model_name_or_path="albert-large-v2"