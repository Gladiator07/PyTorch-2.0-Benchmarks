import argparse
import glob
import os
import shutil
import time
import evaluate
import numpy as np
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def delete_checkpoints(dir):
    for file in glob.glob(f"{dir}/checkpoint-*"):
        shutil.rmtree(file, ignore_errors=True)


def asHours(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    cfg.merge_with_dotlist(unknown_args)
    return cfg


if __name__ == "__main__":

    args = parse_args()

    set_seed(args.training_args.seed)
    if args.wandb_enable:
        wandb.init(
            name=args.name,
            project=args.wandb_project,
            tags=args.tags,
            config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
        )

    OUTPUT_DIR = args.training_args.output_dir = os.path.join(
        args.training_args.output_dir, args.model_name_or_path.replace("/", "-")
    )
    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    ds = load_dataset("imdb")
    label_list = ds["train"].features["label"].names
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    accuracy = evaluate.load("accuracy")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    )

    def tokenize_func(example):
        return tokenizer(
            example["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
            padding=padding,
        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    train_ds, eval_ds = ds["train"], ds["test"]

    if args.max_train_samples is not None:
        max_train_samples = min(len(train_ds), args.max_train_samples)
        train_ds = train_ds.select(range(max_train_samples))
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_ds), args.max_eval_samples)
        eval_ds = eval_ds.select(range(max_eval_samples))

    train_ds = train_ds.map(tokenize_func, batched=True, num_proc=2)
    eval_ds = eval_ds.map(tokenize_func, batched=True, num_proc=2)

    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    train_start_time = time.perf_counter()
    trainer = Trainer(
        model,
        args=TrainingArguments(**args.training_args),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    train_end_time = time.perf_counter()
    elapsed_time = train_end_time - train_start_time

    delete_checkpoints(OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)

    if args.wandb_enable:
        if args.save_artifacts:
            model_artifact = wandb.Artifact(name=args.name, type="model")
            model_artifact.add_dir(OUTPUT_DIR)
            wandb.log_artifact(model_artifact)

        wandb.log({"train/runtime": elapsed_time})
        # log time in human readable format
        wandb.log({"train/runtime_hr": asHours(elapsed_time)})
        wandb.finish()
