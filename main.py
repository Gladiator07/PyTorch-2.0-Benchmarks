import os
import glob
import shutil
import numpy as np

from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List

import wandb
import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Source: https://github.com/huggingface/transformers/blob/0359e2e15f4504513fd2995bdd6dd654c747b313/examples/pytorch/text-classification/run_glue.py#L71
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )


@dataclass
class WandbArguments:
    """
    Arguments pertaining to wandb logging
    """

    wandb_enable: Optional[bool] = field(
        default=False, metadata={"help": "enable wandb logging"}
    )
    name: Optional[str] = field(
        default=None, metadata={"help": "wandb name for the run"}
    )
    wandb_project: Optional[str] = field(
        default="PyTorch 2.0 Benchmarks", metadata={"help": "wandb project name"}
    )
    tags: Optional[str] = field(
        default=None,
        metadata={"help": "wandb tags for the run (delimited list of strings)"},
    )
    save_artifacts: Optional[bool] = field(
        default=False,
        metadata={"help": "save model checkpoints as wandb artifacts"},
    )


def delete_checkpoints(dir):
    for file in glob.glob(f"{dir}/checkpoint-*"):
        shutil.rmtree(file, ignore_errors=True)


if __name__ == "__main__":

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, WandbArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        wandb_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    if wandb_args.wandb_enable:
        wandb.init(
            name=wandb_args.name,
            project=wandb_args.wandb_project,
            tags=wandb_args.tags.split(",") if wandb_args.tags else None,
        )

    training_args.output_dir = os.path.join(
        training_args.output_dir, model_args.model_name_or_path.replace("/", "-")
    )
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    ds = load_dataset("imdb")
    label_list = ds["train"].features["label"].names
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    accuracy = evaluate.load("accuracy")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=num_labels
    )

    def tokenize_func(example):
        return tokenizer(
            example["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=padding,
        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    train_ds, eval_ds = ds["train"], ds["test"]

    train_ds = train_ds.map(tokenize_func, batched=True)
    eval_ds = eval_ds.map(tokenize_func, batched=True)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_ds), data_args.max_train_samples)
        train_dataset = train_ds.select(range(max_train_samples))
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_ds), data_args.max_eval_samples)
        eval_dataset = eval_ds.select(range(max_eval_samples))

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    delete_checkpoints(training_args.output_dir)

    if wandb_args.wandb_enable:
        if wandb_args.save_artifacts:
            model_artifact = wandb.Artifact(name=wandb_args.name, type="model")
            model_artifact.add_dir(training_args.output_dir)
            wandb.log_artifact(model_artifact)

        wandb.finish()
