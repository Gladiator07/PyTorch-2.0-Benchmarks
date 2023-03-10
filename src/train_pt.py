""" Finetuning a 🤗 Transformers model for sequence classification on imdb.
Note: This script is modified from https://github.com/sgugger/torchdynamo-tests"""
import argparse
import logging
import os
import time
from logging import getLogger
from pprint import pprint

import evaluate
import torch
import wandb
from datasets import load_dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

logger = getLogger(__name__)


def asHours(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    # data args
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded unless `--dynamic_padding` is passed."
        ),
    )
    parser.add_argument(
        "--dynamic_padding",
        action="store_true",
        help="If passed, pad all samples to maximum length in a batch (dynamic padding). \
        Otherwise, all the samples are padded uniformly to `max_length`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="bert-large-uncased",
    )
    # training args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the dataloaders.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If passed, use gradient checkpointing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--fp16", action="store_true", help="If passed, enable mixed precision training"
    )

    # torch compile args
    parser.add_argument(
        "--torch_compile", action="store_true", help="If passed, compile the model"
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="mode to use for torch compile",
    )
    parser.add_argument(
        "--torch_compile_backend",
        type=str,
        default="no",
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "nvfuser",
            "aot_nvfuser",
            "aot_cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
            "ipex",
        ],
        help="torch compile backend aka dynamo backend",
    )
    parser.add_argument(
        "--torch_compile_dynamic",
        action="store_true",
        help="whether to enable the code path for Dynamic Shapes",
    )
    # wandb args
    parser.add_argument(
        "--wandb_enable",
        action="store_true",
        help="Enable Weights & Biases logging if passed",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="PyTorch 2.0 Benchmarks v2",
        help="Weights & Biases project name",
    )
    parser.add_argument("--run_name", type=str, default="baseline", help="W&B run name")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (train and infer on 1000 samples)",
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(f"===== Configuration =====")
    pprint(vars(args))
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_enable:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load data
    raw_datasets = load_dataset("imdb")
    del raw_datasets["unsupervised"]

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    if args.debug:
        logger.warning("Debug mode enabled: training and inference on 1000 samples")
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].select(range(1000))
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    ).to(device)

    # Preprocessing the datasets
    # if dynamic padding is true, pad the inputs later
    padding = False if args.dynamic_padding else "max_length"

    def tokenize_func(example):
        return tokenizer(
            example["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_length,
            padding=padding,
        )

    processed_datasets = raw_datasets.map(
        tokenize_func,
        batched=True,
        remove_columns=["text"],
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    # DataLoaders creation:
    if not args.dynamic_padding:
        logger.info("Using default data collator")
        data_collator = default_data_collator
    elif args.dynamic_padding and args.fp16:
        logger.info(
            "Using DataCollatorWithPadding and pad_to_multiple_of=8 (dynamic padding enabled + fp16)"
        )
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        logger.info("Using DataCollatorWithPadding (dynamic padding enabled)")
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=min(6, os.cpu_count()),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        drop_last=not args.dynamic_padding,
        pin_memory=True,
        num_workers=min(6, os.cpu_count()),
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0.2 * num_training_steps,
        num_training_steps=num_training_steps,
    )

    # enable gradient checkpointing if passed
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # compile model
    if args.torch_compile:
        logger.info("=== Compiling model ===")
        logger.info(f"mode: {args.torch_compile_mode}")
        logger.info(f"backend: {args.torch_compile_backend}")
        logger.info(f"dynamic: {args.torch_compile_dynamic}")

        model = torch.compile(
            model,
            mode=args.torch_compile_mode,
            dynamic=args.torch_compile_dynamic,
            backend=args.torch_compile_backend,
        )

    # Get the metric function
    metric = evaluate.load("accuracy")

    # Train!
    train_steps = len(train_dataloader) * args.num_epochs
    progress_bar = tqdm(range(train_steps), desc="Training")

    scaler = GradScaler(enabled=args.fp16)
    train_start_time = time.perf_counter()
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            with autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            metric.add_batch(
                predictions=outputs.logits.argmax(dim=-1), references=batch["labels"]
            )
            progress_bar.update(1)
            if step == 0 and epoch == 0:
                first_train_step_time = time.perf_counter() - train_start_time

                if args.wandb_enable:
                    wandb.log({"train/loss": loss.item()})

        train_accuracy = metric.compute()["accuracy"]

        if args.wandb_enable:
            wandb.log({"train/accuracy": train_accuracy})
        # print(f"Training Accuracy at epoch {epoch}: {train_accuracy:.3f}")

    total_training_time = time.perf_counter() - train_start_time
    avg_train_iteration_time = (total_training_time - first_train_step_time) / (
        train_steps - 1
    )
    train_iterations_per_sec = 1 / avg_train_iteration_time
    train_samples_per_sec = train_iterations_per_sec * args.batch_size
    print("Training finished.")
    print(f"First train iteration took: {first_train_step_time:.2f}s")
    print(
        f"Average train iteration time after the first iteration: {avg_train_iteration_time * 1000:.2f}ms"
    )
    print(f"Train iterations per second: {train_iterations_per_sec:.2f}")
    print(f"Train samples per second: {train_samples_per_sec:.2f}")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Total training time (h:mm:ss): {asHours(total_training_time)}")

    # inference
    model.eval()
    progress_bar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Inference")

    inference_start_time = time.perf_counter()
    for step, batch in enumerate(eval_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            with autocast(enabled=args.fp16):
                outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)
        if step == 0:
            first_infer_step_time = time.perf_counter() - inference_start_time

    total_inference_time = time.perf_counter() - inference_start_time
    avg_inference_iteration_time = (total_inference_time - first_infer_step_time) / (
        len(eval_dataloader) - 1
    )
    inference_iterations_per_sec = 1 / avg_inference_iteration_time
    inference_samples_per_sec = inference_iterations_per_sec * args.batch_size

    test_accuracy = metric.compute()["accuracy"]
    print(f"Test Accuracy: {test_accuracy:.3f}")

    if args.wandb_enable:
        wandb.log({"test/accuracy": test_accuracy})

    print("Inference finished.")
    print(f"First inference iteration took: {first_infer_step_time:.2f}s")
    print(
        f"Average inference iteration time after the first iteration: {avg_inference_iteration_time * 1000:.2f}ms"
    )
    print(f"Inference iterations per second: {inference_iterations_per_sec:.2f}")
    print(f"Inference samples per second: {inference_samples_per_sec:.2f}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Total inference time (h:mm:ss): {asHours(total_inference_time)}")

    if args.wandb_enable:
        wandb.log(
            {
                "train/first_iter_time": first_train_step_time,
                "infer/first_iter_time": first_infer_step_time,
                "train/avg_iter_time": avg_train_iteration_time,
                "infer/avg_iter_time": avg_inference_iteration_time,
                "train/iter_per_sec": train_iterations_per_sec,
                "infer/iter_per_sec": inference_iterations_per_sec,
                "train/samples_per_sec": train_samples_per_sec,
                "infer/samples_per_sec": inference_samples_per_sec,
                "train/total_time": total_training_time,
                "infer/total_time": total_inference_time,
            }
        )

    summary_dict = {
        **vars(args),
        "train/first_iter_time": f"{first_train_step_time:.2f} s",
        "infer/first_iter_time": f"{first_infer_step_time:.2f} s",
        "train/avg_iter_time": f"{avg_train_iteration_time * 1000:.2f} ms",
        "infer/avg_iter_time": f"{avg_inference_iteration_time* 1000:.2f} ms",
        "train/iter_per_sec": f"{train_iterations_per_sec:.2f}",
        "infer/iter_per_sec": f"{inference_iterations_per_sec:.2f}",
        "train/samples_per_sec": f"{train_samples_per_sec:.2f}",
        "infer/samples_per_sec": f"{inference_samples_per_sec:.2f}",
        "train/total_time": f"{total_training_time:.2f}",
        "infer/total_time": f"{total_inference_time:.2f}",
        "train/total_time_hr": f"{asHours(total_training_time)}",
        "infer/total_time_hr": f"{asHours(total_inference_time)}",
    }

    if args.wandb_enable:
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v


if __name__ == "__main__":
    main()
