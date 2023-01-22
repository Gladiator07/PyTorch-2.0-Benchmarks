""" Finetuning a 🤗 Transformers model for sequence classification on imdb.
Note: This script is modified from https://github.com/sgugger/torchdynamo-tests"""
import argparse
import logging
import os
import time

import datasets
import evaluate
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
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
torch.backends.cuda.matmul.allow_tf32 = True  # use tensor cores
logger = get_logger(__name__)


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
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16"],
        help="`no` or `fp16`",
    )

    # torch compile args
    parser.add_argument(
        "--torch_compile", action="store_true", help="If passed, compile the model"
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotone"],
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
        default="PyTorch 2.0 Benchmarks - HF Accelerate",
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
    set_seed(args.seed)

    accelerator = Accelerator(
        dynamo_backend=args.torch_compile_backend if args.torch_compile else None,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.wandb_enable else None,
    )

    if args.wandb_enable:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

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
    )

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

    with accelerator.main_process_first():
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
    elif args.dynamic_padding and args.mixed_precision == "fp16":
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

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # enable gradient checkpointing if passed
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # compile model
    # if args.torch_compile:
    #     logger.info("=== Compiling model ===")
    #     logger.info(f"mode: {args.torch_compile_mode}")
    #     logger.info(f"backend: {args.torch_compile_backend}")
    #     logger.info(f"dynamic: {args.torch_compile_dynamic}")

    #     model = torch.compile(
    #         model,
    #         mode=args.torch_compile_mode,
    #         dynamic=args.torch_compile_dynamic,
    #         backend=args.torch_compile_backend,
    #     )

    # Get the metric function
    metric = evaluate.load("accuracy")
    # Train!
    # Only show the progress bar once on each machine.
    train_steps = len(train_dataloader) * args.num_epochs
    progress_bar = tqdm(
        range(train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    train_start_time = time.perf_counter()
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            predictions, references = accelerator.gather_for_metrics(
                (outputs.logits.argmax(dim=-1), batch["labels"])
            )
            metric.add_batch(predictions=predictions, references=references)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if step == 0 and epoch == 0:
                    first_train_step_time = time.perf_counter() - train_start_time
                accelerator.log({"train/loss": loss.item()})

        train_accuracy = metric.compute()["accuracy"]
        accelerator.log({"train/accuracy": train_accuracy})
        # accelerator.print(f"Training Accuracy at epoch {epoch}: {train_accuracy:.3f}")

    total_training_time = time.perf_counter() - train_start_time
    avg_train_iteration_time = (total_training_time - first_train_step_time) / (
        train_steps - 1
    )
    train_iterations_per_sec = 1 / avg_train_iteration_time
    train_samples_per_sec = train_iterations_per_sec * args.batch_size
    accelerator.print("Training finished.")
    accelerator.print(f"First train iteration took: {first_train_step_time:.2f}s")
    accelerator.print(
        f"Average train iteration time after the first iteration: {avg_train_iteration_time * 1000:.2f}ms"
    )
    accelerator.print(f"Train iterations per second: {train_iterations_per_sec:.2f}")
    accelerator.print(f"Train samples per second: {train_samples_per_sec:.2f}")
    accelerator.print(f"Total training time: {total_training_time:.2f}s")
    accelerator.print(f"Total training time (h:mm:ss): {asHours(total_training_time)}")

    # inference
    model.eval()
    progress_bar = tqdm(
        eval_dataloader,
        total=len(eval_dataloader),
        disable=not accelerator.is_local_main_process,
        desc="Inference",
    )
    inference_start_time = time.perf_counter()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=1)
        predictions, references = accelerator.gather_for_metrics(
            (predictions, batch["labels"])
        )
        metric.add_batch(predictions=predictions, references=references)
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
    accelerator.print(f"Test Accuracy: {test_accuracy:.3f}")
    accelerator.log({"test/accuracy": test_accuracy})

    accelerator.print("Inference finished.")
    accelerator.print(f"First inference iteration took: {first_infer_step_time:.2f}s")
    accelerator.print(
        f"Average inference iteration time after the first iteration: {avg_inference_iteration_time * 1000:.2f}ms"
    )
    accelerator.print(
        f"Inference iterations per second: {inference_iterations_per_sec:.2f}"
    )
    accelerator.print(f"Inference samples per second: {inference_samples_per_sec:.2f}")
    accelerator.print(f"Total inference time: {total_inference_time:.2f}s")
    accelerator.print(
        f"Total inference time (h:mm:ss): {asHours(total_inference_time)}"
    )

    accelerator.log(
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

    if accelerator.is_main_process and args.wandb_enable:
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v


if __name__ == "__main__":
    main()
