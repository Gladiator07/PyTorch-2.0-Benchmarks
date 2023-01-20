import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


def tokenize_func(example):
    tokenized = tokenizer(
        example["text"],
        add_special_tokens=True,
    )
    seq_length = {"seq_length": len(tokenized["input_ids"])}
    return seq_length


dataset = load_dataset("imdb")
train_ds, eval_ds = dataset["train"], dataset["test"]

train_ds = train_ds.map(tokenize_func, num_proc=2)
eval_ds = eval_ds.map(tokenize_func, num_proc=2)

train_df = train_ds.to_pandas()
eval_df = eval_ds.to_pandas()
train_table = wandb.Table(dataframe=train_df)
eval_table = wandb.Table(dataframe=eval_df)

wandb.init(project="PyTorch 2.0 Benchmarks", name="data-exploration", job_type="upload")
train_artifact = wandb.Artifact(name="train_imdb", type="dataset")
train_artifact.add(train_table, "train_table")

eval_artifact = wandb.Artifact(name="eval_imdb", type="dataset")
eval_artifact.add(eval_table, "eval_table")
wandb.log_artifact(train_artifact)
wandb.log_artifact(eval_artifact)

wandb.log(
    {
        "train hist": wandb.plot.histogram(
            train_table, "seq_length", title="Train Sequence Length Histogram"
        )
    }
)
wandb.log(
    {
        "eval_hist": wandb.plot.histogram(
            eval_table, "seq_length", title="Eval Sequence Length Histogram"
        )
    }
)
wandb.finish()
