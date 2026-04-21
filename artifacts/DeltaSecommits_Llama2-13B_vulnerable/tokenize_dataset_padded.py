#!/usr/bin/env python
"""Tokenize dataset with padding to max_length."""

import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

from code_sae.logger import logger


def tokenize_dataset_padded(config_file):
    """Load config and tokenize dataset with padding."""
    with open(config_file, "r") as f:
        config = json.load(f)

    logger.info("Loading config", path=config_file)

    tokenizer_name = config["tokenizer_name"]
    dataset_path = config["dataset_path"]
    max_length = config["max_length"]
    column_name = config["column_name"]
    hf_repo_id = config["hf_repo_id"]

    logger.info(
        "Config loaded",
        tokenizer=tokenizer_name,
        dataset=dataset_path,
        max_length=max_length,
    )

    # Load tokenizer
    logger.info("Loading tokenizer", name=tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load dataset
    logger.info("Loading dataset", path=dataset_path, split="train")
    dataset = load_dataset(dataset_path, split="train", trust_remote_code=False)
    logger.info("Dataset loaded", num_examples=len(dataset))

    # Tokenize with padding to max_length
    def tokenize_function(examples):
        texts = [text if text is not None else "" for text in examples[column_name]]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    logger.info("Tokenization complete", num_rows=len(tokenized_dataset))

    # Split into train/test
    split_dataset = tokenized_dataset.train_test_split(
        train_size=0.8,
        seed=42,
    )
    logger.info(
        "Split dataset",
        train_size=len(split_dataset["train"]),
        test_size=len(split_dataset["test"]),
    )

    # Push to HuggingFace Hub
    from datasets import DatasetDict

    dataset_dict = DatasetDict(
        {
            "train": split_dataset["train"],
            "test": split_dataset["test"],
        }
    )
    logger.info("Pushing to HuggingFace Hub", repo_id=hf_repo_id)
    dataset_dict.push_to_hub(
        repo_id=hf_repo_id,
        num_shards={"train": 32, "test": 32},
        private=False,
    )
    logger.info("Pushed to HuggingFace Hub")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tokenize_dataset_padded.py <config_file>")
        sys.exit(1)

    tokenize_dataset_padded(sys.argv[1])
