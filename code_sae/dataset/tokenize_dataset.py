import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from code_sae.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize a dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="tokenization_config.json",
        help="Path to the config file",
    )

    return parser.parse_args()


def tokenize_dataset(
    dataset,
    tokenizer,
    column_name,
    max_length=None,
    padding=False,
    num_proc=4,
):
    """
    Tokenize a dataset with 1:1 row mapping.
    Each row in the original dataset maps to exactly one row in the output.
    """

    def tokenize_function(examples):
        texts = [text if text is not None else "" for text in examples[column_name]]

        if max_length:
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length" if padding else False,
                return_attention_mask=padding,
            )
        else:
            tokenized = tokenizer(
                texts,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

        result = {"input_ids": tokenized["input_ids"]}
        if padding and "attention_mask" in tokenized:
            result["attention_mask"] = tokenized["attention_mask"]
        return result

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    return tokenized


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        raise ValueError(f"Config path {config_path} does not exist.")

    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info("Loading config", path=config_path)

    # Extract config values
    tokenizer_name = config.get("tokenizer_name", "gpt2")
    dataset_path = config.get("dataset_path")
    dataset_name = config.get("dataset_name", None)
    split = config.get("split", "train")
    shuffle = config.get("shuffle", False)
    num_proc = config.get("num_proc", 4)
    max_length = config.get("max_length", None)
    padding = config.get("padding", True)
    column_name = config.get("column_name", "text")
    save_path = config.get("save_path", None)
    hf_repo_id = config.get("hf_repo_id", None)
    hf_num_shards = config.get("hf_num_shards", 1)
    trust_remote_code = config.get("dataset_trust_remote_code", False)

    logger.info("Config loaded",
                tokenizer=tokenizer_name,
                dataset=dataset_path,
                max_length=max_length)

    # Load tokenizer
    logger.info("Loading tokenizer", name=tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)

    # Load dataset
    logger.info("Loading dataset", path=dataset_path, name=dataset_name, split=split)
    dataset = load_dataset(
        dataset_path,
        name=dataset_name,
        split=split,
        trust_remote_code=trust_remote_code,
    )

    num_examples = len(dataset)
    logger.info("Dataset loaded", num_examples=num_examples)

    # Shuffle if requested
    if shuffle:
        logger.info("Shuffling dataset")
        dataset = dataset.shuffle(seed=42)

    # Tokenize dataset (1:1 row mapping)
    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        column_name=column_name,
        max_length=max_length,
        padding=padding,
        num_proc=num_proc,
    )

    logger.info("Tokenization complete", num_rows=len(tokenized_dataset))

    # Save locally if save_path is specified
    if save_path:
        logger.info("Saving to disk", path=save_path)
        os.makedirs(save_path, exist_ok=True)
        tokenized_dataset.save_to_disk(save_path)
        logger.info("Saved to disk")

    # Push to HuggingFace Hub if repo_id is specified
    if hf_repo_id:
        # Cap num_shards to dataset size
        actual_num_shards = min(hf_num_shards, len(tokenized_dataset))
        if actual_num_shards != hf_num_shards:
            logger.info(
                "Capping hf_num_shards to dataset size",
                old=hf_num_shards,
                new=actual_num_shards,
            )

        logger.info("Pushing to HuggingFace Hub", repo_id=hf_repo_id, num_shards=actual_num_shards)
        tokenized_dataset.push_to_hub(hf_repo_id, num_shards=actual_num_shards)
        logger.info("Pushed to HuggingFace Hub")


if __name__ == "__main__":
    main()
