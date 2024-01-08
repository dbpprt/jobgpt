import random
import re

import torch

# pyright: reportPrivateImportUsage=false
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from accelerate import Accelerator


def get_dataloaders(accelerator: Accelerator, model_name: str, batch_size: int, seq_len: int):
    # some boilerplate required to work with different model architectures
    # TODO: really required?
    if any(k in model_name for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # attempt to fix "Cannot handle batch sizes > 1 if no padding token is defined." with gpt2
    # if getattr(tokenizer, "pad_token") is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    def __preprocess_function(qualifications, descriptions) -> None:
        batch_size = len(qualifications)
        inputs = [
            f"Write a modern and engaging job posting for the following basic qualifications: {x}\r\nResponse: \r\n"
            for x in qualifications
        ]
        targets = [str(x) for x in descriptions]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)

        # (heavily) inspired by hf peft example
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            # all labels set to -100 are ignored (masked), the loss is only computed for the labels set to [0, ..., config.vocab_size - 1]
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                seq_len - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (seq_len - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (seq_len - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:seq_len])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:seq_len])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:seq_len])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function(examples: dict):
        qualifications = [x.replace("· ", "\r\n- ") for x in examples["BASIC QUALIFICATIONS"] if x is not None]
        preferred_qualifications = [
            x.replace("· ", "\r\n- ") for x in examples["PREFERRED QUALIFICATIONS"] if x is not None
        ]

        qualifications = [f"{x}{y}" for x, y in zip(qualifications, preferred_qualifications)]

        for idx, qualification in enumerate(qualifications):
            qualification = qualification.split("\r\n- ")
            # remove empty strings
            qualification = [x for x in qualification if x]
            random.shuffle(qualification)
            qualifications[idx] = "\r\n- " + "\r\n- ".join(qualification)

        # fix some formatting issues in the dataset
        descriptions = [x.replace("· ", "\r\n- ") for x in examples["DESCRIPTION"] if x is not None]
        descriptions = [re.sub(r"([.\-,;:!?])(?=\S)", r"\1 ", x) for x in descriptions]

        return __preprocess_function(qualifications, descriptions)

    data_files = {"train": "./data/amazon_jobs_dataset.csv"}

    dataset = load_dataset(
        "csv",
        data_files=data_files,
    )

    with accelerator.main_process_first():
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            # ideally we should use the cache, but this is a sample script, hence we don't
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    accelerator.wait_for_everyone()

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )

    return train_dataloader, tokenizer
