"""
trainer_sft.py — SFT training module using TRL SFTTrainer.

Yike Zhang @Arcadia_Ebendie

Expects processed_data to be a list of {"text": "..."} dicts,
consistent with Member 4's build_training_examples() output.

Usage (called from train_sft.py main entry):
    trainer = build_trainer(processed_data, config)
    trainer.train()
"""

import os
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from src.training.lora_utils import load_model_and_tokenizer


def build_trainer(processed_data, config):
    """
    Build a TRL SFTTrainer from processed data and config.

    Args:
        processed_data: list of {"text": "..."} dicts
        config: dict loaded from yaml config file

    Returns:
        SFTTrainer instance (call .train() to start)
    """
    # load model + tokenizer + LoRA
    model, tokenizer = load_model_and_tokenizer(config)

    # convert list[dict] to HF Dataset
    dataset = Dataset.from_list(processed_data)

    # training arguments
    train_cfg = config.get("training", {})
    output_dir = train_cfg.get("output_dir", "outputs/default_run")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=config.get("run_name", "sft_run"),
        seed=config.get("seed", 42),
        # steps / epochs
        max_steps=train_cfg.get("max_steps", -1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        # batch size
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        # optimizer
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        # mixed precision
        bf16=train_cfg.get("bf16", torch.cuda.is_available()),
        fp16=False,
        # logging
        logging_steps=train_cfg.get("logging_steps", 1),
        # checkpoint
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        # misc
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=train_cfg.get("max_seq_length", 1024),
    )

    return trainer
