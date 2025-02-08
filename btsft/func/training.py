import os
import random
import torch
import bitsandbytes as bnb
from datetime import datetime
from typing import Optional

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
    get_cosine_schedule_with_warmup
)

from btsft.func.format_reward import format_reward_func
from btsft.func.parameters import get_parameters_count
from btsft.func.mapping import map_iio, map_conversations

from transformers.trainer_pt_utils import get_parameter_names
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel

def train(
    model_name: str,
    checkpoint: str,
    threshold: float = 0.2,
    bf_beta: float = 0.05,
    lora_rank: int = 64,
    dataset_train: str = None,
    max_length: int = 512,
    batch_size: int = 32,
    accumulation_iter: int = 128,
    epochs: int = 1,
    lr: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_dir: str = "./logs",
    output_dir: str = "./results",
    save_steps: int = 500,
    train_test_split: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    num_workers: int = 24,
    skip: int = 0,
    take: Optional[int] = None,
    tokenizer_name: Optional[str] = None,
    trainer_checkpoint: Optional[str] = None,
    response_template: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> None:
    # Set unsloth environment variable
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    print(f"Training {model_name} on {dataset_train} with {tokenizer_name} tokenizer.")

    set_seed(seed)

    if tokenizer_name is None:
        tokenizer_name = checkpoint

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        load_in_4bit=True,
        max_lora_rank=lora_rank,
        cache_dir=cache_dir,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    print("Model parameters:", get_parameters_count(model))
    dataset_validation = None

    if dataset_train is not None:
        datasets = load_dataset(dataset_train, 
                                split="train",
                                cache_dir=cache_dir)

        if "conversations" in datasets.column_names:
            datasets = datasets.map(lambda x: {'conversations': map_conversations(x['conversations'])},
                                    num_proc=num_workers).filter(
                lambda x: x["conversations"] is not None, num_proc=num_workers
            )
        else:
            datasets = datasets.map(
                lambda x: {"conversations": map_iio(x)}, num_proc=num_workers
            ).filter(lambda x: x["conversations"] is not None, num_proc=num_workers)

        datasets = [datasets]
    else:
        raise ValueError("Dataset not provided.")
    
    if len(datasets[0]) == 0:
        raise ValueError("No valid examples found in the dataset.")

    dataset = concatenate_datasets(datasets).shuffle(seed=seed)

    if take is not None:
        dataset = dataset.take(take)

    if not dataset_validation:
        dataset = dataset.train_test_split(test_size=train_test_split)

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n" if response_template is None else response_template

    def tokenize_function(examples):
        """
        Tokenize the examples and mask tokens between <think> tags.
        This is the main function for tokenizing the dataset introducing Blurred Thoughts.
        Setting the label to -100 effectively masks the token for the model. 
        This prevents the model from strictly following the training data and encourages it to produce more diverse responses, aligned with its own probability distribution.

        Args:
            examples: Examples from the dataset

        Returns:
            Dictionary containing input_ids, attention_mask and labels
        """
        tokens = tokenizer(
            tokenizer.apply_chat_template(
                examples["conversations"], tokenize=False, add_generation_prompt=False
            ),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            verbose=False,
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        think_open_token = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_close_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
        response_template_len = len(
            tokenizer.encode(response_template, add_special_tokens=False)
        )

        masked_counter = 0

        for i, token in enumerate(tokens["labels"]):
            think_open = -1
            for j, y in enumerate(token):
                if y == think_open_token:
                    think_open = j
                    tokens["labels"][i][: j - response_template_len - 1] = -100
                    continue

                if y == think_close_token:
                    think_open = -1
                    break

                if think_open > 0 and think_open + 5 < j:
                    if random.random() < threshold:
                        tokens["labels"][i][j] = -100
                        masked_counter += 1

        tokens["input_ids"] = tokens["input_ids"].squeeze(0)
        tokens["attention_mask"] = tokens["attention_mask"].squeeze(0)
        tokens["labels"] = tokens["labels"].squeeze(0)

        return tokens

    items_to_skip = skip

    print("Tokenizing dataset...")
    if not dataset_validation:
        train_samples = dataset["train"].shape[0]
        test_samples = dataset["test"].shape[0]
        print(f"{train_samples} training samples, {test_samples} test samples")
        if trainer_checkpoint is None:
            train_ds = (
                dataset["train"]
                .skip(items_to_skip)
                .map(
                    tokenize_function,
                    num_proc=num_workers,
                    remove_columns=dataset["train"].column_names,
                )
            )
            test_ds = dataset["test"].map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
        else:
            train_ds = dataset["train"].map(
                tokenize_function,
                num_proc=num_workers,
                batched=False,
                remove_columns=dataset["train"].column_names,
            )
            test_ds = dataset["test"].map(
                tokenize_function,
                num_proc=num_workers,
                batched=False,
                remove_columns=dataset["train"].column_names,
            )
        print(
            f"{dataset['train'].shape[0]} training samples, {dataset['test'].shape[0]} test samples"
        )
    else:
        if trainer_checkpoint is None:
            train_ds = dataset.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
            test_ds = dataset_validation.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
        else:
            train_ds = dataset.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
            test_ds = dataset_validation.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )

        train_samples = dataset.shape[0] - items_to_skip
        print(
            f"{train_samples} training samples, {dataset_validation.dataset_size} test samples"
        )

    max_steps = train_samples // (batch_size * accumulation_iter) * epochs

    print(f"Batch size {batch_size}")

    output_dir = os.path.join(
        os.getcwd(),
        output_dir,
        model_name,
        datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n in decay_parameters
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    print("Setting up optimizer.")
    optimizer = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
        lr=float(lr),
    )

    print("Setting up scheduler.")
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        save_strategy="steps",
        save_steps=save_steps,
        disable_tqdm=False,
        push_to_hub=False,
        logging_strategy="steps",
        logging_dir=os.path.join(
            os.getcwd(),
            logging_dir,
            model_name,
            datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
        ),
        logging_steps=1,
        logging_nan_inf_filter=True,
        gradient_accumulation_steps=accumulation_iter,
        output_dir=output_dir,
        max_steps=max_steps,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        eval_accumulation_steps=accumulation_iter,
        seed=seed,
        bf16=True,
        include_num_input_tokens_seen=True,
        save_safetensors=True,
        neftune_noise_alpha=0.1,
        split_batches=True,
        save_total_limit=1,
        use_cpu=device == "cpu",
    )

    class BlurredThoughtsSFTTrainer(Trainer):
        def compute_loss(
            self, 
            model, 
            inputs, 
            num_items_in_batch=None, 
            return_outputs=False
        ):
            outputs = self.model(
                inputs["input_ids"],
                labels=inputs["labels"],
                num_items_in_batch=num_items_in_batch,
                return_dict=True,
            )
            logits = outputs.logits
            loss = outputs.loss

            completions = tokenizer.batch_decode(
                logits.argmax(dim=-1), skip_special_tokens=False
            )

            rewards = format_reward_func(
                completions,
                tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False),
            )
            rewards = torch.tensor(rewards).mean()
            rewards = 1 - rewards

            loss = loss + bf_beta * rewards

            return (loss, outputs) if return_outputs else loss

    print("Training model.")
    trainer = BlurredThoughtsSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.with_format("torch"),
        eval_dataset=test_ds.with_format("torch"),
        data_collator=collator,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train(resume_from_checkpoint=trainer_checkpoint)

    print("Training complete.")
    trainer.save_model(os.path.join(output_dir, "final"))