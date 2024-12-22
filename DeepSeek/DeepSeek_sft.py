import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import copy
import random
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq
import transformers
from tqdm import tqdm_notebook as tqdm
import json
from datetime import datetime
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from datasets import load_dataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def main():
    # load customized dataset
    train_dataset = load_dataset('json', data_files='dataset/train_data_deepseek-coder_sft.jsonl', split="train")
    # eval_dataset = load_dataset('json', data_files='', split="train")

    # load base model
    base_model = "./deepseek-coder-6.7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"  # up to you
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # fine-tuning
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model.train()  # put model back into training mode
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    batch_size = 4
    # per_device_train_batch_size = 2
    per_device_train_batch_size = 4
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    output_dir = "deepseek-sft"

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,
        # warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-5,
        fp16=True,
        # logging_strategy="epoch",
        logging_steps=100,
        optim="adamw_torch",
        # evaluation_strategy="epoch", # if val_set_size > 0 else "no",
        eval_steps=None,
        # save_strategy="epoch",
        save_steps=500,
        output_dir=output_dir,
        load_best_model_at_end=False,
        # group_by_length=True, # group sequences of roughly the same length together to speed up training
        gradient_checkpointing=True,
        report_to="none",  # if use_wandb else "none", wandb
        run_name=f"deepseek-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # if use_wandb else None,
        save_safetensors=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        max_seq_length=4096,
        args=training_args,
    )

    checkpoints_path = './checkpoints/deepseek/'
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # trainer
    T1 = time.perf_counter()
    model.config.use_cache = False
    trainer.train()
    T2 = time.perf_counter()
    print('Training Time Total: %s s' % (T2 - T1))

    trainer.model.save_pretrained(checkpoints_path, safe_serialifzation=False)


if __name__ == "__main__":
    main()
