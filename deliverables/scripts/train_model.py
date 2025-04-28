import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from datetime import datetime
import glob

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Create a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")  # for filenames and folder names
formatted_timestamp = now.strftime("%-m/%-d/%Y %H:%M:%S")  # for printing/logging nicely

# Model and tokenizer names
base_model_name = "/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/models/Llama-3.1-8B"
new_model_name = "Llama-3.1-8B_FT"

# New folders for checkpoints
# Base output directory
base_output_dir = f"/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/models/{new_model_name}"



# Check for latest existing checkpoint if it exists
# Locate latest run folder if resuming
existing_runs = sorted(glob.glob(f"{base_output_dir}/run_*"))
resume_checkpoint = None
latest_run = None

# Look for the most recent run that has a checkpoint
for run in reversed(existing_runs):  # Check most recent first
    checkpoints = sorted(glob.glob(os.path.join(run, "checkpoint-*")))
    if checkpoints:
        latest_run = run
        resume_checkpoint = checkpoints[-1]
        print(f"Resuming from: {resume_checkpoint} at {formatted_timestamp}")
        break

if not resume_checkpoint:
    print(f"No previous checkpoints found. Starting fresh at {formatted_timestamp}")

# === NOW Create a New Run Folder (even if resuming, to isolate logs) ===
output_dir = f"{base_output_dir}/run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, local_files_only=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    local_files_only=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Data set
dataset = load_from_disk("/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/deliverables/processed_data/processed_hf_dataset")
training_data = dataset

# Training Params
train_params = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    #optim="paged_adamw_32bit",
    save_steps=8000,
    logging_steps=4000,
    save_strategy="steps",
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
)

from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    processing_class=llama_tokenizer,
    args=train_params
)

# Training
# Get the current time only
start_time = datetime.now()
formatted_start_time = start_time.strftime("%H:%M:%S")
print(f"Training started at: {formatted_start_time}")
if resume_checkpoint:
    with open(os.path.join(output_dir, "resume_info.txt"), "w") as f:
        f.write(f"Resumed from checkpoint: {resume_checkpoint} at {formatted_start_time}\n")
fine_tuning.train(resume_from_checkpoint=resume_checkpoint)


# Save model
end_time = datetime.now()
formatted_end_time = end_time.strftime("%H:%M:%S")
print(f"Training ended and saving file at: {formatted_end_time}")
model.save_pretrained(output_dir)
print(f"Model saved to: {output_dir}")

