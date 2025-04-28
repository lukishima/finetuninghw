from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Local path to the cloned LLaMA 3.1 8B repo - this is the base model
model_path = "/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/models/Llama-3.1-8B"  

# # Local path to the cloned LLaMA 3.1 8B Instruct repo - Instruct is finetuned version of the base model
# model_path = "/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/models/Llama-3.1-8B-Instruct"  

print("Loading LLaMA 3.1 8B Model from local path ...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Loading tokenizer from local path ...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_safetensors=True
)

print("Model and tokenizer successfully loaded from local clone.")