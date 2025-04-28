from datasets import load_dataset

# Load dataset from local folder
dataset = load_dataset("csv", data_files="data/full_dataset.csv")

# Save dataset in HuggingFace Format
dataset.save_to_disk("data/hf_dataset")