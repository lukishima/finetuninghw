# UTK DSE 697 Finetuning Assignment

## Contents
- `conda_requirements.txt`
- `pip_requirements.txt`
- `data_download_instructions.txt`
- `data_samples/`
- `model_interactions/`
- `processed_data/`
- `scripts/`

---

## Environment Setup

### Conda Environment
```bash
conda create -n ft-env python=3.10
conda activate ft-env
pip install -r pip_requirements.txt
```
or
```bash
conda install --file conda_requirements.txt
```

---

## Dataset Download and Preparation

See `data_download_instructions.txt` for full manual steps.

### Summary:

1. Download `full_dataset.csv` from [RecipeNLG Website](https://recipenlg.cs.put.poznan.pl/).
2. Upload it to the Odo server:
   ```bash
   scp /path/to/full_dataset.csv luki@login1.odo.olcf.ornl.gov:/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/data/
   ```
3. Convert to HuggingFace format:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("csv", data_files="path/to/full_dataset.csv")
   dataset.save_to_disk("path/to/save/hf_dataset")
   ```
4. Processed datasets are available in `deliverables/processed_data/`.

---

## Scripts Overview

- `download_and_convert.py`: Prepares RecipeNLG for HuggingFace use.
- `download_model.py`: Downloads the base model (Llama-3.1-8B).
- `train_model.py`: Full finetuning script.
- `short_training.py`: Shortened finetuning.
- `sbatch_scripts/finetune_job.sbatch`: Full training SLURM job.
- `sbatch_scripts/short_finetune_job.batch`: Short training SLURM job.
- `sbatch_scripts/interactions.sbatch`: Automates model interaction testing.

---

## Model Interactions

### Models Evaluated

- Base model: `Llama-3.1-8B`
- Short fine-tuned: `Llama-3.1-8B_FT_short`
- Full fine-tuned: `Llama-3.1-8B_FT` (with checkpoint recovery)

### Prompt Styles

1. **Basic Prompt**: Simple instruction to write a recipe.
2. **Structured Prompt**: Adds clear formatting rules (ingredients, steps).
3. **Highly Structured Prompt**: Demands clean separation of ingredients and steps with strict formatting.
4. **Example-driven Prompt**: Provides a sample recipe before generating a new one.

Each model was evaluated across all 4 prompt styles.

---

## Output Organization

- Individual model outputs are stored in `deliverables/model_interactions/` under model-specific folders.
- Full combined interaction output saved as:
  ```
  deliverables/model_interactions/full_combined_output_<timestamp>.txt
  ```

---

## Rerunning Model Interactions

Run the interaction batch job:

```bash
cd scripts/sbatch_scripts
sbatch interactions.sbatch
```

This will re-run all interactions with the latest models and generate new combined outputs automatically.

---

## Additional Notes

- Full fine-tuned model (`Llama-3.1-8B_FT`) automatically resumes from latest checkpoint when interacting.
- Outputs are consistent but slight variations can occur due to model sampling randomness.
- Requires GPU node availability and Conda environment `ft-env` to be active.

---

## Final Deliverable Folder Layout

```plaintext
deliverables/
├── conda_requirements.txt
├── pip_requirements.txt
├── data_download_instructions.txt
├── data_samples/
│   ├── dataset_examples.txt
│   ├── full_dataset_sample.csv
│   └── processed_dataset_examples.txt
├── model_interactions/
│   ├── Llama-3.1-8B/
│   ├── Llama-3.1-8B_FT/
│   ├── Llama-3.1-8B_FT_short/
│   └── full_combined_output_<timestamp>.txt
├── processed_data/
│   └── processed_hf_dataset/
└── scripts/
    ├── download_and_convert.py
    ├── download_model.py
    ├── train_model.py
    └── sbatch_scripts/
```
---

All deliverables are organized and ready for submission.
