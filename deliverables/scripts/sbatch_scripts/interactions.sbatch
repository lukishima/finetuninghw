#!/bin/bash
#SBATCH --job-name=run_interactions
#SBATCH --output=/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/scripts/sbatch_scripts/logs/interact_%j.out
#SBATCH --error=/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/scripts/sbatch_scripts/logs/interact_%j.err
#SBATCH --account=trn040
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

# Load GPU drivers
module load rocm/6.1.3

# Load Python/Conda
module load miniforge3

# Initialize Conda for non-interactive shell
source /sw/odo/miniforge3/23.11.0/etc/profile.d/conda.sh

# Activate your environment
conda activate ft-env

# Combined output file
RUN_TIME=$(date +"%Y%m%d_%H%M%S")
COMBINED_OUTPUT="/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/deliverables/model_interactions/full_combined_output_$RUN_TIME.txt"
echo "" > "$COMBINED_OUTPUT"

# Base model - basic prompt
echo -e "\n===== Base Model (basic prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_base.py >> "$COMBINED_OUTPUT"

# Base model - structured prompt
echo -e "\n===== Base Model (structured prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_base_mod.py >> "$COMBINED_OUTPUT"

# Base model - highly structured prompt
echo -e "\n===== Base Model (highly structured prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_base_mod2.py >> "$COMBINED_OUTPUT"

# Base model - example-driven prompt
echo -e "\n===== Base Model (example-driven prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_base_mod3.py >> "$COMBINED_OUTPUT"

# Short fine-tuned model - basic prompt
echo -e "\n===== Short Fine-tuned Model (basic prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_short.py >> "$COMBINED_OUTPUT"

# Short fine-tuned model - structured prompt
echo -e "\n===== Short Fine-tuned Model (structured prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_short_mod.py >> "$COMBINED_OUTPUT"

# Short fine-tuned model - highly structured prompt
echo -e "\n===== Short Fine-tuned Model (highly structured prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_short_mod2.py >> "$COMBINED_OUTPUT"

# Short fine-tuned model - example-driven prompt
echo -e "\n===== Short Fine-tuned Model (example-driven prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_short_mod3.py >> "$COMBINED_OUTPUT"

# Long fine-tuned model (full training - basic prompt)
echo -e "\n===== Full Fine-tuned Model (latest checkpoint) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft.py >> "$COMBINED_OUTPUT"

# Long fine-tuned model (structured prompt)
echo -e "\n===== Full Fine-tuned Model (structured prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_mod.py >> "$COMBINED_OUTPUT"

# Long fine-tuned model (highly structured prompt)
echo -e "\n===== Full Fine-tuned Model (highly structured prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_mod2.py >> "$COMBINED_OUTPUT"

# Long fine-tuned model (example-driven prompt)
echo -e "\n===== Full Fine-tuned Model (example-driven prompt) =====\n" >> "$COMBINED_OUTPUT"
python scripts/interact_ft_mod3.py >> "$COMBINED_OUTPUT"

# Done
echo -e "\n✅ All model interactions finished. Output saved to: $COMBINED_OUTPUT\n"
