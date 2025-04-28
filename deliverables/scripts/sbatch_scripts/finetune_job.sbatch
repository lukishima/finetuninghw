#!/bin/bash
#SBATCH --job-name=finetune-resume
#SBATCH --output=/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/scripts/sbatch_scripts/logs/finetune_%j.out
#SBATCH --error=/gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/scripts/sbatch_scripts/logs/finetune_%j.err
#SBATCH --account=trn040
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --mail-user=lukishim@vols.utk.edu
#SBATCH --mail-type=ALL
#SBATCH --signal=B:USR1@300

# Trap USR1 signal (sent before time limit)
trap 'echo "Caught USR1 signal! Preparing to requeue..."; scontrol requeue $SLURM_JOB_ID' USR1


# Load GPU drivers
module load rocm/6.1.3

# Load Python/Conda
module load miniforge3

# Initialize Conda for non-interactive shell
source /sw/odo/miniforge3/23.11.0/etc/profile.d/conda.sh

# Activate your environment
conda activate ft-env

# Navigate to your scripts folder
cd /gpfs/wolf2/olcf/trn040/scratch/luki/FinetuningHW/scripts

# Run training
echo "Starting training (or resuming from checkpoint)..."
python train_model.py
