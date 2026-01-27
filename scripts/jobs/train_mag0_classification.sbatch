#!/bin/bash
#SBATCH --job-name=mag0_class
#SBATCH --output=jobs/slurm_logs/mag0_class_%j.out
#SBATCH --error=jobs/slurm_logs/mag0_class_%j.err
#SBATCH -p arguelles_delgado_h100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --mem=64000

# Environment setup
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate mlenv

# Navigate to neptune directory
cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/ML/sources/neptune

# Install neptune and ml_common if not already installed
# pip install -e . --quiet
# pip install -e ml_common --quiet

# Run training
python scripts/run.py -c scripts/configs/mag0_classification.cfg

echo "Training complete!"
