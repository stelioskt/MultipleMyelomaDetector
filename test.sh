#!/bin/bash
#SBATCH --output=slurm-%j.out                         # Output file with job ID
#SBATCH --error=slurm-%j.err                          # Error file with job ID
#SBATCH --gres=gpu:RTX4080:1                          # Request GPU "generic resource"
#SBATCH --job-name=gnn_seg                            # Job name

 
# Activate the environment
source activate venv

 
# Submit the job
main_path="/home/gbotis/stelios/MultipleMyelomaDetector"
 
cd $main_path

python clustering.py