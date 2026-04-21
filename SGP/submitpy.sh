#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=ventana
#SBATCH --time=4-00:00:00
#SBATCH --mem=0
#SBATCH --output=COR-lidar-eddies-%j.txt

# Optional lines. Yours will be different. Change as necessary or remove/comment out.
source /etc/profile
source ~/.bashrc
conda activate /home/scott.powell/anaconda3/envs/ventana

# Execute the code.
# time python sgp_eddysizes_noPBLH.py
time python sgp_eddysizes.py
