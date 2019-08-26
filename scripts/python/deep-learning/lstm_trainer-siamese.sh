#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --partition=sgpu
#SBATCH --qos=normal
#SBATCH --output=sample-%j.out
#SBATCH --job-name=test-job

module purge

source /curc/sw/anaconda3/2019.03/bin/activate
conda activate $PYTORCH
echo "== This is the scripting step! =="
sleep 30
python /projects/kadh5719/thesis_dl-fnirs/scripts/python/deep-learning/lstm_trainer-siamese-wm.py
echo "== End of Job =="
