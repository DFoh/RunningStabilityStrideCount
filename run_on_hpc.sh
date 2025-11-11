#!/bin/bash

#SBATCH --job-name=lds_analysis_df
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --output=%j.log
#SBATCH --partition=workq

start_ts=$(date +%s)

echo "hpc job running as $USER on $HOSTNAME"
date
echo SLURM_JOB_ID $SLURM_JOB_ID

source /etc/profile.d/conda.sh


PATH_CONDA_ENV="/data01/FORDATA/HAMBURG/IIES/AG-Hollander/DF/Projects/RunningStabilityStrideCount/RunningStabilityStrideCount/env"
conda activate "$PATH_CONDA_ENV"

PATH_SCRIPT="/data01/FORDATA/HAMBURG/IIES/AG-Hollander/DF/Projects/RunningStabilityStrideCount/RunningStabilityStrideCount"
cd "$PATH_SCRIPT"

python3 analysis.py

conda deactivate

end_ts=$(date +%s)
runtime=$(( end_ts - start_ts ))

echo "runtime seconds: $runtime"
date