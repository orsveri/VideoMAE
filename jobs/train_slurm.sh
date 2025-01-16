#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:20:00
#SBATCH --output=jobs/job_outputs/check_env.out

module load 2022

__conda_setup="$('/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /home/sorlova/anaconda3/envs/video

cd /home/sorlova/repos/AITHENA/NewStage/VideoMAE

srun python data_tools/ds_ydid/gen_dets_yolo.py