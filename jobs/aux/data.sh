#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=staging
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=jobs/job_outputs/copy_logs_%j.out

module load 2023
module load Anaconda3/2023.07-2

__conda_setup="$('/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /home/sorlova/anaconda3/envs/video

export PYTHONPATH=$PYTHONPATH:/home/sorlova/repos/AITHENA/NewStage/VideoMAE
cd /home/sorlova/repos/AITHENA/NewStage/VideoMAE

cp -r /home/sorlova/repos/AITHENA/NewStage/VideoMAE/logs/ft_after_pretrain/bl1k700 /gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/ 

echo "Done!"
