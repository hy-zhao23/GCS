#!/bin/bash
#SBATCH --job-name=cpu_kav_mpi   # create a name for your job
#SBATCH --partition=general
#SBATCH --output=/dev/null
#SBATCH --error=/logs/cpu/%x_%j_%a.err
#SBATCH --qos=standard
#SBATCH --nodes=20  # node count
#SBATCH --ntasks-per-node=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=200G         # memory per cpu-core
#SBATCH --time=1-00:00:00          # total run time limit (HH:MM:SS)

# Create log directory if it doesn't exist
# mkdir -p logs; mkdir -p logs/cpu/ 

# Remove all .out and .err files in the log directory
# rm -f logs/cpu/*.err logs/cpu/*.out logs/main*.log

# set up fork safety for ucx
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_btl_openib_want_fork_support=1

eval "$(micromamba shell hook --shell bash)"
micromamba activate /project/md748/hpcadmin/hz54/env/steer
export PYTHONPATH=$PYTHONPATH:/mmfs1/project/md748/hpcadmin/hz54/env/steer/lib/python3.11/site-packages/

export PROJ=/mmfs1/project/md748/hz54/kav/KAV

# Run the Python script
# srun python script/bv.py # Bird-village comparison
srun python script/get_probing_classifier.py
# srun python script/get_sampled_classifier.py
# srun python script/evaluation.py
# srun python script/steer_emotion.py
# srun python script/score_steered_text.py

# Capture the exit status
exit_status=$?

# Cancel the job regardless of its status
scancel $SLURM_JOB_ID
# Exit with the original exit status
exit $exit_status