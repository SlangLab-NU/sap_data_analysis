srun --constraint=ib --partition=gpu --nodes=1 --gres=gpu:v100-sxm2 --mem=15G --cpus-per-task=8 --time=08:00:00 --pty /bin/bash

export PYTHONPATH=""
export VALLE_ROOT=/scratch/lewis.jor
export VALLE_REPO_ROOT=$VALLE_ROOT/VallE
export APPTAINERENV_PYTHONPATH="$VALLE_REPO_ROOT:/workspace/icefall:$PYTHONPATH"
export apptainer_image=$VALLE_REPO_ROOT/valle_container.sif

apptainer shell --nv \
        --bind /scratch/lewis.jor:/scratch/lewis.jor \
       	$apptainer_image

python calculate_sap_wer.py \
    --working-dir /scratch/lewis.jor \
    --max-speakers 1

scancel -u $USER