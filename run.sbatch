#!/bin/bash
#SBATCH --account=nvr_lpr_nvgptvision
#SBATCH --partition=batch_singlenode,grizzly,polar,polar3,polar4
#SBATCH --time=04:00:00                  # Adjust time limit as needed
#SBATCH --overcommit                     # allows more than one process per CPU
#SBATCH --cpus-per-task=16               # number of cores
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --job-name=caption_agent
#SBATCH --signal=SIGUSR1@300             # Send SIGUSR1 300 seconds before time limit

# Usage:
#
# sbatch --gres=gpu:1 ./scripts/caption.sbatch \
#     --tag=gemma3 \  # Docker tag. Default is "latest"
#     --sweep_id=[SWEEP_ID] \
#     --server_url=http://cchoy-dt.nvidia.com:8081

# Set default TAG to "latest" if not provided
TAG="uv"
IMAGE_URL="gitlab-master.nvidia.com/3dmmllm/datagen:${TAG}"

NODE=$(hostname -s)
USER=$(whoami)

# Project specific variables
ACCOUNT=nvr_lpr_nvgptvision
LUSTRE_NVR=/lustre/fsw/portfolios/nvr
LUSTRE_HOME="${LUSTRE_NVR}/users/${USER}"
PROJECT_ROOT="${LUSTRE_NVR}/projects/${ACCOUNT}"

# HF_HOME for credentials and other Hugging Face data.
# HF_HUB_CACHE for caching repositories from the Hub.
# HF_ASSETS_CACHE for caching other assets.
# Setting environment variables
HF_HUB_CACHE="${PROJECT_ROOT}/huggingface_cache"

# Image and code root
CODE_ROOT="${LUSTRE_HOME}/projects/vipe"
DATA_ROOT="${PROJECT_ROOT}/datasets"
HF_HUB_CACHE="${CODE_ROOT}/huggingface_cache"

# Process arguments
PYTHON_ARGS=() # Store python args
while [[ $# -gt 0 ]]; do
    case $1 in
    --tag=*)
        TAG="${1#*=}" # Extract value after =
        IMAGE_URL="gitlab-master.nvidia.com/3dmmllm/datagen:${TAG}"
        shift 1 # Shift only once as flag and value are one arg
        ;;
    -t | --tag)
        TAG="$2"
        IMAGE_URL="gitlab-master.nvidia.com/3dmmllm/datagen:${TAG}"
        shift 2
        ;;
    -i | --image_url)
        IMAGE_URL="$2"
        shift 2
        ;;
    -c | --code_root)
        CODE_ROOT="$2"
        shift 2
        ;;
    -d | --data_root)
        DATA_ROOT="$2"
        shift 2
        ;;
    *)
        # Assume this is an argument for the python script
        PYTHON_ARGS+=("$1")
        shift 1 # Consume only the flag/argument
        ;;
    esac
done

# Remaining arguments are treated as TRAINING_ARGS
TRAINING_ARGS=("${PYTHON_ARGS[@]}")

# Get the number of gpus from scontrol and strip all the white spaces and new lines
# example output
# >    TresPerNode=gres:gpu:2
GPU_COUNT=$(scontrol show job $SLURM_JOBID | grep gres | awk -F: '{print $3}' | tr -d '[:space:]')

# Print tunneling instructions
echo -e "
Running a GPU job on

    Node: ${NODE}
    JOB_ID ${SLURM_JOB_ID}
    Date: $(TZ=America/Los_Angeles date)
    User: ${USER}
    Image: ${IMAGE_URL}
    GPU_COUNT: ${GPU_COUNT}
    TRAINING_ARGS: ${TRAINING_ARGS[@]}
"

# Cache docker file
# Check if the cache_image.sh exists
if [ -f $LUSTRE_HOME/sbatch/cache_image.sh ]; then
    # Cache the image
    source $LUSTRE_HOME/sbatch/cache_image.sh

    SQSH_CACHE_DIR=${PROJECT_ROOT}/enroot-cache
    IMAGE_CACHE_FILE=$(cache_image $IMAGE_URL $ACCOUNT $SQSH_CACHE_DIR)
else
    echo "$LUSTRE_HOME/sbatch/cache_image.sh does not exist. Using ${IMAGE_URL} without caching."
    IMAGE_CACHE_FILE=${IMAGE_URL}
fi
echo "IMAGE_CACHE_FILE=${IMAGE_CACHE_FILE}"

# Your training script here
CMD="
TZ=America/Los_Angeles date;
cd /workspace;
conda deactivate;
nvidia-smi;
export UV_PYTHON=/python/.venv/bin/python;
export HF_HUB_CACHE=${HF_HUB_CACHE};
export CACHE_DIR=${HF_HUB_CACHE};
export REQUESTS_TIMEOUT=300;
pip install -r envs/requirements.my.txt;
pip install --no-build-isolation -e .;
python run.py \
    pipeline=default \
    streams=dl3dv_stream \
    streams.base_path=/datasets/dl3dv-10k/ \
    streams.frame_skip=2 \
    pipeline.output.artifacts_format=tar \
    pipeline.output.z_up=True \
    pipeline.output.path=/datasets/mosaic3d++/source_datasets/dl3dv  \
    ${TRAINING_ARGS[@]}
"

set -x
# Pass CMD with double quotes
srun \
    --container-image=$IMAGE_CACHE_FILE \
    --container-mounts="$HOME:/root,/lustre:/lustre,${CODE_ROOT}:/workspace,${DATA_ROOT}:/datasets" \
    bash -c "$CMD"

# Get the status from $OUTPUT_FOLDER/status.txt
STATUS_FILE="logs/${SLURM_JOB_ID}/status.txt"
if [ -f $STATUS_FILE ]; then
    STATUS=$(cat $STATUS_FILE)
    echo "Status: $STATUS"
    if [ "${STATUS}" == "STOPPED" ] || [ "${STATUS}" == "RUNNING" ]; then
        echo "Resubmitting the job with script: ${SCRIPT_PATH} ${@}"
        scontrol requeue $SLURM_JOB_ID
    fi
else
    echo "Status file not found: $STATUS_FILE"
fi
