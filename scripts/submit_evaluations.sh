#!/bin/bash

# This script submits an evaluation job for each run directory found under the specified base directory.

PROJECT_BASE="/gpfs/scratch/wz1492/DL25SP-Final-Project"
# Allow overriding the base directory of runs via the first script argument
if [ -n "$1" ]; then
  RUNS_BASE_DIR="$PROJECT_BASE/$1"
  echo "Using custom runs directory: $RUNS_BASE_DIR"
else
  RUNS_BASE_DIR="$PROJECT_BASE/runs"
  echo "Using default runs directory: $RUNS_BASE_DIR"
fi
SCRIPTS_DIR="$PROJECT_BASE/scripts"
SBATCH_SCRIPT="$SCRIPTS_DIR/evaluate_run.sbatch"
LOG_BASE_DIR="$PROJECT_BASE/logs" # Base for SBATCH .out/.err logs

if [ ! -d "$RUNS_BASE_DIR" ]; then
  echo "Error: Base runs directory not found: $RUNS_BASE_DIR" >&2
  exit 1
fi

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "Error: SBATCH script not found: $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_BASE_DIR" # Ensure SBATCH log directory exists

echo "Searching for run directories containing model checkpoints in $RUNS_BASE_DIR..."

# Find directories directly under RUNS_BASE_DIR.
# Adjust depth or criteria if structure is different (e.g., use -maxdepth 2 for runs/experiment/run_name)
find "$RUNS_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r RUN_DIR; do
    # Check if the directory actually contains checkpoints before submitting
    if find "$RUN_DIR" -name 'model_epoch_*.pth' -print -quit | grep -q '.'; then
        RUN_NAME=$(basename "$RUN_DIR")
        # Sanitize RUN_NAME for use in job name and log files (replace non-alphanumeric with underscore)
        SAFE_RUN_NAME=$(echo "$RUN_NAME" | sed 's/[^a-zA-Z0-9]/-/g')
        JOB_NAME="eval_${SAFE_RUN_NAME}"
        # Define separate SBATCH output and error logs per job
        SBATCH_LOG_OUT="$LOG_BASE_DIR/sbatch_eval_${SAFE_RUN_NAME}_%j.out"
        SBATCH_LOG_ERR="$LOG_BASE_DIR/sbatch_eval_${SAFE_RUN_NAME}_%j.err"

        echo "-----------------------------------"
        echo "Found run: $RUN_DIR"
        echo "Submitting job: $JOB_NAME"
        echo "  SBATCH script: $SBATCH_SCRIPT"
        echo "  Run directory argument: $RUN_DIR"
        echo "  SBATCH stdout log: $SBATCH_LOG_OUT"
        echo "  SBATCH stderr log: $SBATCH_LOG_ERR"

        sbatch \
          --job-name="$JOB_NAME" \
          --output="$SBATCH_LOG_OUT" \
          --error="$SBATCH_LOG_ERR" \
          "$SBATCH_SCRIPT" "$RUN_DIR"

        if [ $? -ne 0 ]; then
            echo "Error submitting job for $RUN_DIR" >&2
        else
            echo "Job submitted for $RUN_DIR."
        fi
    else
        echo "-----------------------------------"
        echo "Skipping directory (no checkpoints found): $RUN_DIR"
    fi
done

echo "-----------------------------------"
echo "Finished submitting evaluation jobs."
echo "Monitor jobs using: squeue -u $USER"
echo "Check SBATCH output/error logs in: $LOG_BASE_DIR"
echo "Check evaluation output logs (from main.py) in: $LOG_BASE_DIR/eval_outputs"
echo "Check generated plots in: $PROJECT_BASE/plots" 