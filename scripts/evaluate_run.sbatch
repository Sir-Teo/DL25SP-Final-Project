#!/bin/bash
#SBATCH -p a100_short,radiology,a100_long,gpu4_medium,gpu4_long,gpu8_medium,gpu8_long # Adjust partitions as needed
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 # Evaluation might not need 16 CPUs
#SBATCH --mem=120GB      # Evaluation might not need 120GB
#SBATCH --time=48:00:00  # Adjust time limit per run evaluation
#SBATCH --job-name=eval_run # Will be overridden by submit script
#SBATCH --output=/gpfs/scratch/wz1492/DL25SP-Final-Project/logs/sbatch_eval_%j.log # Default log, might be overridden

## Base project and directories
PROJECT_BASE="/gpfs/scratch/wz1492/DL25SP-Final-Project"
LOG_DIR="$PROJECT_BASE/eval_outputs"
PLOTS_DIR="$PROJECT_BASE/plots"

# Determine RUN_PARENT: accept absolute or project-relative paths, or subdir under 'runs'
if [ -n "$1" ]; then
  if [ -d "$1" ]; then
    RUN_PARENT="$1"
  elif [ -d "$PROJECT_BASE/$1" ]; then
    RUN_PARENT="$PROJECT_BASE/$1"
  elif [ -d "$PROJECT_BASE/runs/$1" ]; then
    RUN_PARENT="$PROJECT_BASE/runs/$1"
  else
    echo "Warning: directory '$1' not found; defaulting to '$PROJECT_BASE/runs'"
    RUN_PARENT="$PROJECT_BASE/runs"
  fi
else
  if [ -d "$PROJECT_BASE/runs" ]; then
    RUN_PARENT="$PROJECT_BASE/runs"
  else
    echo "Error: No runs directory at '$PROJECT_BASE/runs'" >&2
    exit 1
  fi
fi

# Create log and plot directories
mkdir -p "$LOG_DIR" "$PLOTS_DIR"

# Path to plotting script
PLOT_SCRIPT_PATH="$(dirname "$0")/parse_and_plot.py"

# Activate conda environment
echo "Loading modules and activating conda environment..."
module purge # Start with a clean environment
module load gcc/8.1.0 # Load necessary modules
source ~/.bashrc
conda activate dino_wm # Make sure this environment has dependencies for main.py and matplotlib
if [ $? -ne 0 ]; then
    echo "Error activating conda environment dino_wm" >&2
    exit 1
fi
echo "Environment activated."

## Define a function to evaluate one run directory
process_run() {
  local SUBRUN="$1"
  local RUN_NAME=$(basename "$SUBRUN")
  local EVAL_RESULTS_FILE="$LOG_DIR/${RUN_NAME}_eval_summary.csv"
  local PLOT_OUTPUT_PATH="$PLOTS_DIR/${RUN_NAME}_eval_plot.png"

  echo "-----------------------------------------------------"
  echo "Starting evaluation for run: $SUBRUN"

  # Run-level cache
  if [ -f "$PLOT_OUTPUT_PATH" ]; then
    echo "Plot already exists: $PLOT_OUTPUT_PATH; skipping this run."
    return
  fi

  # Hyperparameters (default from main.py)
  ENCODER_NAME="dinov2_vits14"
  FEATURE_KEY="x_norm_patchtokens"
  NUM_HIST=16
  PREDICTOR_DEPTH=4
  PREDICTOR_HEADS=4
  PREDICTOR_DIM_HEAD=32
  PREDICTOR_MLP_DIM=512
  PREDICTOR_DROPOUT=0.0
  PREDICTOR_EMB_DROPOUT=0.0
  PREDICTOR_POOL="attn"

  # Find checkpoints (allow searching in subdirectories)
  local CHECKPOINTS=( $(find "$SUBRUN" -name 'model_epoch_*.pth' | sort -V) )
  if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found in $SUBRUN; skipping."
    return
  fi
  echo "Found ${#CHECKPOINTS[@]} checkpoints in $SUBRUN."

  # Initialize CSV if needed
  if [ ! -f "$EVAL_RESULTS_FILE" ]; then
    echo "epoch,probe_attr,loss" > "$EVAL_RESULTS_FILE"
  fi

  local SUCCESS_COUNT=0
  local FAIL_COUNT=0
  local SKIPPED_COUNT=0

  for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
    local FILENAME=$(basename "$CHECKPOINT_PATH")
    if [[ $FILENAME =~ ^model_epoch_([0-9]+)\.pth$ ]]; then
      local EPOCH=${BASH_REMATCH[1]}
    else
      echo "Warning: could not parse epoch from $FILENAME; skipping." >&2
      continue
    fi

    # Epoch-level cache
    if grep -q "^${EPOCH}," "$EVAL_RESULTS_FILE"; then
      echo "Epoch $EPOCH already evaluated; skipping.";
      SKIPPED_COUNT=$((SKIPPED_COUNT+1))
      SUCCESS_COUNT=$((SUCCESS_COUNT+1))
      continue
    fi

    # Evaluate the checkpoint
    local OUTPUT_LOG="$LOG_DIR/${RUN_NAME}_epoch_${EPOCH}.log"
    echo "Evaluating epoch $EPOCH: $CHECKPOINT_PATH"
    python main.py \
      --checkpoint "$CHECKPOINT_PATH" \
      --encoder-name "$ENCODER_NAME" \
      --feature-key "$FEATURE_KEY" \
      --num-hist "$NUM_HIST" \
      --predictor-depth "$PREDICTOR_DEPTH" \
      --predictor-heads "$PREDICTOR_HEADS" \
      --predictor-dim-head "$PREDICTOR_DIM_HEAD" \
      --predictor-mlp-dim "$PREDICTOR_MLP_DIM" \
      --predictor-dropout "$PREDICTOR_DROPOUT" \
      --predictor-emb-dropout "$PREDICTOR_EMB_DROPOUT" \
      --predictor-pool "$PREDICTOR_POOL" > "$OUTPUT_LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "Error in main.py for epoch $EPOCH; see $OUTPUT_LOG" >&2
      FAIL_COUNT=$((FAIL_COUNT+1))
      continue
    fi

    # Parse results
    local LINES_PARSED=0
    grep " loss: " "$OUTPUT_LOG" | while IFS=':' read -r key value; do
      local probe_attr=$(echo "$key" | sed 's/ loss$//' | xargs)
      local loss=$(echo "$value" | xargs)
      if [[ -n "$probe_attr" && -n "$loss" ]]; then
        echo "$EPOCH,$probe_attr,$loss" >> "$EVAL_RESULTS_FILE"
        LINES_PARSED=$((LINES_PARSED+1))
      fi
    done
    if [ $LINES_PARSED -gt 0 ]; then
      SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    else
      echo "Warning: no loss lines parsed for epoch $EPOCH." >&2
      FAIL_COUNT=$((FAIL_COUNT+1))
    fi
  done

  echo "Run $RUN_NAME summary: $SUCCESS_COUNT success, $FAIL_COUNT fail, $SKIPPED_COUNT skipped."
  if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Plotting results for $RUN_NAME"
    python "$PLOT_SCRIPT_PATH" \
      --csv "$EVAL_RESULTS_FILE" \
      --output "$PLOT_OUTPUT_PATH" \
      --title "Evaluation for $RUN_NAME"
    if [ $? -ne 0 ]; then
      echo "Error plotting for $RUN_NAME" >&2
    else
      echo "Plot saved to $PLOT_OUTPUT_PATH"
    fi
  else
    echo "Skipping plot for $RUN_NAME (no successful epochs)."
  fi
  echo "-----------------------------------------------------"
}

## Dispatch: single-run or multi-run
if find "$RUN_PARENT" -maxdepth 1 -name 'model_epoch_*.pth' -print -quit | grep -q .; then
  # Single run evaluation
  process_run "$RUN_PARENT"
else
  # Multi-run: loop over subdirectories of RUN_PARENT
  echo "Detected multiple run directories under $RUN_PARENT"
  echo "Iterating through items in $RUN_PARENT/* ..."
  for SUB in "$RUN_PARENT"/*; do
    echo "  Checking item: $SUB"
    if [ -d "$SUB" ]; then
      echo "    Item is a directory."
      # Check for checkpoints allowing subdirectories
      if find "$SUB" -name 'model_epoch_*.pth' -print -quit | grep -q .; then
        echo "    Directory (or its subdirectories) contains checkpoints. Calling process_run..."
        process_run "$SUB"
      else
        echo "    Directory does NOT contain checkpoints. Skipping."
      fi
    else
      echo "    Item is NOT a directory. Skipping."
    fi
  done
fi
exit 0

echo "Evaluation and plotting script for $RUN_NAME complete." 