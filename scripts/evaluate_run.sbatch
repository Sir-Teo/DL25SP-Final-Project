#!/bin/bash
#SBATCH -p a100_short,radiology,a100_long,gpu4_medium,gpu4_long,gpu8_medium,gpu8_long # Adjust partitions as needed
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 # Evaluation might not need 16 CPUs
#SBATCH --mem=120GB      # Evaluation might not need 120GB
#SBATCH --time=24:00:00  # Adjust time limit per run evaluation
#SBATCH --job-name=eval_run # Will be overridden by submit script
#SBATCH --output=/gpfs/scratch/wz1492/DL25SP-Final-Project/logs/sbatch_eval_%j.log # Default log, might be overridden

RUN_DIR=$1 # This should be passed by the submit script
PROJECT_BASE="/gpfs/scratch/wz1492/DL25SP-Final-Project"
LOG_DIR="$PROJECT_BASE/eval_outputs" # Specific dir for main.py outputs
PLOTS_DIR="$PROJECT_BASE/plots"

if [ -z "$RUN_DIR" ]; then
  echo "Usage: sbatch evaluate_run.sbatch <path_to_run_directory>"
  exit 1
fi

RUN_NAME=$(basename "$RUN_DIR")
mkdir -p "$LOG_DIR"
mkdir -p "$PLOTS_DIR"

PLOT_SCRIPT_PATH="$(dirname "$0")/parse_and_plot.py" # Assumes plot script is in the same dir
PLOT_OUTPUT_PATH="$PLOTS_DIR/${RUN_NAME}_eval_plot.png"
EVAL_RESULTS_FILE="$LOG_DIR/${RUN_NAME}_eval_summary.csv"

# --- Run-level Cache Check ---
if [ -f "$PLOT_OUTPUT_PATH" ]; then
  echo "Plot $PLOT_OUTPUT_PATH already exists. Assuming run $RUN_NAME is fully evaluated. Exiting."
  exit 0
fi
# --- End Run-level Cache Check ---


if [ ! -d "$RUN_DIR" ]; then
  echo "Error: Run directory $RUN_DIR not found." >&2
  exit 1
fi

echo "Evaluating run: $RUN_DIR"
echo "Run name: $RUN_NAME"
echo "Output logs will be in: $LOG_DIR"
echo "Plots will be in: $PLOTS_DIR"
echo "Summary CSV: $EVAL_RESULTS_FILE"
echo "Final Plot: $PLOT_OUTPUT_PATH"


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

# --- NOTE: Using default hyperparameters from main.py's argparse ---
# --- This might be incorrect if runs used different settings. ---
# --- Consider modifying this script to load args from a run-specific file (e.g., args.json) ---
ENCODER_NAME="dinov2_vits14"
FEATURE_KEY="x_norm_patchtokens"
NUM_HIST=16
PREDICTOR_DEPTH=4
PREDICTOR_HEADS=4
PREDICTOR_DIM_HEAD=32
PREDICTOR_MLP_DIM=512
PREDICTOR_DROPOUT=0.0
PREDICTOR_EMB_DROPOUT=0.0
PREDICTOR_POOL="attn" # Based on train.sbatch and main.py default

echo "Finding checkpoints in $RUN_DIR..."
CHECKPOINTS=($(find "$RUN_DIR" -name 'model_epoch_*.pth' | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "Error: No checkpoints found in $RUN_DIR matching model_epoch_*.pth" >&2
  exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints."

# Initialize CSV only if it doesn't exist
if [ ! -f "$EVAL_RESULTS_FILE" ]; then
  echo "Initializing results file: $EVAL_RESULTS_FILE"
  echo "epoch,probe_attr,loss" > "$EVAL_RESULTS_FILE"
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIPPED_COUNT=0

for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
  FILENAME=$(basename "$CHECKPOINT_PATH")
  # Extract epoch number more robustly
  if [[ "$FILENAME" =~ model_epoch_([0-9]+)\\.pth ]]; then
      EPOCH=${BASH_REMATCH[1]}
  else
      echo "Warning: Could not extract epoch number from $FILENAME. Skipping check for this file." >&2
      continue
  fi

  # --- Epoch-level Cache Check ---
  if grep -q "^${EPOCH}," "$EVAL_RESULTS_FILE"; then
      echo "Epoch $EPOCH already found in $EVAL_RESULTS_FILE. Skipping evaluation."
      SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
      # We need to count successes based on what's *already* in the CSV for plotting later
      # Check if *any* line for this epoch exists, assume it was a success previously for simplicity
      if grep -q "^${EPOCH}," "$EVAL_RESULTS_FILE"; then
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      fi
      continue # Skip to the next checkpoint
  fi
  # --- End Epoch-level Cache Check ---


  OUTPUT_LOG="$LOG_DIR/${RUN_NAME}_epoch_${EPOCH}.log"
  echo "-----------------------------------------------------"
  echo "Processing Epoch $EPOCH: $FILENAME"
  echo "Checkpoint: $CHECKPOINT_PATH"
  echo "Output log: $OUTPUT_LOG"

  # Run main.py for evaluation
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

  # Check if python command was successful
  if [ $? -ne 0 ]; then
    echo "Error running main.py for epoch $EPOCH. Check log $OUTPUT_LOG" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
    # Continue to next checkpoint even if one fails
  else
    echo "Finished evaluation for Epoch $EPOCH."
    # Parse the output log for losses and append to CSV
    # Example output format: "normal loss: 0.123", "wall loss: 0.456"
    LINES_PARSED=0
    grep " loss: " "$OUTPUT_LOG" | while IFS=':' read -r key value; do
      # Clean up key and value
      probe_attr=$(echo "$key" | sed 's/ loss$//' | xargs)
      loss=$(echo "$value" | xargs)
      # Validate extracted values (simple check)
      if [[ -n "$probe_attr" && -n "$loss" ]]; then
          # Append to CSV
          echo "$EPOCH,$probe_attr,$loss" >> "$EVAL_RESULTS_FILE"
          echo "  Parsed and Appended: Epoch=$EPOCH, Attribute='$probe_attr', Loss=$loss"
          LINES_PARSED=$((LINES_PARSED + 1))
      else
          echo "  Warning: Failed to parse line: $key:$value" >&2
      fi
    done
    # Check if any lines were actually parsed and added for this epoch
    if [ $LINES_PARSED -gt 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "Error: main.py completed but no loss lines found/parsed in $OUTPUT_LOG" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
  fi
  echo "-----------------------------------------------------"

done

echo "Finished processing checkpoints for $RUN_NAME."
echo "Summary: $SUCCESS_COUNT successful, $FAIL_COUNT failed, $SKIPPED_COUNT skipped (cached)."
echo "Consolidated results are in $EVAL_RESULTS_FILE"

# Check if any evaluations were successful (either newly run or previously cached) before plotting
# Note: SUCCESS_COUNT now includes previously successful runs inferred from cache skips
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Generating plot..."

    python "$PLOT_SCRIPT_PATH" --csv "$EVAL_RESULTS_FILE" --output "$PLOT_OUTPUT_PATH" --title "Evaluation for $RUN_NAME"

    if [ $? -ne 0 ]; then
        echo "Error generating plot. Check python script $PLOT_SCRIPT_PATH and input $EVAL_RESULTS_FILE" >&2
        # Do not mark as complete if plotting fails
    else
        echo "Plot saved to $PLOT_OUTPUT_PATH"
        echo "Marking run as complete (plot generated)."
    fi
else
    echo "Skipping plot generation as no evaluations succeeded or were previously cached."
fi

echo "Evaluation and plotting script for $RUN_NAME complete." 