import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_evaluation_results(csv_path, output_path, title):
    """
    Reads evaluation results from a CSV file and generates a plot.

    Args:
        csv_path (str): Path to the input CSV file (epoch, probe_attr, loss).
        output_path (str): Path to save the output plot image.
        title (str): Title for the plot.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    if df.empty:
        print(f"No data found in {csv_path}. Skipping plot generation.")
        return

    # Ensure epoch is integer for plotting and sorting
    df['epoch'] = df['epoch'].astype(int)
    df = df.sort_values('epoch') # Sort by epoch before plotting

    plt.figure(figsize=(10, 6))

    # Plot loss for each probe_attr
    for attr in sorted(df['probe_attr'].unique()): # Sort attributes for consistent legend order
        subset = df[df['probe_attr'] == attr]
        plt.plot(subset['epoch'], subset['loss'], marker='o', linestyle='-', label=f'{attr} loss')

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(df['epoch'].unique()) # Show ticks for each epoch evaluated

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close() # Close the figure to free memory
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results from CSV.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file with columns 'epoch', 'probe_attr', 'loss'.")
    parser.add_argument("--output", required=True, help="Path to save the output plot PNG file.")
    parser.add_argument("--title", default="Evaluation Results", help="Title for the plot.")

    args = parser.parse_args()

    plot_evaluation_results(args.csv, args.output, args.title) 