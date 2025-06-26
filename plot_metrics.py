import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_metrics(metrics_file, save_path=None):
    # Read metrics from CSV
    metrics = pd.read_csv(metrics_file, header=None)
    
    # Extract metrics
    epochs = range(1, len(metrics) + 1)
    train_wer = metrics[3]  # Column 3 contains training WER
    val_wer = metrics[4]    # Column 4 contains validation WER
    
    # Create a single figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and validation WER on the same plot
    ax.plot(epochs, train_wer, 'b-', label='Training WER')
    ax.plot(epochs, val_wer, 'r-', label='Validation WER')
    ax.set_title('Training and Validation WER')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('WER')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Get all results directories
    results_dir = 'results'
    model_dirs = glob.glob(os.path.join(results_dir, '*'))
    
    for model_dir in model_dirs:
        metrics_file = os.path.join(model_dir, 'Metrics.csv')
        if os.path.exists(metrics_file):
            print(f"Plotting metrics for {os.path.basename(model_dir)}")
            save_path = os.path.join(model_dir, 'training_curves.png')
            plot_metrics(metrics_file, save_path)

if __name__ == "__main__":
    main()
