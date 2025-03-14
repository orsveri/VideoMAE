import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from anaysis.make_plots.train_logs import DoTA_dirs, DADA2K_dirs, color_dict, marker_dict, markersize_dict, linestyle_dict, linestyle_marker_dict


def extract_threshold(col_name, prefix):
    """Extracts the numeric threshold from a column name with the given prefix.
       E.g., extract_threshold("mcc_0.25", "mcc_") returns 0.25."""
    return float(col_name.replace(prefix, ""))


def plot_ID_3x2_curves(file_paths_left, file_paths_right, file_labels):
    # Prepare lists to store curves for each metric for the left dataset
    mcc_curves_left = []
    acc_curves_left = []
    f1_curves_left = []
    
    # Loop through left files
    for path in file_paths_left:
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.set_index("group")
        row = df.loc["all_samples"]
        
        mcc_cols = sorted([col for col in row.index if col.startswith("mcc_")],
                          key=lambda x: extract_threshold(x, "mcc_"))
        acc_cols = sorted([col for col in row.index if col.startswith("acc_")],
                          key=lambda x: extract_threshold(x, "acc_"))
        f1_cols = sorted([col for col in row.index if col.startswith("f1_")],
                         key=lambda x: extract_threshold(x, "f1_"))
        
        mcc_thresholds = np.array([extract_threshold(col, "mcc_") for col in mcc_cols])
        mcc_values = row[mcc_cols].values.astype(float)
        
        acc_thresholds = np.array([extract_threshold(col, "acc_") for col in acc_cols])
        acc_values = row[acc_cols].values.astype(float)
        
        f1_thresholds = np.array([extract_threshold(col, "f1_") for col in f1_cols])
        f1_values = row[f1_cols].values.astype(float)
        
        mcc_curves_left.append((mcc_thresholds, mcc_values))
        acc_curves_left.append((acc_thresholds, acc_values))
        f1_curves_left.append((f1_thresholds, f1_values))
    
    # Prepare lists for the right dataset
    mcc_curves_right = []
    acc_curves_right = []
    f1_curves_right = []
    
    for path in file_paths_right:
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.set_index("group")
        row = df.loc["all_samples"]
        
        mcc_cols = sorted([col for col in row.index if col.startswith("mcc_")],
                          key=lambda x: extract_threshold(x, "mcc_"))
        acc_cols = sorted([col for col in row.index if col.startswith("acc_")],
                          key=lambda x: extract_threshold(x, "acc_"))
        f1_cols = sorted([col for col in row.index if col.startswith("f1_")],
                         key=lambda x: extract_threshold(x, "f1_"))
        
        mcc_thresholds = np.array([extract_threshold(col, "mcc_") for col in mcc_cols])
        mcc_values = row[mcc_cols].values.astype(float)
        
        acc_thresholds = np.array([extract_threshold(col, "acc_") for col in acc_cols])
        acc_values = row[acc_cols].values.astype(float)
        
        f1_thresholds = np.array([extract_threshold(col, "f1_") for col in f1_cols])
        f1_values = row[f1_cols].values.astype(float)
        
        mcc_curves_right.append((mcc_thresholds, mcc_values))
        acc_curves_right.append((acc_thresholds, acc_values))
        f1_curves_right.append((f1_thresholds, f1_values))
    
    # Define metric titles and store the corresponding curves for left/right in dictionaries.
    metric_titles = ["MCC", "Accuracy", "F1"]
    metrics_left = {
        "MCC": mcc_curves_left,
        "Accuracy": acc_curves_left,
        "F1": f1_curves_left
    }
    metrics_right = {
        "MCC": mcc_curves_right,
        "Accuracy": acc_curves_right,
        "F1": f1_curves_right
    }
    
    # Create a figure with 6 rows x 2 columns using GridSpec.
    # Row 0 will be for the legend. Rows 1-5 for the 5 metrics.
    fig = plt.figure(figsize=(9, 12))
    gs = gridspec.GridSpec(4, 2, height_ratios=[0.5, 1, 1, 1])
    gs.update(hspace=0.3)
    
    # Create the legend axis spanning both columns in the top row.
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    
    # We'll capture legend handles from one of the metric plots (e.g., from MCC on the left).
    legend_handles = None
    legend_labels = None
    
    # Create subplots for each metric and dataset.
    for i, metric in enumerate(metric_titles):
        # Left subplot for the current metric
        ax_left = fig.add_subplot(gs[i+1, 0])
        for (thresh, values), label in zip(metrics_left[metric], file_labels):
            c = color_dict.get(label, "black")       # default to black if not found
            m = linestyle_marker_dict.get(label, None)          # default marker if not found
            ms = markersize_dict.get(label, 6)       # default size if not found
            ls = linestyle_dict.get(label, "solid")
            ax_left.plot(
                thresh,
                values,
                label=label,
                color=c,
                marker=m,
                markersize=ms/7,
                markevery=8,
                #linewidth=1.5,  # optional line width
                linestyle=ls 
            )
        #ax_left.set_xlabel("Threshold")
        ax_left.set_ylabel(metric)
        ax_left.set_title(f"{metric} vs. Threshold (DoTA)", pad=5)
        ax_left.grid(True)
        
        # Right subplot for the current metric
        ax_right = fig.add_subplot(gs[i+1, 1])
        for (thresh, values), label in zip(metrics_right[metric], file_labels):
            c = color_dict.get(label, "black")       # default to black if not found
            m = linestyle_marker_dict.get(label, None)          # default marker if not found
            ms = markersize_dict.get(label, 6)       # default size if not found
            ls = linestyle_dict.get(label, "solid")
            ax_right.plot(
                thresh,
                values,
                label=label,
                color=c,
                marker=m,
                markersize=ms/7,
                markevery=8,
                #linewidth=1.5,  # optional line width
                linestyle=ls
            )
        #ax_right.set_xlabel("Threshold")
        ax_right.set_ylabel(metric)
        ax_right.set_title(f"{metric} vs. Threshold (DADA2K)", pad=5)
        ax_right.grid(True)
        
        # For the first metric (MCC), capture the legend handles/labels.
        if i == 0:
            legend_handles, legend_labels = ax_left.get_legend_handles_labels()
    
    # Create the legend in the top row.
    ax_legend.legend(legend_handles, legend_labels, loc="center", ncol=5, fontsize=14)
    
    # Save and show the figure.
    fig.savefig("anaysis/plots/plot_3x2_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

# Example usage:
# file_paths_left and file_paths_right should be lists of directories where "thresh_stats.csv" is found.
# file_labels should be a list of model names.
# plot_ID_5x2_curves(file_paths_left, file_paths_right, file_labels)



if __name__ == "__main__":
    # List of file paths and corresponding labels for each file.
    numbers_dota = (0, 2, 4, 6, 8, 12, 16, 20, 24)  #(0, 2, 4, 6, 8, 36, 40, 44, 48)  # (0, 2, 4, 6, 8, 12, 16, 20, 24)
    numbers_dada = (1, 3, 5, 7, 10, 14, 18, 22, 26) # (1, 3, 5, 7, 10, 38, 42, 46, 50)  # (1, 3, 5, 7, 10, 14, 18, 22, 26)
    labels = ("MOVAD", "VidNeXt", "ConvNeXt", "ResNet+NST", "R(2+1)D", "VideoMAE-S", "VideoMAE2-S", "InternVideo2-S", "ViViT-B")
    file_paths_dota = [DoTA_dirs[i] for i in numbers_dota]
    file_paths_dada = [DADA2K_dirs[i] for i in numbers_dada]
    #file_labels = [os.path.basename(os.path.dirname(fp)).split("_")[0] for fp in file_paths]  # labels for the curves
    #print(file_labels)

    plot_ID_3x2_curves(file_paths_left=file_paths_dota, file_paths_right=file_paths_dada, file_labels=labels)


