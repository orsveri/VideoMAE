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
    E.g., extract_threshold("r_0.25", "r_") returns 0.25."""
    return float(col_name.replace(prefix, ""))


def plot_group_thresholded_metrics(file_paths_top, file_paths_bottom, file_labels):
    """
    For two lists of files (top and bottom), read each "thresh_stats.csv" file,
    then extract curves for specific groups:
      - For groups "ego" and "no_ego": use recall curves (columns starting with "r_").
      - For groups "night" and "day": extract three curves each for MCC ("mcc_"),
        Accuracy ("acc_") and F1 ("f1_").
    
    The final figure has 9 rows and 2 columns (using GridSpec):
      Row 0: Legend.
      Row 1: Ego Recall.
      Row 2: No Ego Recall.
      Row 3: Night MCC.
      Row 4: Night Accuracy.
      Row 5: Night F1.
      Row 6: Day MCC.
      Row 7: Day Accuracy.
      Row 8: Day F1.
    
    Each subplot shows the curve (recall or other metric vs threshold) for all methods,
    with the left column from the top file list and the right column from the bottom file list.
    """
    # Define the keys we need:
    # For recall groups, the metric key is "recall"
    recall_groups = ["ego", "no_ego"]
    # For night/day, we want three metrics per group.
    other_groups = ["night", "day"]
    other_metrics = ["mcc", "acc", "f1"]
    
    # We'll build dictionaries for top and bottom.
    # Keys: (group, metric) where metric is "recall" for ego/no_ego,
    # or one of "mcc","acc","f1" for night/day.
    curves_top = {}
    curves_bottom = {}
    # Initialize keys:
    for g in recall_groups:
        curves_top[(g, "recall")] = []
        curves_bottom[(g, "recall")] = []
    for g in other_groups:
        for m in other_metrics:
            curves_top[(g, m)] = []
            curves_bottom[(g, m)] = []
    
    # Process top files
    for path, label in zip(file_paths_top, file_labels):
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.set_index("group")
        # For each required group, if present, extract the corresponding columns.
        # For recall groups: look for columns starting with "r_"
        for g in recall_groups:
            if g in df.index:
                recall_cols = sorted([col for col in df.columns if col.startswith("r_")],
                                      key=lambda c: extract_threshold(c, "r_"))
                thresholds = np.array([extract_threshold(c, "r_") for c in recall_cols])
                values = df.loc[g, recall_cols].values.astype(float)
                curves_top[(g, "recall")].append((thresholds, values, label))
        # For other groups ("night" and "day"), for each metric.
        for g in other_groups:
            if g in df.index:
                for m in other_metrics:
                    prefix = m + "_"  # e.g., "mcc_", "acc_", "f1_"
                    metric_cols = sorted([col for col in df.columns if col.startswith(prefix)],
                                         key=lambda c: extract_threshold(c, prefix))
                    thresholds = np.array([extract_threshold(c, prefix) for c in metric_cols])
                    values = df.loc[g, metric_cols].values.astype(float)
                    curves_top[(g, m)].append((thresholds, values, label))
    
    # Process bottom files similarly
    for path, label in zip(file_paths_bottom, file_labels):
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.set_index("group")
        for g in recall_groups:
            if g in df.index:
                recall_cols = sorted([col for col in df.columns if col.startswith("r_")],
                                      key=lambda c: extract_threshold(c, "r_"))
                thresholds = np.array([extract_threshold(c, "r_") for c in recall_cols])
                values = df.loc[g, recall_cols].values.astype(float)
                curves_bottom[(g, "recall")].append((thresholds, values, label))
        for g in other_groups:
            if g in df.index:
                for m in other_metrics:
                    prefix = m + "_"
                    metric_cols = sorted([col for col in df.columns if col.startswith(prefix)],
                                         key=lambda c: extract_threshold(c, prefix))
                    thresholds = np.array([extract_threshold(c, prefix) for c in metric_cols])
                    values = df.loc[g, metric_cols].values.astype(float)
                    curves_bottom[(g, m)].append((thresholds, values, label))
    
    # Define the order and titles for our 8 rows (legend row will be row 0).
    # Rows 1-2: recall groups; rows 3-5: night metrics; rows 6-8: day metrics.
    row_keys = [
        ("ego", "recall"),
        ("no_ego", "recall"),
        ("night", "mcc"),
        ("night", "acc"),
        ("night", "f1"),
        ("day", "mcc"),
        ("day", "acc"),
        ("day", "f1")
    ]
    row_titles = [
        "Ego Recall",
        "No Ego Recall",
        "Night MCC",
        "Night Accuracy",
        "Night F1",
        "Day MCC",
        "Day Accuracy",
        "Day F1"
    ]
    
    n_rows = len(row_keys)
    # Create a figure with n_rows + 1 (legend) rows and 2 columns.
    fig = plt.figure(figsize=(12, 3*(n_rows+1)))
    gs = gridspec.GridSpec(n_rows + 1, 2, height_ratios=[0.5] + [1]*n_rows)
    
    # Create the legend axis in the top row.
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    
    # We will capture legend handles from one of the subplots (say, first row left).
    legend_handles = None
    legend_labels = None
    
    # Loop over each row (category-metric) and create two subplots (left: top files, right: bottom files)
    for i, (key, title) in enumerate(zip(row_keys, row_titles)):
        # Left subplot for top files.
        ax_left = fig.add_subplot(gs[i+1, 0])
        for (thresh, values, label) in curves_top[key]:
            c = color_dict.get(label, "black")
            m = linestyle_marker_dict.get(label, None)  # marker style if defined
            ms = markersize_dict.get(label, 6)
            ls = linestyle_dict.get(label, "solid")
            ax_left.plot(thresh, values, label=label, color=c, marker=m,
                         markersize=ms/10, markevery=5, linestyle=ls)
        ax_left.set_ylabel(title)
        # Only add x-label on the bottom row
        if i == n_rows - 1:
            ax_left.set_xlabel("Threshold")
        ax_left.grid(True)
        ax_left.set_title(f"{title} (DoTA)", pad=10)
        
        # Right subplot for bottom files.
        ax_right = fig.add_subplot(gs[i+1, 1])
        for (thresh, values, label) in curves_bottom[key]:
            c = color_dict.get(label, "black")
            m = linestyle_marker_dict.get(label, None)
            ms = markersize_dict.get(label, 6)
            ls = linestyle_dict.get(label, "solid")
            ax_right.plot(thresh, values, label=label, color=c, marker=m,
                          markersize=ms/10, markevery=5, linestyle=ls)
        if i == n_rows - 1:
            ax_right.set_xlabel("Threshold")
        ax_right.grid(True)
        ax_right.set_title(f"{title} (DADA2K)", pad=10)
        
        # Capture legend handles from the first row left subplot.
        if i == 0:
            legend_handles, legend_labels = ax_left.get_legend_handles_labels()
    
    # Place the legend in the top row.
    ax_legend.legend(legend_handles, legend_labels, loc="center", ncol=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("anaysis/plots/plot_egonight_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # List of file paths and corresponding labels for each file.
    numbers_dota = (0, 2, 4, 6, 8, 12, 16, 20, 24)
    numbers_dada = (1, 3, 5, 7, 10, 14, 18, 22, 26)
    labels = ("MOVAD", "VidNeXt", "ConvNeXt", "ResNet+NST", "R(2+1)D", "VideoMAE-S", "VideoMAE2-S", "InternVideo2-S", "ViViT-B")
    file_paths_dota = [DoTA_dirs[i] for i in numbers_dota]
    file_paths_dada = [DADA2K_dirs[i] for i in numbers_dada]
    #file_labels = [os.path.basename(os.path.dirname(fp)).split("_")[0] for fp in file_paths]  # labels for the curves
    #print(file_labels)

    plot_group_thresholded_metrics(file_paths_top=file_paths_dota, file_paths_bottom=file_paths_dada, file_labels=labels)


