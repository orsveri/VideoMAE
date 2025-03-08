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



def plot_category_recall_curves(file_paths_top, file_paths_bottom, file_labels):
    """
    For each file in two lists (top and bottom), read the CSV ("thresh_stats.csv"),
    select the rows whose index starts with "cat_", and extract recall curves (columns starting with "r_").
    Then, for each category (e.g. "cat_0", "cat_1", ...), plot the recall curves as normal line plots.
    The final figure has 3 rows: row 0 for a legend and rows 1...N for each category,
    with two columns (left = top files, right = bottom files).
    
    Parameters:
      file_paths_top : list of str
          Directories (each containing "thresh_stats.csv") for the top dataset.
      file_paths_bottom : list of str
          Directories (each containing "thresh_stats.csv") for the bottom dataset.
      file_labels : list of str
          Labels for each method (assumed to be in the same order as file_paths).
    """
    # Dictionaries to store recall curves for each category.
    # Keys: category name (e.g. "cat_0"), values: list of (thresholds, recall_values) for each method.
    cat_curves_top = {}
    for path, label in zip(file_paths_top, file_labels):
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.set_index("group")
        # Select only rows whose index starts with "cat_"
        df_cats = df.loc[df.index.str.startswith("cat_")]
        # Identify recall columns (those starting with "r_") and sort them by threshold value.
        recall_cols = sorted([col for col in df.columns if col.startswith("r_")],
                             key=lambda c: extract_threshold(c, "r_"))
        thresholds = np.array([extract_threshold(c, "r_") for c in recall_cols])
        # For each category (row) in df_cats, store the curve.
        for cat in df_cats.index:
            values = df_cats.loc[cat, recall_cols].values.astype(float)
            if cat not in cat_curves_top:
                cat_curves_top[cat] = []
            cat_curves_top[cat].append((thresholds, values))
    
    cat_curves_bottom = {}
    for path, label in zip(file_paths_bottom, file_labels):
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.set_index("group")
        df_cats = df.loc[df.index.str.startswith("cat_")]
        recall_cols = sorted([col for col in df.columns if col.startswith("r_")],
                             key=lambda c: extract_threshold(c, "r_"))
        thresholds = np.array([extract_threshold(c, "r_") for c in recall_cols])
        for cat in df_cats.index:
            values = df_cats.loc[cat, recall_cols].values.astype(float)
            if cat not in cat_curves_bottom:
                cat_curves_bottom[cat] = []
            cat_curves_bottom[cat].append((thresholds, values))
    
    # Assume both dictionaries have the same set of categories.
    categories = sorted(cat_curves_top.keys())
    
    n_categories = len(categories)
    # Create a figure with (n_categories + 1) rows and 2 columns.
    # Row 0: legend; Rows 1...n_categories: one row per category.
    fig = plt.figure(figsize=(12, 3*(n_categories+1)))
    gs = gridspec.GridSpec(n_categories + 1, 2, height_ratios=[0.5] + [1]*n_categories)
    
    # Create legend axis (row 0 spanning both columns)
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    
    # We'll capture legend handles from one of the subplots (e.g., from the first category, top dataset)
    legend_handles = None
    legend_labels = None
    
    # For each category, create two subplots: left for top files, right for bottom files.
    for i, cat in enumerate(categories):
        # Left subplot (top dataset)
        ax_left = fig.add_subplot(gs[i+1, 0])
        for (thresh, values), label in zip(cat_curves_top[cat], file_labels):
            c = color_dict.get(label, "black")
            m = linestyle_marker_dict.get(label, None)  # marker style if defined
            ms = markersize_dict.get(label, 6)
            ls = linestyle_dict.get(label, "solid")
            ax_left.plot(thresh, values, label=label, color=c, marker=m,
                         markersize=ms/10, markevery=5, linestyle=ls)
        ax_left.set_ylabel(f"{cat}\nRecall")
        # Optionally, set no x-label for these (only bottom row gets an x-label)
        if i == n_categories - 1:
            ax_left.set_xlabel("Threshold")
        ax_left.grid(True)
        ax_left.set_title(f"{cat} (DoTA)", pad=10)
        
        # Right subplot (bottom dataset)
        ax_right = fig.add_subplot(gs[i+1, 1])
        for (thresh, values), label in zip(cat_curves_bottom[cat], file_labels):
            c = color_dict.get(label, "black")
            m = linestyle_marker_dict.get(label, None)
            ms = markersize_dict.get(label, 6)
            ls = linestyle_dict.get(label, "solid")
            ax_right.plot(thresh, values, label=label, color=c, marker=m,
                          markersize=ms/10, markevery=5, linestyle=ls)
        # Only label the x-axis on the bottom row
        if i == n_categories - 1:
            ax_right.set_xlabel("Threshold")
        ax_right.grid(True)
        ax_right.set_title(f"{cat} (Also DoTA, DADA2K has 60+ categories)", pad=10)
        
        # For the first category, capture legend handles from the left subplot.
        if i == 0:
            legend_handles, legend_labels = ax_left.get_legend_handles_labels()
    
    # Create the legend in the top row.
    ax_legend.legend(legend_handles, legend_labels, loc="center", ncol=3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("anaysis/plots/plot_cat_recall.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # List of file paths and corresponding labels for each file.
    numbers_dota = (0, 2, 4, 6, 8, 12, 16, 20, 24)
    numbers_dada = (1, 3, 5, 7, 10, 14, 18, 22, 26)
    labels = ("MOVAD", "VidNeXt", "ConvNeXt", "ResNet+NST", "R(2+1)D", "VideoMAE-S", "VideoMAE2-S", "InternVideo2-S", "ViViT-B")
    file_paths_dota = [DoTA_dirs[i] for i in numbers_dota]
    #file_paths_dada = [DADA2K_dirs[i] for i in numbers_dada]
    #file_labels = [os.path.basename(os.path.dirname(fp)).split("_")[0] for fp in file_paths]  # labels for the curves
    #print(file_labels)

    plot_category_recall_curves(file_paths_top=file_paths_dota, file_paths_bottom=file_paths_dota, file_labels=labels)


