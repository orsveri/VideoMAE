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


def align_ylim(ax0, ax2):
    # Assume ax0 and ax2 are the two axes you want to synchronize.
    ylim0 = ax0.get_ylim()
    ylim2 = ax2.get_ylim()

    # Compute the common limits
    common_lower = min(ylim0[0], ylim2[0])
    common_upper = max(ylim0[1], ylim2[1])
    common_ylim = (common_lower, common_upper)

    # Set the new y-limits for both axes
    ax0.set_ylim(common_ylim)
    ax2.set_ylim(common_ylim)


def plot_ID_3x4_curves(file_paths_left_id, file_paths_left_ood, file_paths_right_id, file_paths_right_ood, file_labels, column_labels):
    # Prepare lists to store curves for each metric for the left dataset
    mcc_curves = []
    acc_curves = []
    f1_curves = []

    for fpaths in (file_paths_left_id, file_paths_left_ood, file_paths_right_id, file_paths_right_ood):
        mcc_curves_part = []
        acc_curves_part = []
        f1_curves_part = []
    
        # Loop through left files
        for path in fpaths:
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
            
            mcc_curves_part.append((mcc_thresholds, mcc_values))
            acc_curves_part.append((acc_thresholds, acc_values))
            f1_curves_part.append((f1_thresholds, f1_values))

        mcc_curves.append(mcc_curves_part)
        acc_curves.append(acc_curves_part)
        f1_curves.append(f1_curves_part)
    
    # Define metric titles and store the corresponding curves for left/right in dictionaries.
    metric_titles = ["MCC", "Accuracy", "F1"]
    metrics_values = []
    for mcc_curves_part, acc_curves_part, f1_curves_part in zip(mcc_curves, acc_curves, f1_curves):
        metrics_values.append({
            "MCC": mcc_curves_part,
            "Accuracy": acc_curves_part,
            "F1": f1_curves_part
        })
    
    # Create a figure with 4 rows x 4 columns using GridSpec.
    # Row 0 will be for the legend. Rows 1-3 for the 5 metrics.
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 4, height_ratios=[0.25, 1, 1, 1])
    gs.update(hspace=0.4)
    
    # Create the legend axis spanning both columns in the top row.
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    
    # We'll capture legend handles from one of the metric plots (e.g., from MCC on the left).
    legend_handles = None
    legend_labels = None
    
    # Create subplots for each metric and dataset.
    for i, m_title in enumerate(metric_titles):
        # Left subplot for the current metric
        axes_list = []
        for ic, (m_values, col_label) in enumerate(zip(metrics_values, column_labels)):
            ax_c = fig.add_subplot(gs[i+1, ic])
            axes_list.append(ax_c)
            for (thresh, values), label in zip(m_values[m_title], file_labels):
                c = color_dict.get(label, "black")       # default to black if not found
                m = linestyle_marker_dict.get(label, None)          # default marker if not found
                ms = markersize_dict.get(label, 6)       # default size if not found
                ls = linestyle_dict.get(label, "solid")
                ax_c.plot(
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
            #ax_c.set_xlabel("Threshold")
            ax_c.set_ylabel(m_title)
            ax_c.set_title(f"{m_title} ({col_label})", pad=5)
            ax_c.grid(True)
        
            # For the first metric (MCC), capture the legend handles/labels.
            if i == 0 and ic == 0:
                legend_handles, legend_labels = ax_c.get_legend_handles_labels()

        # make ylim equal for column pairs
        align_ylim(axes_list[0], axes_list[1])
        align_ylim(axes_list[2], axes_list[3])
    
    # Create the legend in the top row.
    ax_legend.legend(legend_handles, legend_labels, loc="center", ncol=5, fontsize=14)
    
    # Save and show the figure.
    fig.savefig("anaysis/plots/PT2_plot_3x4_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

# Example usage:
# file_paths_left and file_paths_right should be lists of directories where "thresh_stats.csv" is found.
# file_labels should be a list of model names.
# plot_ID_5x2_curves(file_paths_left, file_paths_right, file_labels)



if __name__ == "__main__":
    # List of file paths and corresponding labels for each file.
    numbers_dota_id = (0, 2, 4, 6, 8, 12, 16, 20, 24)
    numbers_dota_ood = (1, 3, 5, 7, 9, 14, 18, 22, 26)
    numbers_dada_id = (1, 3, 5, 7, 10, 14, 18, 22, 26)
    numbers_dada_ood = (0, 2, 4, 6, 8, 12, 16, 20, 24)
    # PT2
    numbers_dota_id = (0, 2, 4, 6, 8, 36, 40, 44, 48)
    numbers_dota_ood = (1, 3, 5, 7, 9, 38, 42, 46, 50)
    numbers_dada_id = (1, 3, 5, 7, 10, 38, 42, 46, 50)
    numbers_dada_ood = (0, 2, 4, 6, 8, 36, 40, 44, 48)
    #
    labels = ("MOVAD", "VidNeXt", "ConvNeXt", "ResNet+NST", "R(2+1)D", "VideoMAE-S", "VideoMAE2-S", "InternVideo2-S", "ViViT-B")
    fpaths1 = [DoTA_dirs[i] for i in numbers_dota_id]
    fpaths2 = [DoTA_dirs[i] for i in numbers_dota_ood]
    fpaths3 = [DADA2K_dirs[i] for i in numbers_dada_id]
    fpaths4 = [DADA2K_dirs[i] for i in numbers_dada_ood]
    columns = ("DoTA->DoTA", "DADA2K->DoTA", "DADA2K->DADA2K", "DoTA->DADA2K")
    #file_labels = [os.path.basename(os.path.dirname(fp)).split("_")[0] for fp in file_paths]  # labels for the curves
    #print(file_labels)

    plot_ID_3x4_curves(fpaths1, fpaths2, fpaths3, fpaths4, file_labels=labels, column_labels=columns)


