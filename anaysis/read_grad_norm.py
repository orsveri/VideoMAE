import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def plot_gradnorm_heatmap_qkv(param_array, param_name=None, save_to=""):
    # param_array[i,j] is the mean grad norm for layer i, head j
    num_layers, num_heads = param_array.shape
    fig, ax = plt.subplots(figsize=(num_heads+2, num_layers))  # Adjust size as needed
    sns.heatmap(
        param_array,
        annot=True, fmt=".3f",  # or False if no numbers
        cmap="viridis",         # choose your favorite colormap
        xticklabels=[f"H{h}" for h in range(num_heads)],
        yticklabels=[f"L{i}" for i in range(num_layers)],
        ax=ax
    )
    ax.set_title(f"Mean Gradient Norm Heatmap - {param_name}")
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers (0 = earliest)")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig) # ?


def plot_gradnorm_heatmap_2D(param_array, x_labels, param_name=None, x_axis_label=None, save_to=""):
    """
    Plots a heatmap for a 2D parameter array and saves it if a path is specified.

    Args:
        param_array (numpy.ndarray): Array of shape (num_layers, num_params).
        x_labels (list): Labels for the x-axis (columns of the array).
        param_name (str, optional): Title for the plot.
        x_axis_label (str, optional): Label for the x-axis.
        save_to (str, optional): Path to save the plot. If empty, the plot is displayed.
    """
    num_layers, num_params = param_array.shape
    assert len(x_labels) == num_params, "Number of x_labels must match the number of columns in param_array."

    # Adjust figure size dynamically
    fig, ax = plt.subplots(figsize=(num_params * 1.5, num_layers * 0.8))  # Width depends on x_labels

    sns.heatmap(
        param_array,
        annot=True, fmt=".3f",  # Show numbers with 3 decimal places
        cmap="viridis",         # Colormap for the heatmap
        xticklabels=x_labels,
        yticklabels=[f"L{i}" for i in range(num_layers)],
        cbar_kws={"shrink": 0.8},  # Shrink colorbar for better formatting
        linewidths=0.5,           # Add lines between cells
        linecolor="gray"
    )
    ax.set_title(f"Mean Gradient Norm Heatmap - {param_name}", fontsize=14, pad=12)
    ax.set_xlabel(x_axis_label or "Parameters", fontsize=12)
    extra = " (0 = earliest)" if num_layers > 1 else ""
    ax.set_ylabel("Layers" + extra, fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout to fit everything
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_gradnorm_heatmap_2D_separate_scales(param_array, x_labels, param_name=None, x_axis_label=None, save_to=""):
    """
    Plots a heatmap for a 2D parameter array with separate color scales for each column.

    Args:
        param_array (numpy.ndarray): Array of shape (num_layers, num_params).
        x_labels (list): Labels for the x-axis (columns of the array).
        param_name (str, optional): Title for the plot.
        x_axis_label (str, optional): Label for the x-axis.
        save_to (str, optional): Path to save the plot. If empty, the plot is displayed.
    """
    num_layers, num_params = param_array.shape
    assert len(x_labels) == num_params, "Number of x_labels must match the number of columns in param_array."

    fig, axes = plt.subplots(1, num_params, figsize=(2.0 * num_params, 6), constrained_layout=True)

    if num_params == 1:  # If there's only one parameter, axes is not iterable
        axes = [axes]

    for col_idx in range(num_params):
        sns.heatmap(
            param_array[:, col_idx:col_idx+1],  # Select the single column
            annot=True, fmt=".3f",
            cmap="viridis",
            cbar=True, ax=axes[col_idx],
            xticklabels=[x_labels[col_idx]],  # Single column label
            yticklabels=[f"L{i}" for i in range(num_layers)],
            cbar_kws={"shrink": 0.8}
        )
        axes[col_idx].set_title(f"{x_labels[col_idx]}", fontsize=12)
        axes[col_idx].set_xlabel(x_axis_label or "Parameters", fontsize=10)
        if col_idx == 0:
            axes[col_idx].set_ylabel("Layers (0 = earliest)", fontsize=10)
        else:
            axes[col_idx].set_ylabel("")  # Avoid duplicate y-axis labels

    if param_name:
        fig.suptitle(f"Mean Gradient Norm Heatmap - {param_name}", fontsize=14, y=1.02)

    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_gradnorm_vis(epoch, inp_path, output_dir):
    loaded_data = np.load(inp_path)
    # Convert to dictionary if needed
    data_dict = {key: loaded_data[key] for key in loaded_data}
    #
    qkv = data_dict["qkv"]
    proj_ = data_dict["proj"]
    if proj_.shape[-1] == 6:
        proj = proj_[:,:2]
        mlp1 = proj_[:,2:4]
        mlp2 = proj_[:,4:]
    else:
        proj = proj_
    patch_embed = np.expand_dims(data_dict["patch_embed"], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    plot_gradnorm_heatmap_qkv(qkv[:,:,0], param_name=f"Q ep {epoch}", save_to=os.path.join(output_dir, f"Qw_ep{epoch}.png"))
    plot_gradnorm_heatmap_qkv(qkv[:,:,1], param_name=f"K ep {epoch}", save_to=os.path.join(output_dir, f"Kw_ep{epoch}.png"))
    plot_gradnorm_heatmap_qkv(qkv[:,:,2], param_name=f"V ep {epoch}", save_to=os.path.join(output_dir, f"Vw_ep{epoch}.png"))
    plot_gradnorm_heatmap_qkv(qkv[:,:,3], param_name=f"Q bias ep {epoch}", save_to=os.path.join(output_dir, f"Qb_ep{epoch}.png"))
    plot_gradnorm_heatmap_qkv(qkv[:,:,4], param_name=f"V bias ep {epoch}", save_to=os.path.join(output_dir, f"Vb_ep{epoch}.png"))
    plot_gradnorm_heatmap_2D_separate_scales(proj, x_labels=["weight", "bias"], param_name=f"Proj layer ep {epoch}", save_to=os.path.join(output_dir, f"Proj_ep{epoch}.png"))
    if proj_.shape[-1] == 6:
        plot_gradnorm_heatmap_2D_separate_scales(mlp1, x_labels=["weight", "bias"], param_name=f"MLP-1 ep {epoch}", save_to=os.path.join(output_dir, f"Mlp1_ep{epoch}.png"))
        plot_gradnorm_heatmap_2D_separate_scales(mlp2, x_labels=["weight", "bias"], param_name=f"MLP-2 ep {epoch}", save_to=os.path.join(output_dir, f"Mlp2_ep{epoch}.png"))
    plot_gradnorm_heatmap_2D(patch_embed, x_labels=["weight", "bias"], param_name=f"\nPatch embed proj ep {epoch}", save_to=os.path.join(output_dir, f"PE_ep{epoch}.png"))
    print("Done!")


epoch = 6

#
inp_path = "logs/baselines/bl3/dota_QKVbias_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/grad_norms/gradnorm_ep{}.npz"
output_dir = os.path.join(os.path.dirname(os.path.dirname(inp_path)), "grad_norms_vis")
print(inp_path)
for epoch in tqdm(range(5+1)):
    save_gradnorm_vis(epoch=epoch, inp_path=inp_path.format(epoch), output_dir=output_dir)


# inp_path = "/home/sorlova/repos/AITHENA/NewStage/VideoMAE/logs/my_pretrain/vits_vidmae_k400/bdd-capdata_lightcrop_b200x4_mask075/grad_norms/gradnorm_ep{}.npz"
# output_dir = os.path.join(os.path.dirname(os.path.dirname(inp_path)), "grad_norms_vis")
# print(inp_path)
# for epoch in tqdm(range(11+1)):
#     save_gradnorm_vis(epoch=epoch, inp_path=inp_path.format(epoch), output_dir=output_dir)

