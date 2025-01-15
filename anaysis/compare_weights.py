import os
import torch
import matplotlib.pyplot as plt


def compare_weights(pretrained_checkpoint, fine_tuned_checkpoint):
    layer_names = []
    diff = []
    for key in pretrained_checkpoint.keys():
        if key in fine_tuned_checkpoint.keys():
            deb1 = pretrained_checkpoint[key]
            deb2 = fine_tuned_checkpoint[key]
            if deb1.shape == deb2.shape:
                delta = (pretrained_checkpoint[key] - fine_tuned_checkpoint[key]).abs().mean().item()
                print(f"Key: {key}, Mean Difference: {delta}")
                layer_names.append(key)
                diff.append(delta)
            else:
                print(f"Key {key} has different shapes. Pretrained: {deb1.shape}, finetuned: {deb2.shape}")
        else:
            print(f"Key {key} not found in the second checkpoint.")
    return layer_names, diff


if False:
    ckpt_pre1 = "/home/sorlova/repos/NewStart/VideoMAE/logs/pretrained/k400_vits/checkpoint.pth"
    ckpt_pre2 = "/home/sorlova/repos/NewStart/VideoMAE/logs/pretrained/distill/vit_s_k710_dl_from_giant.pth"
    save_name = f"vits_{dset}_2gpufocal_vs_k710_part.png"

    pretrained_checkpoint = torch.load(ckpt_pre2, map_location='cpu')
    fine_tuned_checkpoint = torch.load(ckpt_pre1, map_location='cpu')

    pretrained_checkpoint = pretrained_checkpoint["module"]
    # For pretrained K40
    fine_tuned_checkpoint = {k.replace("encoder.", ""): fine_tuned_checkpoint["model"][k] for k in fine_tuned_checkpoint["model"].keys() if k.startswith("encoder.")}
    value = fine_tuned_checkpoint.pop("norm.bias")
    fine_tuned_checkpoint["fc_norm.bias"] = value
    value = fine_tuned_checkpoint.pop("norm.weight")
    fine_tuned_checkpoint["fc_norm.weight"] = value

    s1 = set(pretrained_checkpoint.keys())
    s2 = set(fine_tuned_checkpoint.keys())
    s12 = s1 - s2
    s21 = s2 - s1

    layer_names, diff = compare_weights(pretrained_checkpoint, fine_tuned_checkpoint)

    plt.figure(figsize=(30, 15))
    plt.plot(list(range(len(layer_names))), diff)

    # Label the axes
    plt.xlabel("Layer Index (from earlier to later)", fontsize=12)
    plt.ylabel("Mean Absolute Change", fontsize=12)

    # Add a title
    plt.title("Mean Absolute Weight Changes Across Layers", fontsize=14)

    # Set x-axis ticks with layer names
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right', fontsize=10)
    #plt.ylim(0., 0.055)

    # Add a text box in the upper-left corner
    textstr = "[ViT-S] Weights mean absolute distance across layers:\n" + \
               f"pretrained model: K710 distill vs pretrained K400"
    plt.gca().text(
        0.02, 0.98, textstr, fontsize=40, color='black', transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.8)
    )
    plt.savefig(os.path.join(output_dir, "vits_K710_vs_K400.png"), dpi=300, bbox_inches="tight", format="png")
    exit(0)


ckpts = [1, 3]
dset = "DoTA"
ckpt_tuned = "/home/sorlova/repos/NewStart/VideoMAE/logs/check_things/freeze_finetune/checkpoint-{}/mp_rank_00_model_states.pt"
ckpt_pre = "/home/sorlova/repos/NewStart/VideoMAE/logs/pretrained/distill/vit_s_k710_dl_from_giant.pth"
output_dir = "/home/sorlova/repos/NewStart/VideoMAE/logs/weight_diff_analysis"
save_name = f"vits_{dset}_{'frozen'}_vs_k710.png"

# ==============

all_diffs = []
all_layer_names = []

for ckpt in ckpts:
    # Load checkpoints
    pretrained_checkpoint = torch.load(ckpt_pre, map_location='cpu')
    fine_tuned_checkpoint = torch.load(ckpt_tuned.format(ckpt), map_location='cpu')

    pretrained_checkpoint = pretrained_checkpoint["module"]
    fine_tuned_checkpoint = fine_tuned_checkpoint["module"]
    # For pretrained K400
    #fine_tuned_checkpoint = {k.replace("encoder.", ""): fine_tuned_checkpoint["model"][k] for k in fine_tuned_checkpoint["model"].keys() if k.startswith("encoder.")}

    layer_names, diff = compare_weights(pretrained_checkpoint, fine_tuned_checkpoint)
    all_diffs.append(diff)
    all_layer_names.append(layer_names)

assert all([item == all_layer_names[0] for item in all_layer_names])

plt.figure(figsize=(30, 15))
for c, x, y in zip(ckpts, all_layer_names, all_diffs):
    plt.plot(x, y, label=c)

# Label the axes
plt.xlabel("Layer Index (from earlier to later)", fontsize=12)
plt.ylabel("Mean Absolute Change", fontsize=12)

# Add a title
plt.title("Mean Absolute Weight Changes Across Layers", fontsize=14)

# Set x-axis ticks with layer names
plt.xticks(range(len(all_layer_names[0])), all_layer_names[0], rotation=45, ha='right', fontsize=10)
#plt.ylim(0., 0.055)

# Add a text box in the upper-left corner
textstr = "[ViT-S] Weights mean absolute distance across layers:\n" + \
           f"pretrained model: K710 distill, fine-tuned: {dset} ep {ckpts}"
plt.gca().text(
    0.02, 0.98, textstr, fontsize=40, color='black', transform=plt.gca().transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.8)
)

plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches="tight", format="png")

# Display the plot
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
