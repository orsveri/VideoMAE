import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import label
from torch.nn.functional import softmax
from natsort import natsorted


def get_pos_and_neg_probs():
    logits = torch.tensor(df[["logits_safe", "logits_risk"]].to_numpy())
    probs = softmax(logits, dim=-1)
    probs = probs[:, 1].numpy()
    labels = df["label"].to_numpy().astype(bool)

    pos_preds = probs[labels]
    neg_preds = probs[~labels]
    return neg_preds, pos_preds


# Prediction (model output) as a tensor
prediction = torch.tensor([0.5], dtype=torch.float32)  # Predicted probability
target = torch.tensor([1.0], dtype=torch.float32)  # Ground truth label

# Avoid log(0) by clipping prediction
epsilon = 1e-7
prediction_clipped = torch.clamp(prediction, epsilon, 1.0)

# Alternatively, use PyTorch's BCE function
bce_loss_builtin = torch.nn.functional.binary_cross_entropy(prediction_clipped, target)
print(f"Binary Cross-Entropy Loss (Builtin): {bce_loss_builtin.item()}")

# "_fixttc" ?
predictions1 = "/home/sorlova/repos/NewStart/VideoMAE/logs/clean_datasets/DoTA/b32x2x1gpu_ce_VAL/checkpoint-{}/OUTval/predictions_0.csv"
clip_err_out = "err_report.csv"
out_figures_dir = "err_report"
epoch = 15
tag = "all val" # "_train" or ""
show_hists = True
save_plots = True

loss_tag = "CE"


# ======================================================
predictions = predictions1.format(epoch)
clip_err_out = os.path.join(os.path.dirname(predictions), clip_err_out)
out_figures_dir = os.path.join(os.path.dirname(predictions), out_figures_dir)
df = pd.read_csv(predictions)
logits = torch.tensor(df[["logits_safe", "logits_risk"]].to_numpy())
probs = softmax(logits, dim=-1)
probs = probs[:, 1].numpy()
labels = df["label"].to_numpy().astype(bool)

pos_preds = probs[labels]
neg_preds = probs[~labels]

if save_plots:
    os.makedirs(out_figures_dir, exist_ok=True)

if show_hists:
    fig = plt.figure(figsize=(8, 6), num="probs histogram")
    plt.hist([neg_preds, pos_preds], bins=101, cumulative=False, edgecolor='black', label=['neg', 'pos'])
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.ylim(0, 20000)
    plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram ')
    plt.legend()
    plt.show()
    if save_plots:
        fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

    fig = plt.figure(figsize=(8, 6), num="probs histogram cumulative")
    plt.hist(neg_preds, bins=101, cumulative=1, edgecolor='black', alpha=0.7, label='neg')
    plt.hist(pos_preds, bins=101, cumulative=-1, edgecolor='black', alpha=0.7, label='pos')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Cumulative histogram')
    plt.legend()
    plt.show()
    if save_plots:
        fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

df["probs_anomaly"] = probs
# Get misclassifications
err_pos_condition = (df["label"] == 1) & (df["probs_anomaly"] < 0.5)  # missed, FN
err_neg_condition = (df["label"] == 0) & (df["probs_anomaly"] > 0.5)  # FP
proximity_condition = df["ttc"].between(-2., 0., inclusive="left") | df["ttc"].between(0., 1., inclusive="right")
err_pos_df = df[err_pos_condition]
err_neg_df = df[err_neg_condition]
df_transition = df[proximity_condition]
# get clips
clip_list = natsorted(df["clip"].unique().tolist())
err_df = pd.DataFrame({"clip": clip_list})
err_df["nb_samples"] = 0
err_df["nb_err_pos"] = 0
err_df["nb_err_neg"] = 0
err_df["nb_err_pos_transition"] = 0
err_df["nb_err_neg_transition"] = 0
err_df["nb_err_pos_far"] = 0
err_df["nb_err_neg_far"] = 0

for i, row in err_df.iterrows():
    clip_name = row["clip"]
    err_df.loc[i, "nb_samples"] = len(df[df["clip"] == clip_name])
    err_df.loc[i, "nb_err_pos"] = len(err_pos_df[err_pos_df["clip"] == clip_name])
    err_df.loc[i, "nb_err_neg"] = len(err_neg_df[err_neg_df["clip"] == clip_name])
    err_df.loc[i, "nb_err_pos_transition"] = len(df[err_pos_condition & (df["clip"] == clip_name) & proximity_condition])
    err_df.loc[i, "nb_err_neg_transition"] = len(df[err_neg_condition & (df["clip"] == clip_name) & proximity_condition])
    err_df.loc[i, "nb_err_pos_far"] = err_df.loc[i, "nb_err_pos"] - err_df.loc[i, "nb_err_pos_transition"]
    err_df.loc[i, "nb_err_neg_far"] = err_df.loc[i, "nb_err_neg"] - err_df.loc[i, "nb_err_neg_transition"]

err_df["nb_errs"] = err_df["nb_err_pos"] + err_df["nb_err_neg"]
err_df["err_score"] = err_df["nb_errs"] / err_df["nb_samples"]
err_df["err_pos_score"] = err_df["nb_err_pos"] / err_df["nb_samples"]
err_df["err_neg_score"] = err_df["nb_err_neg"] / err_df["nb_samples"]
err_df["err_pos_trans_score"] = err_df["nb_err_pos_transition"] / err_df["nb_samples"]
err_df["err_neg_trans_score"] = err_df["nb_err_neg_transition"] / err_df["nb_samples"]
err_df["err_pos_far_score"] = err_df["nb_err_pos_far"] / err_df["nb_samples"]
err_df["err_neg_far_score"] = err_df["nb_err_neg_far"] / err_df["nb_samples"]
err_df["err_far_score"] = (err_df["nb_err_neg_far"] + err_df["nb_err_pos_far"]) / err_df["nb_samples"]


# check classes:
err_df["category"] = None
err_df["ego"] = None
err_df["night"] = None
import json
for i, row in err_df.iterrows():
    clip_name = row["clip"]
    anno_path = os.path.join("/mnt/experiments/sorlova/datasets/DoTA/dataset/annotations", clip_name + ".json")
    with open(anno_path) as f:
        anno = json.load(f)
        err_df.loc[i, "category"] = anno["accident_name"]
        err_df.loc[i, "ego"] = anno["ego_involve"]
        err_df.loc[i, "night"] = anno["night"]

err_df.sort_values(by="err_score", ascending=False, inplace=True)
err_df.to_csv(clip_err_out)

# statistics by categories
cats = natsorted(err_df["category"].unique().tolist())
scores_cat = []
scores_far_cat = []
scores_ego = []
scores_night = []
for cat in cats:
    cat_df = err_df[err_df["category"] == cat]
    score = cat_df["err_score"].mean()
    score_far = cat_df["err_far_score"].mean()
    scores_cat.append(score)
    scores_far_cat.append(score_far)

ego_df = err_df[err_df["ego"]]
noego_df = err_df[err_df["ego"] == False]
night_df = err_df[err_df["night"]]
nonight_df = err_df[err_df["night"] == False]

scores_ego = [ego_df["err_score"].mean(), ego_df["err_far_score"].mean(), noego_df["err_score"].mean(), noego_df["err_far_score"].mean()]
scores_ego_labels = ["ego_score", "ego_far_score", "noego_score", "noego_far_score"]
scores_night = [night_df["err_score"].mean(), night_df["err_far_score"].mean(), nonight_df["err_score"].mean(), nonight_df["err_far_score"].mean()]
scores_night_labels = ["night_score", "night_far_score", "day_score", "day_far_score"]

fig = plt.figure(figsize=(8, 6), num="scores_categories")
plt.bar(cats, scores_cat, color='blue', label='all errs')
plt.xlabel('categories')
plt.ylabel('mean err score')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Mean err scores by category')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="scores_far_categories")
plt.bar(cats, scores_cat, color='blue', label='all errs', alpha=0.7)
plt.bar(cats, scores_far_cat, color='green', label='major', alpha=0.7)
plt.xlabel('categories')
plt.ylabel('mean err score')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Mean err scores by category')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="scores_ego")
plt.bar(scores_ego_labels[:2], scores_ego[:2], color='orange', label='ego')
plt.bar(scores_ego_labels[2:], scores_ego[2:], color='blue', label='no_ego')
plt.xlabel('ego participation')
plt.ylabel('mean err score')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Mean err scores by ego participation')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="scores_night")
plt.bar(scores_night_labels[:2], scores_night[:2], color='blue', label='night')
plt.bar(scores_night_labels[2:], scores_night[2:], color='orange', label='day')
plt.xlabel('day/night')
plt.ylabel('mean err score')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Mean err scores by day/night')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))


# DEFINE videos for analysis
nd_bad1 = len(err_df[err_df["err_far_score"] > 0.9])
nd_bad2 = len(err_df[err_df["err_far_score"] > 0.8])
nd_bad3 = len(err_df[err_df["err_far_score"] > 0.7])
nd_bad4 = len(err_df[err_df["err_far_score"] > 0.6])
nd_bad5 = len(err_df[err_df["err_far_score"] > 0.5])
nd_bad6 = len(err_df[err_df["err_far_score"] > 0.4])
nd_bad7 = len(err_df[err_df["err_far_score"] > 0.3])
nd_bad8 = len(err_df[err_df["err_far_score"] > 0.2])
nd_bad9 = len(err_df[err_df["err_far_score"] > 0.1])
nd_bad0 = len(err_df[err_df["err_far_score"] > 0.05])
nd_bad0t = len(err_df[err_df["err_far_score"] > 0.03])
nd_bad0s = len(err_df[err_df["err_far_score"] > 0.025])
nd_bad0s = len(err_df[err_df["err_far_score"] > 0.02])

nd_bad00 = len(err_df[err_df["err_score"] > 0.3])
nd_bad01 = len(err_df[err_df["err_score"] > 0.05])
nd_bad02 = len(err_df[err_df["err_score"] > 0.04])
nd_bad03 = len(err_df[err_df["err_score"] > 0.03])

th = 0.025
bad_clips = err_df[err_df["err_far_score"] > th]
bad_clips.to_csv(os.path.splitext(clip_err_out)[0] + f"_bad{th}.csv")

fig = plt.figure(figsize=(8, 6), num="categories")
plt.hist(
    bad_clips["category"].to_list(),
    #bins=bins,
    #range=score_range,
    cumulative=False,
    edgecolor='black',
    label='categories'
)
plt.xlabel('categories')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram categories')
plt.legend()
plt.show()

# =========================================================================================

bins = 21
score_range = (0, 1)
x_indices = range(len(err_df))

fig = plt.figure(figsize=(10, 6), num="clip scores")
plt.bar(x_indices, err_df["err_score"].to_numpy(), width=1, color='blue', label='normal')
plt.bar(x_indices, err_df["err_far_score"].to_numpy(), width=1, color='orange', label='far')
cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.80
plt.text(cx, cy,
         f'Avg err score \nnormal: {round(err_df["err_score"].mean(), 2)} \n  far: {round(err_df["err_far_score"].mean(), 2)}',
         fontsize=18, color='blue', ha='center', va='center', weight='demibold')
plt.xlabel('clip')
plt.ylabel('err score')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Err scores by clip')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="all scores")
plt.hist(
    err_df["err_score"].to_numpy(),
    bins=bins,
    range=score_range,
    cumulative=False,
    edgecolor='black',
    label='err_score'
)
cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.9
plt.text(cx, cy, f'Avg score: {round(err_df["err_score"].mean(), 2)}', fontsize=14, color='blue', ha='center', va='center', weight='demibold')
plt.xlabel('err_score')
plt.ylabel('Count')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram scores')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="FN scores")
plt.hist(
    err_df["err_pos_score"].to_numpy(),
    bins=bins,
    range=score_range,
    cumulative=False,
    edgecolor='black',
    label='FN'
)
cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.9
plt.text(cx, cy, f'Avg score: {round(err_df["err_pos_score"].mean(), 2)}', fontsize=14, color='blue', ha='center', va='center', weight='demibold')
plt.xlabel('err_score')
plt.ylabel('Count')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram scores FN')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="FP scores")
plt.hist(
    err_df["err_neg_score"].to_numpy(),
    bins=bins,
    range=score_range,
    cumulative=False,
    edgecolor='black',
    label='FP'
)
cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.9
plt.text(cx, cy, f'Avg score: {round(err_df["err_neg_score"].mean(), 2)}', fontsize=14, color='blue', ha='center', va='center', weight='demibold')
plt.xlabel('err_score')
plt.ylabel('Count')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram scores FP')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="all scores no transition")
plt.hist(
    err_df["err_far_score"].to_numpy(),
    bins=bins,
    range=score_range,
    cumulative=False,
    edgecolor='black',
    label='scores'
)
cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.9
plt.text(cx, cy, f'Avg score: {round(err_df["err_far_score"].mean(), 2)}', fontsize=14, color='blue', ha='center', va='center', weight='demibold')
plt.xlabel('err_score')
plt.ylabel('Count')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram scores no transition')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

# fig = plt.figure(figsize=(8, 6), num="FN scores no transition")
# plt.hist(
#     err_df["err_pos_far_score"].to_numpy(),
#     bins=bins,
#     range=score_range,
#     cumulative=False,
#     edgecolor='black',
#     label='FN'
# )
# cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
# cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.9
# plt.text(cx, cy, f'Avg score: {round(err_df["err_pos_far_score"].mean(), 2)}', fontsize=14, color='blue', ha='center', va='center', weight='demibold')
# plt.xlabel('err_score')
# plt.ylabel('Count')
# plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram FN scores no transition')
# plt.legend()
# plt.show()
# if save_plots:
#     fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="FP scores no transition")
plt.hist(
    err_df["err_neg_far_score"].to_numpy(),
    bins=bins,
    range=score_range,
    cumulative=False,
    edgecolor='black',
    label='FP'
)
cx = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Midpoint of x-axis
cy = (plt.ylim()[0] + plt.ylim()[1]) * 0.9
plt.text(cx, cy, f'Avg score: {round(err_df["err_neg_far_score"].mean(), 2)}', fontsize=14, color='blue', ha='center', va='center', weight='demibold')
plt.xlabel('err_score')
plt.ylabel('Count')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram FP scores no transition')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))



