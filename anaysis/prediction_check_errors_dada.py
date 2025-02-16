import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import label
from torch.nn.functional import softmax
from natsort import natsorted

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from dada import FrameClsDataset_DADA
from engine_for_frame_finetuning import calculate_metrics


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

dada_anno_folder = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K/annotation/full_anno.csv"
predictions1 = "logs/other_models_results/min_results_DADA2K/pred_min_best_model_dada2k.csv"
clip_err_out = "err_report.csv"
out_figures_dir = "err_report"
epoch = -1
tag = "_FULL" # "_train" or ""
show_hists = True
save_plots = True

# if "crossentropy" in predictions1:
#     loss_tag = "CE"
# elif "focal" in predictions1:
#     loss_tag = "Focal"
# else:
#     raise ValueError("Impossible loss directory!")
loss_tag = "train CAPDATA FULL, test DADA2K"

# ======================================================
predictions = predictions1.format(epoch, tag)
clip_err_out = os.path.join(os.path.dirname(predictions), clip_err_out)
out_figures_dir = os.path.join(os.path.dirname(predictions), out_figures_dir)
df = pd.read_csv(predictions)
logits = torch.tensor(df[["logits_safe", "logits_risk"]].to_numpy())
probs = softmax(logits, dim=-1)
probs = probs[:, 1].numpy()
labels = df["label"].to_numpy().astype(bool)

pos_preds = probs[labels]
neg_preds = probs[~labels]

# save stats
metr_acc, recall, precision, f1, confmat, auroc, ap, pr_curve, roc_curve = calculate_metrics(logits, torch.tensor(labels))
lines = ["\n===================================",
         f"mAP: {ap}, auroc: {auroc}, acc: {metr_acc}",
         f"P@0.5: {precision}, R@0.5: {recall}, F1@0.5: {f1}",
         f"Confmat: \n\t{confmat[0][0]} | {confmat[0][1]} \n\t{confmat[1][0]} | {confmat[1][1]}",
         f"----------------------------"]
with open(os.path.join(os.path.dirname(predictions), "general_stats.txt"), "w") as f:
    for l in lines:
        f.write(l + "\n")
        print(l)

if save_plots:
    os.makedirs(out_figures_dir, exist_ok=True)

if show_hists:
    plt.figure(figsize=(8, 6))
    plt.hist([neg_preds, pos_preds], bins=101, cumulative=False, edgecolor='black', label=['neg', 'pos'])
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.ylim(0, 20000)
    plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram ')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(neg_preds, bins=101, cumulative=1, edgecolor='black', alpha=0.7, label='neg')
    plt.hist(pos_preds, bins=101, cumulative=-1, edgecolor='black', alpha=0.7, label='pos')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Cumulative histogram')
    plt.legend()
    plt.show()

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
#err_df["night"] = None

anno_path = dada_anno_folder
anno = pd.read_csv(anno_path)
for i, row in err_df.iterrows():
    clip_name = row["clip"]
    clip_type, clip_subfolder = clip_name.split("/")
    row = anno[(anno["video"] == int(clip_subfolder)) & (anno["type"] == int(clip_type))]
    assert len(row) == 1, f"Multiple results! \n{clip_name}"
    err_df.loc[i, "category"] = clip_type
    err_df.loc[i, "ego"] = clip_type in FrameClsDataset_DADA.ego_categories
    #err_df.loc[i, "night"] = anno["night"]

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
#night_df = err_df[err_df["night"]]
#nonight_df = err_df[err_df["night"] == False]

scores_ego = [ego_df["err_score"].mean(), ego_df["err_far_score"].mean(), noego_df["err_score"].mean(), noego_df["err_far_score"].mean()]
scores_ego_labels = ["ego_score", "ego_far_score", "noego_score", "noego_far_score"]
#scores_night = [night_df["err_score"].mean(), night_df["err_far_score"].mean(), nonight_df["err_score"].mean(), nonight_df["err_far_score"].mean()]
#scores_night_labels = ["night_score", "night_far_score", "day_score", "day_far_score"]

if save_plots:
    os.makedirs(out_figures_dir, exist_ok=True)

fig = plt.figure(figsize=(8, 6), num="scores_categories")
plt.bar(cats, scores_cat, color='blue', label='normal')
plt.xlabel('categories')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram categories')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="scores_far_categories")
plt.bar(cats, scores_far_cat, color='blue', label='normal')
plt.xlabel('categories')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram categories')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

fig = plt.figure(figsize=(8, 6), num="scores_ego")
plt.bar(scores_ego_labels, scores_ego, color='blue', label='normal')
plt.xlabel('categories')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram categories')
plt.legend()
plt.show()
if save_plots:
    fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))

# fig = plt.figure(figsize=(8, 6), num="scores_night")
# plt.bar(scores_night_labels, scores_night, color='blue', label='normal')
# plt.xlabel('categories')
# plt.ylabel('Count')
# plt.xticks(rotation=45, ha='right')
# plt.title(f'{tag} [{loss_tag}, epoch {epoch}] Histogram categories')
# plt.legend()
# plt.show()
# if save_plots:
#     fig.savefig(os.path.join(out_figures_dir, f"{fig.get_label()}.png".replace(" ", "_")))


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

th = 0.5
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



