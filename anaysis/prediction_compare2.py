import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
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

predictions1 = "/home/sorlova/repos/NewStart/VideoMAE/logs/auroc_behavior/crossentropy/checkpoint-{}/OUT{}/predictions_0.csv"
predictions2 = "/home/sorlova/repos/NewStart/VideoMAE/logs/auroc_behavior/focal/checkpoint-{}/OUT{}/predictions_0.csv"
epoch = 15
tag = "_train" # "_train" or ""
show_hists = False


# ======================================================
predictions = predictions1.format(epoch, tag)
df = pd.read_csv(predictions)
logits = torch.tensor(df[["logits_safe", "logits_risk"]].to_numpy())
probs = softmax(logits, dim=-1)
probs = probs[:, 1].numpy()
labels = df["label"].to_numpy().astype(bool)

pos_preds = probs[labels]
neg_preds = probs[~labels]

if show_hists:
    plt.figure(figsize=(8, 6))
    plt.hist([neg_preds, pos_preds], bins=101, cumulative=False, edgecolor='black', label=['neg', 'pos'])
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.ylim(0, 20000)
    plt.title(f'{tag} [CE, epoch {epoch}] Histogram ')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(neg_preds, bins=101, cumulative=1, edgecolor='black', alpha=0.7, label='neg')
    plt.hist(pos_preds, bins=101, cumulative=-1, edgecolor='black', alpha=0.7, label='pos')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title(f'{tag} [CE, epoch {epoch}] Cumulative histogram')
    plt.legend()
    plt.show()

df["probs_anomaly"] = probs
# Get misclassifications
err_pos_condition = (df["label"] == 1) & (df["probs_anomaly"] < 0.5)
err_neg_condition = (df["label"] == 0) & (df["probs_anomaly"] > 0.5)
err_pos_df = df[err_pos_condition]
err_neg_df = df[err_neg_condition]
# get clips
clip_list = natsorted(df["clip"].unique().tolist())
err_df = pd.DataFrame({"clip": clip_list})
err_df["nb_samples"] = 0
err_df["nb_err_pos"] = 0
err_df["nb_err_neg"] = 0
for i, row in err_df.iterrows():
    clip_name = row["clip"]
    err_df.loc[i, "nb_samples"] = len(df[df["clip"] == clip_name])
    err_df.loc[i, "nb_err_pos"] = len(err_pos_df[err_pos_df["clip"] == clip_name])
    err_df.loc[i, "nb_err_neg"] = len(err_neg_df[err_neg_df["clip"] == clip_name])

err_df["nb_errs"] = err_df["nb_err_pos"] + err_df["nb_err_neg"]
err_df["err_score"] = err_df["nb_errs"] / err_df["nb_samples"]
deb = err_df["err_score"].to_numpy()

err_df.to_csv("")

plt.figure(figsize=(8, 6))
plt.hist(err_df["err_score"].to_numpy(), bins=11, cumulative=False, edgecolor='black', label='err_score')
plt.xlabel('err_score')
plt.ylabel('Count')
plt.title(f'{tag} [CE, epoch {epoch}] Histogram ')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(err_df["nb_errs"].to_numpy(), bins=err_df["nb_errs"].max()+1, cumulative=False, edgecolor='black', label='nb_errs')
plt.xlabel('nb_errs')
plt.ylabel('Count')
plt.title(f'{tag} [CE, epoch {epoch}] Histogram ')
plt.legend()
plt.show()

exit(0)
print("")


# ======================================================
predictions = predictions2.format(epoch, tag)
df = pd.read_csv(predictions)
logits = torch.tensor(df[["logits_safe", "logits_risk"]].to_numpy())
probs = softmax(logits, dim=-1)
probs = probs[:, 1].numpy()
labels = df["label"].to_numpy().astype(bool)

pos_preds = probs[labels]
neg_preds = probs[~labels]

if show_hists:
    plt.figure(figsize=(8, 6))
    plt.hist([neg_preds, pos_preds], bins=101, cumulative=False, edgecolor='black', label=['neg', 'pos'])
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title(f'{tag} [Focal, epoch {epoch}] Histogram ')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(neg_preds, bins=101, cumulative=1, edgecolor='black', alpha=0.7, label='neg')
    plt.hist(pos_preds, bins=101, cumulative=-1, edgecolor='black', alpha=0.7, label='pos')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title(f'{tag} [Focal, epoch {epoch}] Cumulative histogram')
    plt.legend()
    plt.show()

# How many of negative samples are actually pre-collision samples?

print("")
