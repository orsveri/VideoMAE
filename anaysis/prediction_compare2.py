import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax


# Prediction (model output) as a tensor
prediction = torch.tensor([0.5], dtype=torch.float32)  # Predicted probability
target = torch.tensor([1.0], dtype=torch.float32)  # Ground truth label

# Avoid log(0) by clipping prediction
epsilon = 1e-7
prediction_clipped = torch.clamp(prediction, epsilon, 1.0)

# Alternatively, use PyTorch's BCE function
bce_loss_builtin = torch.nn.functional.binary_cross_entropy(prediction_clipped, target)
print(f"Binary Cross-Entropy Loss (Builtin): {bce_loss_builtin.item()}")

predictions = "/home/sorlova/repos/NewStart/VideoMAE/logs/dota_fixloss/focal_1gpu/OUT_DoTA/predictions_0.csv"

df = pd.read_csv(predictions)
logits = torch.tensor(df[["logits_safe", "logits_risk"]].to_numpy())
probs = softmax(logits, dim=-1)
probs = probs[:, 1].numpy()
labels = df["label"].to_numpy().astype(bool)

pos_preds = probs[labels]
neg_preds = probs[~labels]

plt.figure(figsize=(8, 6))
plt.hist([neg_preds, pos_preds], bins=101, cumulative=False, edgecolor='black', label=['neg', 'pos'])
plt.xlabel('Probability')
plt.ylabel('Count')
plt.title('Histogram for neg and pos samples')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(pos_preds, bins=101, cumulative=-1, edgecolor='black', alpha=0.7, label='pos')
plt.hist(neg_preds, bins=101, cumulative=1, edgecolor='black', alpha=0.7, label='neg')
plt.xlabel('Probability')
plt.ylabel('Count')
plt.title('REVERSE Cumulative histogram for pos samples')
plt.show()

# How many of negative samples are actually pre-collision samples?

print("")
