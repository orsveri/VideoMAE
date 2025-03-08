import matplotlib.pyplot as plt

# Data from the table
variants = {
    1: {"DoTA_ap": 81.35, "DoTA_auroc": 85.75, "DADA2K_ap": 76.88, "DADA2K_auroc": 82.60, "Label": "DoTA"},
    2: {"DoTA_ap": 79.18, "DoTA_auroc": 84.33, "DADA2K_ap": 75.62, "DADA2K_auroc": 82.20, "Label": "DoTA 1/2"},
    3: {"DoTA_ap": 73.27, "DoTA_auroc": 80.55, "DADA2K_ap": 77.18, "DADA2K_auroc": 82.95, "Label": "DADA2K"},
    4: {"DoTA_ap": 72.16, "DoTA_auroc": 79.24, "DADA2K_ap": 75.18, "DADA2K_auroc": 81.35, "Label": "DADA2K 1/2"},
    13: {"DoTA_ap": 83.11, "DoTA_auroc": 87.06, "DADA2K_ap": 79.63, "DADA2K_auroc": 84.92},
    14: {"DoTA_ap": 81.00, "DoTA_auroc": 85.60, "DADA2K_ap": 77.42, "DADA2K_auroc": 83.27},
    15: {"DoTA_ap": 75.55, "DoTA_auroc": 81.86, "DADA2K_ap": 80.03, "DADA2K_auroc": 85.43},
    16: {"DoTA_ap": 72.80, "DoTA_auroc": 80.17, "DADA2K_ap": 75.96, "DADA2K_auroc": 82.12},
    5: {"DoTA_ap": 82.15, "DoTA_auroc": 86.74, "DADA2K_ap": 77.15, "DADA2K_auroc": 83.41},
    6: {"DoTA_ap": 80.18, "DoTA_auroc": 85.26, "DADA2K_ap": 76.84, "DADA2K_auroc": 81.89},
    7: {"DoTA_ap": 73.91, "DoTA_auroc": 81.02, "DADA2K_ap": 79.02, "DADA2K_auroc": 84.84},
    8: {"DoTA_ap": 73.01, "DoTA_auroc": 80.58, "DADA2K_ap": 76.97, "DADA2K_auroc": 83.01},
}

arrow_w = 0.15
arrow_h = 0.15
# Colors for each arrow set
colors = ['blue', 'orange', 'green', 'red']

# Plot for DoTA
plt.figure(figsize=(8, 6))
for (i, j, k), color in zip([(1, 13, 5), (2, 14, 6), (3, 15, 7), (4, 16, 8)], colors):
    # Main arrow (solid line with arrowhead)
    x_values = [variants[i]["DoTA_ap"], variants[j]["DoTA_ap"]]
    y_values = [variants[i]["DoTA_auroc"], variants[j]["DoTA_auroc"]]
    plt.plot(x_values, y_values, marker="o", color=color, label=variants[i]["Label"])
    plt.arrow(
        x_values[0], y_values[0], x_values[1] - x_values[0], y_values[1] - y_values[0],
        head_width=arrow_w, head_length=arrow_h, fc=color, ec=color
    )

    # Secondary branch (punctured line from start, no arrowhead)
    x_values_punctured = [variants[k]["DoTA_ap"], variants[i]["DoTA_ap"]]
    y_values_punctured = [variants[k]["DoTA_auroc"], variants[i]["DoTA_auroc"]]
    plt.plot(x_values_punctured, y_values_punctured, linestyle='--', color=color)

plt.title("DoTA: Improvement After Extra Pretraining", fontsize=14)
plt.xlabel("AP", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
plt.savefig("dota_extra.png", dpi=300, bbox_inches="tight")

# Plot for DADA-2000
plt.figure(figsize=(8, 6))
for (i, j, k), color in zip([(1, 13, 5), (2, 14, 6), (3, 15, 7), (4, 16, 8)], colors):
    # Main arrow (solid line with arrowhead)
    x_values = [variants[i]["DADA2K_ap"], variants[j]["DADA2K_ap"]]
    y_values = [variants[i]["DADA2K_auroc"], variants[j]["DADA2K_auroc"]]
    plt.plot(x_values, y_values, marker="o", color=color, label=variants[i]["Label"])
    plt.arrow(
        x_values[0], y_values[0], x_values[1] - x_values[0], y_values[1] - y_values[0],
        head_width=arrow_w, head_length=arrow_h, fc=color, ec=color
    )

    # Secondary branch (punctured line from start, no arrowhead)
    x_values_punctured = [variants[k]["DADA2K_ap"], variants[i]["DADA2K_ap"]]
    y_values_punctured = [variants[k]["DADA2K_auroc"], variants[i]["DADA2K_auroc"]]
    plt.plot(x_values_punctured, y_values_punctured, linestyle='--', color=color)

plt.title("DADA-2000: Improvement After Extra Pretraining", fontsize=14)
plt.xlabel("AP", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
plt.savefig("dada_extra.png", dpi=300, bbox_inches="tight")
