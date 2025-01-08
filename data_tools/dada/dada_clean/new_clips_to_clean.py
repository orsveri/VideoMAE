import os
import zipfile
import pandas as pd
from natsort import natsorted


errs1_path = "/home/sorlova/repos/NewStart/VideoMAE/logs/clean_datasets/DADA2K/b32x2x1gpu_ce_TRAIN/checkpoint-15/OUTtrain/err_report_bad0.012.csv"
errs2_path = "/home/sorlova/repos/NewStart/VideoMAE/logs/clean_datasets/DADA2K/b32x2x1gpu_ce_TRAIN/checkpoint-1/OUTtrain/err_report_bad0.2.csv"

errs1 = pd.read_csv(errs1_path)["clip"].tolist()
errs2 = pd.read_csv(errs2_path)["clip"].tolist()

intersection = set(errs2).intersection(errs1)

new_unique = natsorted(list(set(errs2) - set(errs1)))

print(new_unique)
print("")

