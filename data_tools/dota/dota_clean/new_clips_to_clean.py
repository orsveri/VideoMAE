import os
import zipfile
import pandas as pd
from natsort import natsorted


checked_dir = "/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/checkpoint-15_OUT_train"
new_dir1 = "/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/cleaning/checkpoint-3_OUT_train_fixed04"
new_dir2 = "/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/cleaning/checkpoint-3_OUT_train_fixed03"

checked_files = os.listdir(checked_dir)
new_files1 = os.listdir(new_dir1)
new_files2 = os.listdir(new_dir2)
print(f"ALREADY checked: {len(checked_files)}\n\t- new 04: {len(new_files1)}\n\t- new 03: {len(new_files2)}")

unique_new1 = set(new_files1) - set(checked_files)
unique_new2 = set(new_files2) - set(checked_files)
intersection = set(new_files2).intersection(checked_files)
new = set(unique_new1)
new.update(unique_new2)

print(f"UNIQUE new 04: {len(unique_new1)}, new 03: {len(unique_new2)}, new: {len(new)}")


val_dir1 = "/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/cleaning/allval_ckpt-15_bad0025"
val_dir2 = "/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/cleaning/allval_ckpt-1_bad03"
val_files1 = os.listdir(val_dir1)
val_files2 = os.listdir(val_dir2)
lv1 = len(val_files1)
lv2 = len(val_files2)
intersection = set(val_files1).intersection(val_files2)
lvi = len(intersection)

print("")

