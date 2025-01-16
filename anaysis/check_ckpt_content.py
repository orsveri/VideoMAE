import os
import torch
import json

ckpt1 = "logs/pretrained/k400_vits/videomae_vits_k400_pretrain_ckpt.pth"
ckpt2 = "logs/pretrained/distill/videomae_vits_k710_distill_from_giant.pth"
k1 = "model"
k2 = "module"

ckpt = ckpt1
k = k1

ckpt_dict = torch.load(ckpt, map_location='cpu')
# print(ckpt_dict.keys())
# exit(0)
ckpt_dict = ckpt_dict[k]  # model or module
ckpt_dict = list(ckpt_dict.keys())

with open(os.path.splitext(ckpt)[0] + ".json", "w") as f:
    json.dump({k: ckpt_dict}, f, indent=2)

print("Done!")

