import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
import modeling_finetune
from InternVideo2_single_modality.models import internvideo2


steps = 100
x_ = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
x2_ = np.random.randn(1, 3, 8, 224, 224).astype(np.float32)

device = torch.device("cuda")

# VideoMAE ViT-S
model = create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=2,
        all_frames=16,
        tubelet_size=2,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_checkpoint=False,
        final_reduction="fc_norm",
        init_scale=0.001,
        use_flash_attn=False
    )

model.to(device)
x = torch.tensor(x_).to(device)
model.eval()

forward_times = []

print(f"[NOGRD] gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} memory\n")

with torch.no_grad():
    ave_start=time.time()
    for i in range(steps):
        start = time.time()
        with torch.cuda.amp.autocast():
            out = model(x)
        end = time.time()
        t = end - start
        print(f'step {t}: {t} s')
        print(f"\t[NOGRD] gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} memory")
        torch.cuda.reset_peak_memory_stats(device=None)
        forward_times.append(t)

avg_time = sum(forward_times) / steps
print(f'Average time: {avg_time} s')
print(f'Average FPS: {1 / avg_time}')



