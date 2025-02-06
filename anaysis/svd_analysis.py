import torch
import numpy as np


def run_svd_on_q_weights(ckpt, num_layers=12, num_heads=6):
    """
    Example function that:
      1. Extracts Q weight matrix per layer+head
      2. Performs SVD on each head's Q
      3. Prints the top singular values

    We assume a structure similar to your collect_grad_norms, where:
      block.attn.qkv.weight is [3 * embed_dim, embed_dim]
      -> we slice out Q, which is shape [embed_dim, embed_dim].
      Then we slice that [embed_dim, embed_dim] further for each head.

    This is just a demonstration. Adapt as needed.
    """
    for layer_idx in range(12):
        # The shape is [3*embed_dim, embed_dim]
        qkv_weight = ckpt[f"blocks.{layer_idx}.attn.qkv.weight"]
        embed_dim = qkv_weight.shape[1]
        head_dim = (embed_dim // num_heads)

        # Reshape to [3, num_heads, head_dim, embed_dim]
        qkv_view = qkv_weight.view(3, num_heads, head_dim, embed_dim)
        # Q is index 0 in the first dimension
        q_view = qkv_view[0]  # shape [num_heads, head_dim, embed_dim]

        # We'll do per-head SVD: q_view[head_idx] => [head_dim, embed_dim]
        # we might want the transpose, depending on how we interpret rows vs cols.
        for head_idx in range(num_heads):
            q_matrix = q_view[head_idx]  # shape [head_dim, embed_dim]

            # Convert to CPU numpy for SVD
            q_matrix_np = q_matrix.cpu().numpy().astype(np.float32)

            # If we want [embed_dim, head_dim], we might transpose:
            # q_matrix_np = q_matrix_np.T

            U, S, Vt = np.linalg.svd(q_matrix_np, full_matrices=False)
            # S is a 1D array of singular values in descending order

            # Print top singular values
            top_svals = S[:5]  # get first 5 as an example
            print(f"Layer {layer_idx}, Head {head_idx}, top 5 singular values: {top_svals}")

            # or store them in a data structure for further analysis



ckpt_f = "/home/sorlova/repos/AITHENA/NewStage/VideoMAE/logs/baselines/bl1/1_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/checkpoint-16.pth"

ckpt = torch.load(ckpt_f, map_location='cpu')
run_svd_on_q_weights(ckpt)

