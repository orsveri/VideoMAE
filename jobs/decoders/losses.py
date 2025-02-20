"""
Based on:
depth: https://github.com/antocad/FocusOnDepth/blob/master/FOD/Loss.py
optical flow:

"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from decoders.heads import ConvUpsampleHead, SimplePredictionHead, AttentionPredictionHead
from decoders.detr.detr_from_scratch import DETRDecoder



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero(as_tuple=False)

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target):
        # we don't need that singleton dimension in target
        target = torch.squeeze(target)
        prediction = torch.squeeze(prediction)

        #preprocessing
        mask = target > 0

        #calcul
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        # print(scale, shift)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


if __name__ == "__main__":
    from timm.models import create_model
    import modeling_finetune  # for registering our Video VisionTransformer

    # Video shape expected: [B, C, T, H, W]
    B, C, T, H, W = 8, 3, 16, 224, 224
    patch_size = 16
    tokens_side = H // patch_size
    embed_dim = 384  # ViT-small
    num_img_patches = (W // patch_size) * (H // patch_size)  # 14*14=196

    # 1. Load the pretrained VideoMAE ViT-small encoder
    encoder = create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=0, # no classification head needed
        all_frames=16,
        tubelet_size=2,
        fc_drop_rate=0.,
        drop_rate=0.,
        drop_path_rate=0.1,
        attn_drop_rate=0.,
        drop_block_rate=None,
        use_checkpoint=False,
        final_reduction='none',
        init_scale=0.001,
    )

    # Load the weights
    checkpoint = torch.load("/home/sorlova/repos/NewStart/VideoMAE/logs/pretrained/distill/vit_s_k710_dl_from_giant.pth", map_location="cpu")
    encoder.load_state_dict(checkpoint, strict=False)

    # 2. Create a random input sample
    x = torch.randn(B, C, T, H, W)

    print("Passing data through encoder... ")

    # Forward through encoder to get features
    with torch.no_grad():
        # forward_features returns a set of features of shape [B, N, D], i.e. [B, 1568, 384]
        # N is the number of patches * temporal segments
        features = encoder.forward_features(x)
        deb = encoder(x)

    print("Encoder output shape (features):", features.shape)  # e.g. [2, N, 384] for ViT-small

    # 3. Setup DETR decoder
    num_queries_depth = num_img_patches
    num_queries_flow = num_img_patches
    num_queries_risk = 1
    query_embed_depth = nn.Parameter(torch.randn(num_queries_depth, embed_dim))
    query_embed_flow = nn.Parameter(torch.randn(num_queries_flow, embed_dim))
    query_embed_risk = nn.Parameter(torch.randn(num_queries_risk, embed_dim))

    decoder_depth = DETRDecoder(
        d_model=embed_dim,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False
    )
    head_depth = ConvUpsampleHead(out_channels=1, embed_dim=embed_dim, img_size=H, patch_size=patch_size)

    decoder_flow = DETRDecoder(
        d_model=embed_dim,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False
    )
    head_flow = ConvUpsampleHead(out_channels=2, embed_dim=embed_dim, img_size=H, patch_size=patch_size)

    decoder_risk = DETRDecoder(
        d_model=embed_dim,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False
    )
    head_risk = SimplePredictionHead(out_channels=2, query_dim=embed_dim, num_queries=num_queries_risk)
    #head_risk = AttentionPredictionHead(out_channels=2, query_dim=embed_dim, num_queries=num_queries_risk)

    # We will create queries for the decoder by expanding our query embeddings to match the batch size
    tgt_depth = query_embed_depth.unsqueeze(0).expand(B, num_queries_depth, embed_dim).clone()
    tgt_flow = query_embed_flow.unsqueeze(0).expand(B, num_queries_flow, embed_dim).clone()
    tgt_risk = query_embed_risk.unsqueeze(0).expand(B, num_queries_risk, embed_dim).clone()

    # Forward through the DETR decoder
    # For simplicity, we are not using positional encodings or masks here
    out_depth = decoder_depth(
        tgt=tgt_depth,
        features=features,
        tgt_mask=None,
        features_mask=None,
        tgt_key_padding_mask=None,
        features_key_padding_mask=None,
        pos_embed=None,
        query_pos=None
    )
    out_flow = decoder_flow(
        tgt=tgt_flow,
        features=features,
        tgt_mask=None,
        features_mask=None,
        tgt_key_padding_mask=None,
        features_key_padding_mask=None,
        pos_embed=None,
        query_pos=None
    )
    out_risk = decoder_risk(
        tgt=tgt_risk,
        features=features,
        tgt_mask=None,
        features_mask=None,
        tgt_key_padding_mask=None,
        features_key_padding_mask=None,
        pos_embed=None,
        query_pos=None
    )

    print("Decoder output shape:",  # Should be [B, 1, num_queries, d_model]
    f"\n\tdepth: {out_depth.shape})\n\tflow: {out_flow.shape}\n\trisk: {out_risk.shape}")

    tokens_depth, tokens_flow, tokens_risk = out_depth[0], out_flow[0], out_risk[0]  # [B, 196, 384]

    print(f"Passing data through depth head... ")
    depth_map = head_depth(tokens_depth)
    print("Depth map shape:", depth_map.shape)  # Expected: [B, 1, 224, 224]

    print(f"Passing data through flow head... ")
    flow_map = head_flow(tokens_flow)
    print("Depth map shape:", flow_map.shape)  # Expected: [B, 2, 224, 224]

    print(f"Passing data through risk head... ")
    risk = head_risk(tokens_risk)
    print("Risk map shape:", risk.shape)  # Expected: [B, 1]

    # -------------------------
    # Create an optimizer for decoder, heads, and query_embed
    params = [query_embed_depth, query_embed_flow, query_embed_risk] + \
             list(decoder_depth.parameters()) + list(head_depth.parameters()) + \
             list(decoder_flow.parameters()) + list(head_flow.parameters()) + \
             list(decoder_risk.parameters()) #+ list(head_risk.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)
    ssiloss = ScaleAndShiftInvariantLoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    # Dummy targets:
    # Depth target: [B, 1, 224, 224], Flow target: [B, 2, 224, 224]
    depth_target = torch.randn(B, 1, H, W)
    flow_target = torch.randn(B, 2, H, W)
    risk_target = (torch.randn(B) >= 0).long()
    print(f"risk gt: {risk_target}\n")

    # Simple training loop for 100 iterations
    for iteration in range(100):
        # Forward pass
        x = torch.randn(B, C, T, H, W)
        with torch.no_grad():
            features = encoder.forward_features(x)
        # Create queries expanded for batch
        tgt_depth = query_embed_depth.unsqueeze(0).expand(B, num_queries_depth, embed_dim).clone()
        tgt_flow = query_embed_flow.unsqueeze(0).expand(B, num_queries_flow, embed_dim).clone()
        tgt_risk = query_embed_risk.unsqueeze(0).expand(B, num_queries_risk, embed_dim).clone()
        out_depth = decoder_depth(
            tgt=tgt_depth,
            features=features,
            tgt_mask=None,
            features_mask=None,
            tgt_key_padding_mask=None,
            features_key_padding_mask=None,
            pos_embed=None,
            query_pos=None
        )
        out_flow = decoder_flow(
            tgt=tgt_flow,
            features=features,
            tgt_mask=None,
            features_mask=None,
            tgt_key_padding_mask=None,
            features_key_padding_mask=None,
            pos_embed=None,
            query_pos=None
        )
        out_risk = decoder_risk(
            tgt=tgt_risk,
            features=features,
            tgt_mask=None,
            features_mask=None,
            tgt_key_padding_mask=None,
            features_key_padding_mask=None,
            pos_embed=None,
            query_pos=None
        )

        tokens_depth, tokens_flow, tokens_risk = out_depth[0], out_flow[0], out_risk[0]
        depth_map = head_depth(tokens_depth)  # [B, 1, 224, 224]
        flow_map = head_flow(tokens_flow)  # [B, 2, 224, 224]
        risk = head_risk(tokens_risk)  # [B, 1]

        # Compute loss (L1 loss for both)
        loss_depth = ssiloss(depth_map, depth_target)
        loss_flow = F.l1_loss(flow_map, flow_target)
        loss_risk = ce_loss(risk, risk_target)
        loss = loss_depth + loss_flow + loss_risk

        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/100, Loss: {loss.item():.4f}\n"
                  f"\t\tloss depth: {loss_depth.item():.4f}, loss flow: {loss_flow.item():.4f}, loss risk: {loss_risk.item():.4f}")



