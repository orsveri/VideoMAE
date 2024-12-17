import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from decoders.heads import ConvUpsampleHead


class DETRDecoder(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False):
        """
        A DETR-like Transformer decoder.

        Args:
            d_model (int): Dimension of the model (input feature size).
            nhead (int): Number of attention heads.
            num_decoder_layers (int): Number of Transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward networks.
            dropout (float): Dropout probability.
            normalize_before (bool): Whether to pre-normalize in Transformer layers.
        """
        super().__init__()

        # Build decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        )
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, features, tgt_mask=None, features_mask=None,
                tgt_key_padding_mask=None, features_key_padding_mask=None,
                pos_embed=None, query_pos=None):
        """
        Forward pass of the DETR decoder.

        Args:
            tgt (Tensor): Target sequence embeddings (queries), shape [batch_size, num_queries, d_model].
            features (Tensor): Output from the encoder, shape [batch_size, seq_length, d_model].
            tgt_mask (Tensor, optional): Mask for the target sequence (queries).
            features_mask (Tensor, optional): Mask for the features (encoder output).
            tgt_key_padding_mask (Tensor, optional): Padding mask for queries.
            features_key_padding_mask (Tensor, optional): Padding mask for features.
            pos_embed (Tensor, optional): Positional encodings for the features, shape [batch_size, seq_length, d_model].
            query_pos (Tensor, optional): Positional encodings for the queries, shape [batch_size, num_queries, d_model].

        Returns:
            Tensor: Decoded features for each query, shape [batch_size, num_queries, d_model].
        """

        # Add positional embeddings if given
        # In DETR, pos_embed is added to the features (encoder features)
        # and query_pos is added to the tgt (queries) before each attention.
        # If pos_embed is None or query_pos is None, it will just skip adding them.
        if pos_embed is not None:
            features = features + pos_embed
        if query_pos is not None:
            tgt = tgt + query_pos

        # Run the Transformer decoder
        # shape of output: [batch_size, num_queries, d_model]
        decoded = self.decoder(tgt, features,
                               tgt_mask=tgt_mask,
                               features_mask=features_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               features_key_padding_mask=features_key_padding_mask)
        return decoded


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, features,
                tgt_mask: Optional[Tensor] = None,
                features_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                features_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, features, tgt_mask=tgt_mask,
                           features_mask=features_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           features_key_padding_mask=features_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, features,
                     tgt_mask: Optional[Tensor] = None,
                     features_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     features_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(features, pos),
                                   value=features, attn_mask=features_mask,
                                   key_padding_mask=features_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, features,
                    tgt_mask: Optional[Tensor] = None,
                    features_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    features_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(features, pos),
                                   value=features, attn_mask=features_mask,
                                   key_padding_mask=features_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, features,
                tgt_mask: Optional[Tensor] = None,
                features_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                features_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, features, tgt_mask, features_mask,
                                    tgt_key_padding_mask, features_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, features, tgt_mask, features_mask,
                                 tgt_key_padding_mask, features_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



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

    # Replace the final classification head with identity since we only want features
    encoder.head = nn.Identity()

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
    num_queries = num_img_patches
    d_model = features.size(-1)
    query_embed = nn.Parameter(torch.randn(num_queries, d_model))

    decoder = DETRDecoder(
        d_model=d_model,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False
    )

    # We will create queries for the decoder by expanding our query embeddings to match the batch size
    tgt = query_embed.unsqueeze(0).expand(B, num_queries, d_model).clone()

    print(f"Passing data through decoder... (learnable queries shape: {tgt.shape})")

    # Forward through the DETR decoder
    # For simplicity, we are not using positional encodings or masks here
    out = decoder(
        tgt=tgt,
        features=features,
        tgt_mask=None,
        features_mask=None,
        tgt_key_padding_mask=None,
        features_key_padding_mask=None,
        pos_embed=None,
        query_pos=None
    )

    print("Decoder output shape:", out[0].shape)  # Should be [B, num_queries, d_model]

    tokens = out[0]  # [B, 196, 384]
    head1 = ConvUpsampleHead(out_channels=1, embed_dim=embed_dim, img_size=H, patch_size=patch_size)
    head2 = ConvUpsampleHead(out_channels=2, embed_dim=embed_dim, img_size=H, patch_size=patch_size)

    print(f"Passing data through depth head... ")
    depth_map = head1(tokens)
    print("Depth map shape:", depth_map.shape)  # Expected: [B, 1, 224, 224]

    print(f"Passing data through flow head... ")
    flow_map = head2(tokens)
    print("Depth map shape:", flow_map.shape)  # Expected: [B, 2, 224, 224]


