import torch
import torch.nn as nn


class ConvUpsampleHead(nn.Module):
    def __init__(self, out_channels, embed_dim=384, img_size=224, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size

        # We get a feature map [B, D, 14, 14]. We'll decode it to [B, out_channels, 224, 224].
        # A simple decoder:
        # 1. Reduce dimensionality from D=384 to a smaller number if desired.
        # 2. Use a ConvTranspose2d or nn.Upsample to scale up to 224x224.

        self.decoder = nn.Sequential(
            # Start with shape [B, 384, 14, 14]
            nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample to 56x56 (4x):
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample from 56x56 to 224x224 (4x again):
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # single channel output
        )

    def forward(self, tokens):
        # tokens: [B, N, D] with N = 196 for 14x14
        B, N, D = tokens.shape
        assert N == self.num_patches_h * self.num_patches_w, \
            f"Expected {self.num_patches_h * self.num_patches_w} tokens, got {N}"

        # Reshape tokens into a feature map: [B, D, H, W]
        feature_map = tokens.transpose(1, 2).view(B, D, self.num_patches_h, self.num_patches_w)

        # Decode using convolutional layers
        depth_map = self.decoder(feature_map)
        return depth_map


class SimplePredictionHead(nn.Module):
    def __init__(self, out_channels, query_dim, num_queries):
        """
        Args:
            query_dim (int): Dimension of each query embedding (output of DETR decoder).
            num_queries (int): Number of learnable query vectors (N).
        """
        super(SimplePredictionHead, self).__init__()
        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(query_dim * num_queries, 128),  # Aggregate all queries into a smaller vector
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channels)  # Output the final scalar prediction
        )

    def forward(self, queries):
        """
        Args:
            queries (torch.Tensor): Query embeddings of shape (batch_size, num_queries, query_dim).

        Returns:
            torch.Tensor: Scalar predictions of shape (batch_size, 1).
        """
        # Flatten the queries into a single feature vector per sample
        batch_size, num_queries, query_dim = queries.size()
        flattened = queries.view(batch_size, -1)  # Shape: (batch_size, num_queries * query_dim)

        # Apply the fully connected layers
        output = self.fc(flattened)  # Shape: (batch_size, 1)
        return output


class AttentionPredictionHead(nn.Module):
    def __init__(self, out_channels, query_dim, num_queries):
        """
        Args:
            query_dim (int): Dimension of each query embedding (output of DETR decoder).
            num_queries (int): Number of learnable query vectors (N).
        """
        super(AttentionPredictionHead, self).__init__()
        self.attention_weights = nn.Linear(query_dim, 1)  # Learnable weights for each query
        self.fc = nn.Sequential(
            nn.Linear(query_dim, 128),  # Reduce to a smaller feature vector
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channels)  # Final scalar prediction
        )

    def forward(self, queries):
        """
        Args:
            queries (torch.Tensor): Query embeddings of shape (batch_size, num_queries, query_dim).

        Returns:
            torch.Tensor: Scalar predictions of shape (batch_size, 1).
        """
        # Compute attention weights for each query
        attention_scores = self.attention_weights(queries)  # Shape: (batch_size, num_queries, 1)
        attention_scores = torch.softmax(attention_scores, dim=1)  # Normalize scores across queries

        # Apply attention to the queries
        weighted_queries = (queries * attention_scores).sum(dim=1)  # Shape: (batch_size, query_dim)

        # Apply the fully connected layers
        output = self.fc(weighted_queries)  # Shape: (batch_size, 1)
        return output


# Example usage:
if __name__ == "__main__":
    B = 2
    embed_dim = 384
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) * (img_size // patch_size)  # 14*14=196

    tokens = torch.randn(B, num_patches, embed_dim)  # [B, 196, 384]
    head = ConvUpsampleHead(embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
    depth_map = head(tokens)
    print("Depth map shape:", depth_map.shape)  # Expected: [B, 1, 224, 224]
