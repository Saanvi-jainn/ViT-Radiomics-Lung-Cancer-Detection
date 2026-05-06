"""
Vision Transformer model for lung cancer classification.

This module implements a Vision Transformer (ViT) architecture adapted for
CT scan image classification into Normal, Benign, and Malignant classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    
    def __init__(self, img_size=72, patch_size=6, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, img_size, img_size)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """
    Add positional encoding to patch embeddings.
    
    Args:
        embed_dim: Embedding dimension
        n_patches: Number of patches
        dropout: Dropout rate
    """
    
    def __init__(self, embed_dim=64, n_patches=144, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, n_patches, embed_dim)
        x = x + self.pos_embedding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, embed_dim=64, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, n_patches, embed_dim)
        batch_size, n_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, n_patches, embed_dim * 3)
        qkv = qkv.reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_patches, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2)  # (batch_size, n_patches, n_heads, head_dim)
        out = out.reshape(batch_size, n_patches, embed_dim)
        
        # Final projection
        out = self.projection(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention and MLP.
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        mlp_dim: Hidden dimension in MLP
        dropout: Dropout rate
    """
    
    def __init__(self, embed_dim=64, n_heads=8, mlp_dim=128, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Pre-norm residual connection for attention
        x = x + self.attention(self.ln1(x))
        
        # Pre-norm residual connection for MLP
        x = x + self.mlp(self.ln2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer model for classification.
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(self, img_size=72, patch_size=6, in_channels=3, 
                 embed_dim=64, n_heads=8, n_layers=8, n_classes=3, dropout=0.1):
        super().__init__()
        
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding and positional encoding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, self.n_patches, dropout)
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, embed_dim * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim * 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 32, n_classes)
        )
        
    def forward(self, x, return_features=False):
        # x shape: (batch_size, in_channels, img_size, img_size)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.ln(x)  # Layer normalization
        features = x.mean(dim=1)  # Global average pooling over patches
        
        if return_features:
            return features
        
        # Classification head
        logits = self.head(features)
        
        return logits
    
    def extract_features(self, x):
        """
        Extract feature embeddings before classification head.
        
        Args:
            x: Input tensor (batch_size, in_channels, img_size, img_size)
        
        Returns:
            Feature embeddings (batch_size, embed_dim)
        """
        return self.forward(x, return_features=True)


def create_vit_model(img_size=224, patch_size=16, embed_dim=256, n_heads=8, 
                    n_layers=12, n_classes=3, dropout=0.1):
    """
    Create and return a Vision Transformer model.
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        VisionTransformer model
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout=dropout
    )
    
    # Initialize weights
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    model.apply(_init_weights)
    return model


if __name__ == "__main__":
    # Test model
    model = create_vit_model()
    x = torch.randn(2, 3, 72, 72)  # Batch of 2 images
    
    print("=== Vision Transformer Feature Extraction Demo ===\n")
    
    # Test classification output
    logits = model(x)
    print(f"1. Classification Output:")
    print(f"   Input shape: {x.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extraction
    features = model.extract_features(x)
    print(f"\n2. Feature Extraction:")
    print(f"   Features shape: {features.shape}")
    print(f"   Feature dimension: {features.shape[1]}")
    
    # Test with return_features parameter
    features_alt = model(x, return_features=True)
    print(f"\n3. Alternative Feature Extraction:")
    print(f"   Features shape: {features_alt.shape}")
    print(f"   Are features equal: {torch.allclose(features, features_alt)}")
    
    print(f"\n4. Feature Vector Explanation:")
    print(f"   - Features are extracted AFTER all transformer blocks")
    print(f"   - Features are extracted BEFORE classification head")
    print(f"   - Features represent global average pooled patch embeddings")
    print(f"   - Each feature vector has shape: (batch_size, embed_dim)")
    print(f"   - Default embed_dim = 64, so features shape = (batch_size, 64)")
    
    print(f"\n=== End of Demo ===")
