"""
Complete multimodal fusion model with attention-based fusion.
Combines ViT visual features with radiomics features for improved accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_model import VisionTransformer
from typing import Tuple, Optional
import numpy as np


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for feature fusion.
    """
    
    def __init__(self, vit_dim: int, rad_dim: int, hidden_dim: int = 256):
        super(CrossModalAttention, self).__init__()
        
        self.vit_dim = vit_dim
        self.rad_dim = rad_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers - dynamic for pre-extracted features
        self.vit_to_rad = nn.Linear(vit_dim, hidden_dim)  # Dynamic for pre-extracted ViT features
        self.rad_to_vit = nn.Linear(rad_dim, hidden_dim)
        
        self.attention_vit = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.attention_rad = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.output_vit = nn.Linear(hidden_dim, vit_dim)
        self.output_rad = nn.Linear(hidden_dim, rad_dim)
        
    def forward(self, vit_features: torch.Tensor, rad_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross-modal attention.
        
        Args:
            vit_features: Visual features (batch_size, vit_dim)
            rad_features: Radiomics features (batch_size, rad_dim)
            
        Returns:
            Tuple of (enhanced_vit_features, enhanced_rad_features)
        """
        # Project to hidden dimension - simplified for pre-extracted features
        vit_proj = self.vit_to_rad(vit_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        rad_proj = self.rad_to_vit(rad_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Cross-modal attention
        vit_attended, _ = self.attention_vit(vit_proj, rad_proj, rad_proj)
        rad_attended, _ = self.attention_rad(rad_proj, vit_proj, vit_proj)
        
        # Project back to original dimensions
        enhanced_vit = self.output_vit(vit_attended.squeeze(1))
        enhanced_rad = self.output_rad(rad_attended.squeeze(1))
        
        return enhanced_vit, enhanced_rad


class MultimodalFusionComplete(nn.Module):
    """
    Complete multimodal fusion model with attention-based fusion.
    """
    
    def __init__(self, 
                 vit_config: dict = None,
                 rad_dim: int = 100,
                 num_classes: int = 3,
                 fusion_type: str = 'attention'):
        super(MultimodalFusionComplete, self).__init__()
        
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        
        # Vision Transformer
        if vit_config is None:
            vit_config = {
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 384,
                'n_heads': 12,
                'n_layers': 16,
                'n_classes': 3,
                'dropout': 0.3
            }
        
        self.vit_model = VisionTransformer(**vit_config)
        self.vit_dim = vit_config['embed_dim']
        
        # Remove classification head from ViT
        self.vit_model.classifier = nn.Identity()
        
        # Radiomics feature processing
        self.rad_dim = rad_dim
        self.rad_encoder = nn.Sequential(
            nn.Linear(rad_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion mechanism
        if fusion_type == 'attention':
            self.fusion_layer = CrossModalAttention(self.vit_dim, 64, hidden_dim=256)
            fused_dim = self.vit_dim + 64  # Concatenation after attention
        elif fusion_type == 'concat':
            fused_dim = self.vit_dim + 64  # 384 + 64 = 448
        else:  # gated fusion
            self.gate_vit = nn.Linear(self.vit_dim, 1)
            self.gate_rad = nn.Linear(64, 1)
            fused_dim = self.vit_dim + 64
        
        # Classification layers - minimal architecture
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, vit_features, radiomics_features):
        """
        Forward pass through multimodal fusion model.
        
        Args:
            vit_features: Pre-extracted ViT features (batch_size, vit_dim)
            radiomics_features: Radiomics features (batch_size, n_radiomics)
            
        Returns:
            Classification logits
        """
        # Project radiomics features
        rad_features = self.rad_encoder(radiomics_features)  # (batch_size, 64)
        
        # Fusion
        if self.fusion_type == 'attention':
            enhanced_vit, enhanced_rad = self.fusion_layer(vit_features, rad_features)
            fused_features = torch.cat([enhanced_vit, enhanced_rad], dim=1)
        elif self.fusion_type == 'concat':
            fused_features = torch.cat([vit_features, rad_features], dim=1)
        else:  # gated fusion
            gate_vit = torch.sigmoid(self.gate_vit(vit_features))
            gate_rad = torch.sigmoid(self.gate_rad(rad_features))
            fused_features = gate_vit * vit_features + gate_rad * rad_features
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, fused_features
    
    def forward_logits_only(self, vit_features, radiomics_features):
        """
        Forward pass returning only logits (for evaluation).
        """
        logits, _ = self.forward(vit_features, radiomics_features)
        return logits
    
    def get_attention_weights(self, vit_features: torch.Tensor, radiomics_features: torch.Tensor) -> dict:
        """
        Get attention weights for explainability.
        
        Args:
            vit_features: Pre-extracted ViT features
            radiomics_features: Radiomics features
            
        Returns:
            Dictionary containing attention weights
        """
        self.eval()
        with torch.no_grad():
            # For pre-extracted features, just return the inputs
            vit_features = vit_features
            rad_features = self.rad_encoder(radiomics_features)
            
            if self.fusion_type == 'attention':
                enhanced_vit, enhanced_rad = self.fusion_layer(vit_features, rad_features)
                
                return {
                    'vit_features': vit_features,
                    'rad_features': rad_features,
                    'enhanced_vit': enhanced_vit,
                    'enhanced_rad': enhanced_rad
                }
            else:
                return {
                    'vit_features': vit_features,
                    'rad_features': rad_features
                }


def create_multimodal_model(vit_config: dict = None, 
                           rad_dim: int = 100, 
                           num_classes: int = 3,
                           fusion_type: str = 'attention') -> MultimodalFusionComplete:
    """
    Create a multimodal fusion model.
    
    Args:
        vit_config: Vision Transformer configuration
        rad_dim: Dimension of radiomics features
        num_classes: Number of output classes
        fusion_type: Type of fusion ('attention', 'concat', 'gated')
        
    Returns:
        Multimodal fusion model
    """
    model = MultimodalFusionComplete(
        vit_config=vit_config,
        rad_dim=rad_dim,
        num_classes=num_classes,
        fusion_type=fusion_type
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Multimodal model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Fusion type: {fusion_type}")
    print(f"  Radiomics dimension: {rad_dim}")
    
    return model


if __name__ == "__main__":
    # Test multimodal fusion model
    print("Testing Multimodal Fusion Model...")
    
    # Create model
    model = create_multimodal_model(rad_dim=100, fusion_type='attention')
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    radiomics = torch.randn(batch_size, 100)
    
    logits = model(images, radiomics)
    
    print(f"Input shapes: images={images.shape}, radiomics={radiomics.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(images, radiomics)
    print(f"Attention keys: {list(attention_weights.keys())}")
    
    print("Multimodal fusion model test completed!")
