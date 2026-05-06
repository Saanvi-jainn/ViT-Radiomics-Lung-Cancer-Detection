"""
Center Loss for Feature Separation
Forces each class to cluster around its own center
Critical for medical image classification where classes overlap
"""

import torch
import torch.nn as nn
import numpy as np


class CenterLoss(nn.Module):
    """
    Center Loss implementation.
    
    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension
        alpha: Learning rate for center update (0.5 is good for medical)
    """
    
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        # Initialize centers with small values to prevent explosion
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)
        
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feat_dim]
            labels: [batch_size]
        """
        batch_size = features.size(0)
        
        # Get centers for each sample
        centers_batch = self.centers[labels]
        
        # Calculate center loss with proper scaling
        # Normalize features to prevent explosion
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        centers_norm = torch.nn.functional.normalize(centers_batch, p=2, dim=1)
        loss = nn.MSELoss()(features_norm, centers_norm)
        
        # Update centers (moving average)
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = (labels == i)
                if mask.sum() > 0:
                    class_features = features[mask]
                    # Moving average update
                    new_center = (1 - self.alpha) * self.centers[i] + self.alpha * class_features.mean(dim=0)
                    self.centers[i] = new_center
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combine CrossEntropyLoss with CenterLoss.
    
    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension  
        ce_weight: Weight for CrossEntropyLoss (default 1.0)
        center_weight: Weight for CenterLoss (default 0.01)
    """
    
    def __init__(self, num_classes, feat_dim, ce_weight=1.0, center_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(num_classes, feat_dim)
        self.ce_weight = ce_weight
        self.center_weight = center_weight
        
    def forward(self, logits, features, labels):
        """
        Args:
            logits: [batch_size, num_classes] - model outputs
            features: [batch_size, feat_dim] - features before classification
            labels: [batch_size] - ground truth
        """
        ce_loss = self.ce_loss(logits, labels)
        center_loss = self.center_loss(features, labels)
        
        total_loss = self.ce_weight * ce_loss + self.center_weight * center_loss
        
        return total_loss, ce_loss, center_loss
