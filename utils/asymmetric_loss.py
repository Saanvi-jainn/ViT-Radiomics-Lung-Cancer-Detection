"""
Asymmetric Loss for Medical Classification
Heavily penalizes false negatives (missing malignant cases)
Critical for cancer detection where missing cancer is worse than false alarms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes false negatives more than false positives.
    
    Args:
        gamma_neg: Focusing parameter for negative class (malignant)
        gamma_pos: Focusing parameter for positive class (benign)
        clip: Clipping value to prevent overflow
        disable_grad: Whether to disable gradient for positive class
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, disable_grad=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_grad = disable_grad
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, num_classes] - model outputs
            labels: [batch_size] - ground truth (0=benign, 1=malignant)
        """
        # Convert labels to one-hot
        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate asymmetric loss
        loss = 0
        
        for i in range(logits.size(1)):  # For each class
            # For malignant class (class 1), apply asymmetric penalty
            if i == 1:  # Malignant class
                # Heavily penalize false negatives (missing malignant)
                pt = (1 - probs[:, i]) * labels_one_hot[:, i] + probs[:, i] * (1 - labels_one_hot[:, i])
                alpha = labels_one_hot[:, i] * 2.0 + (1 - labels_one_hot[:, i]) * 1.0  # Higher weight for malignant
                gamma = self.gamma_neg * labels_one_hot[:, i] + self.gamma_pos * (1 - labels_one_hot[:, i])
                
                if self.disable_grad:
                    with torch.no_grad():
                        if self.clip > 0:
                            pt = torch.clamp(pt, self.clip, 1 - self.clip)
                
                loss += -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8)
            else:
                # Standard cross-entropy for benign class
                loss += F.cross_entropy(logits[:, i:i+1], labels_one_hot[:, i:i+1])
        
        return loss.mean()


class WeightedAsymmetricLoss(nn.Module):
    """
    Combined asymmetric loss with class weights.
    """
    
    def __init__(self, malignant_weight=3.0, benign_weight=1.0, gamma_neg=4):
        super(WeightedAsymmetricLoss, self).__init__()
        self.malignant_weight = malignant_weight
        self.benign_weight = benign_weight
        self.gamma_neg = gamma_neg
        
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, 2] - model outputs
            labels: [batch_size] - ground truth (0=benign, 1=malignant)
        """
        # Standard cross entropy with weights
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Apply asymmetric weights
        weights = torch.where(labels == 1,  # Malignant
                             torch.tensor(self.malignant_weight, device=logits.device),
                             torch.tensor(self.benign_weight, device=logits.device))
        
        weighted_ce = (ce_loss * weights).mean()
        
        # Additional penalty for false negatives (malignant predicted as benign)
        probs = F.softmax(logits, dim=1)
        malignant_probs = probs[:, 1]  # Probability of malignant
        
        # False negative penalty: high penalty when malignant has low probability
        fn_penalty = torch.where(
            labels == 1,  # Actual malignant
            torch.exp(-self.gamma_neg * malignant_probs),  # Penalty increases as prob decreases
            torch.tensor(0.0, device=logits.device)  # No penalty for benign cases
        ).mean()
        
        return weighted_ce + 0.1 * fn_penalty


class FocalAsymmetricLoss(nn.Module):
    """
    Focal loss with asymmetric weighting for medical classification.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, malignant_alpha=0.75):
        super(FocalAsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.malignant_alpha = malignant_alpha
        
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, 2] - model outputs
            labels: [batch_size] - ground truth (0=benign, 1=malignant)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Asymmetric alpha: higher weight for malignant class
        alpha_t = torch.where(
            labels == 1,
            torch.tensor(self.malignant_alpha, device=logits.device),
            torch.tensor(self.alpha, device=logits.device)
        )
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


def create_asymmetric_criterion(loss_type='weighted', **kwargs):
    """
    Create asymmetric loss criterion.
    
    Args:
        loss_type: 'asymmetric', 'weighted', 'focal'
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss function
    """
    if loss_type == 'asymmetric':
        return AsymmetricLoss(**kwargs)
    elif loss_type == 'weighted':
        return WeightedAsymmetricLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalAsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
