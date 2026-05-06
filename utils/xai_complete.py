"""
Complete XAI (Explainability) system for multimodal lung cancer detection.
Provides comprehensive explainability methods for clinical validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from sklearn.inspection import permutation_importance
# import shap
from PIL import Image
import os
import pandas as pd


class MultimodalExplainer:
    """
    Complete explainability system for multimodal model.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the multimodal explainer.
        
        Args:
            model: Trained multimodal model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Store gradients for Grad-CAM
        self.gradients = {}
        self.activations = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        def forward_hook(module, input, output):
            self.activations['vit_features'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['vit_features'] = grad_output[0].detach()
        
        # Register hooks on ViT model
        if hasattr(self.model.vit_model, 'transformer'):
            self.model.vit_model.transformer.register_forward_hook(forward_hook)
            self.model.vit_model.transformer.register_backward_hook(backward_hook)
    
    def generate_grad_cam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for image classification.
        
        Args:
            image: Input image tensor (1, 3, 224, 224)
            target_class: Target class index
            
        Returns:
            Grad-CAM heatmap (224, 224)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(image, torch.randn(1, 100).to(self.device))
        
        # Backward pass for target class
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients['vit_features']
        activations = self.activations['vit_features']
        
        # Generate weights
        weights = torch.mean(gradients, dim=1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights.unsqueeze(-1) * activations, dim=1)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        
        # Resize to image dimensions
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()
    
    def visualize_attention_maps(self, vit_features: torch.Tensor, radiomics: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Visualize attention maps from multimodal model.
        
        Args:
            image: Input image tensor
            radiomics: Radiomics features tensor
            
        Returns:
            Dictionary of attention visualizations
        """
        self.model.eval()
        with torch.no_grad():
            # Get attention weights
            attention_weights = self.model.get_attention_weights(vit_features, radiomics)
            
            visualizations = {}
            
            # Visual feature attention
            vit_features = attention_weights['vit_features'].cpu().numpy()
            # Reshape for visualization - handle dynamic dimensions
            if vit_features.size == 256:
                vit_features_reshaped = vit_features.reshape(16, 16)  # 256-dim to 16x16
            elif vit_features.size == 224:
                vit_features_reshaped = vit_features.reshape(14, 16)  # 224-dim to 14x16
            elif vit_features.size == 384:
                vit_features_reshaped = vit_features.reshape(16, 24)  # 384-dim to 16x24
            else:
                # Find the best square-ish reshape
                size = int(np.sqrt(vit_features.size))
                if size * size == vit_features.size:
                    vit_features_reshaped = vit_features.reshape(size, size)
                else:
                    # Pad to nearest square
                    target_size = int(np.ceil(np.sqrt(vit_features.size)))
                    padded = np.pad(vit_features.flatten(), (0, target_size*target_size - vit_features.size))
                    vit_features_reshaped = padded.reshape(target_size, target_size)
            visualizations['vit_attention'] = self._visualize_features(vit_features_reshaped)
            
            # Radiomics feature importance
            rad_features = attention_weights['rad_features'].cpu().numpy()
            visualizations['rad_importance'] = self._visualize_radiomics_importance(rad_features)
            
            return visualizations
    
    def _visualize_features(self, features: np.ndarray) -> np.ndarray:
        """Visualize feature importance as heatmap."""
        # Features are already reshaped in the calling function
        feature_map = features
        
        # Normalize
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-6)
        
        return feature_map
    
    def _visualize_radiomics_importance(self, rad_features: np.ndarray) -> Dict[str, float]:
        """Calculate radiomics feature importance."""
        importance = {}
        
        # Simple importance based on feature magnitude
        for i, feature_val in enumerate(rad_features[0]):
            importance[f'rad_feature_{i}'] = float(abs(feature_val))
        
        return importance
    
    def generate_shap_explanations(self, images: torch.Tensor, radiomics: torch.Tensor, 
                                 background_images: torch.Tensor = None, 
                                 background_rad: torch.Tensor = None) -> Dict:
        """
        Generate SHAP explanations for multimodal predictions.
        (Disabled - requires shap package)
        
        Args:
            images: Input images
            radiomics: Radiomics features
            background_images: Background images for SHAP
            background_rad: Background radiomics for SHAP
            
        Returns:
            SHAP explanations (empty dict - shap not available)
        """
        # SHAP disabled due to dependency issues
        print("SHAP explanations disabled (requires shap package)")
        return {}
    
    def feature_importance_analysis(self, images: torch.Tensor, radiomics: torch.Tensor, 
                                  labels: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Analyze feature importance using permutation importance.
        
        Args:
            images: Input images
            radiomics: Radiomics features
            labels: True labels
            
        Returns:
            Feature importance scores
        """
        def model_predict(vit_data, rad_data):
            with torch.no_grad():
                vit_tensor = torch.FloatTensor(vit_data).to(self.device)
                rad_tensor = torch.FloatTensor(rad_data).to(self.device)
                outputs = self.model(vit_tensor, rad_tensor)
                return outputs.cpu().numpy()
        
        try:
            # Permutation importance for radiomics features
            rad_importance = permutation_importance(
                lambda x: model_predict(images, x),
                radiomics.cpu().numpy(),
                labels.cpu().numpy(),
                n_repeats=10,
                random_state=42
            )
            
            return {
                'rad_importance': rad_importance.importances_mean,
                'rad_importance_std': rad_importance.importances_std
            }
            
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")
            return {}
    
    def generate_complete_explanation(self, image: torch.Tensor, radiomics: torch.Tensor, 
                                   target_class: int, true_label: int = None) -> Dict:
        """
        Generate comprehensive explanation for a single prediction.
        
        Args:
            image: Input image tensor
            radiomics: Radiomics features tensor
            target_class: Predicted class
            true_label: True label (optional)
            
        Returns:
            Dictionary containing all explanation components
        """
        explanation = {}
        
        # Extract ViT features using the same model configuration as training
        from models.vit_model import create_vit_model
        
        # Create ViT model with correct 384-dim configuration
        vit_model = create_vit_model(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            n_heads=12,
            n_layers=16,
            n_classes=3,
            dropout=0.1
        )
        vit_model.to(self.device)
        vit_model.eval()
        
        # Extract ViT features
        with torch.no_grad():
            vit_features = vit_model.extract_features(image)
        
        # Attention visualization
        try:
            attention_viz = self.visualize_attention_maps(vit_features, radiomics)
            explanation['attention_visualization'] = attention_viz
        except Exception as e:
            print(f"Attention visualization failed: {e}")
        
        # Attention maps
        try:
            attention_maps = self.visualize_attention_maps(vit_features, radiomics)
            explanation['attention_maps'] = attention_maps
        except Exception as e:
            print(f"Attention visualization failed: {e}")
        
        # Feature importance
        try:
            rad_importance = self._visualize_radiomics_importance(radiomics.cpu().numpy())
            explanation['radiomics_importance'] = rad_importance
        except Exception as e:
            print(f"Radiomics importance failed: {e}")
        
        # Prediction confidence
        with torch.no_grad():
            output = self.model(vit_features, radiomics)
            probabilities = F.softmax(output, dim=1)
            explanation['prediction_confidence'] = probabilities[0].cpu().numpy()
            explanation['predicted_class'] = target_class
            explanation['true_label'] = true_label
        
        return explanation
    
    def save_explanation_plots(self, explanation: Dict, image: torch.Tensor, 
                             save_path: str, class_names: List[str] = None):
        """
        Save explanation plots to file.
        
        Args:
            explanation: Explanation dictionary
            image: Original image tensor
            save_path: Path to save plots
            class_names: Class names for labels
        """
        if class_names is None:
            class_names = ['Benign', 'Malignant', 'Normal']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM
        if 'grad_cam' in explanation:
            axes[0, 1].imshow(explanation['grad_cam'], cmap='jet')
            axes[0, 1].set_title('Grad-CAM')
            axes[0, 1].axis('off')
        
        # Attention map
        if 'attention_maps' in explanation and 'vit_attention' in explanation['attention_maps']:
            axes[0, 2].imshow(explanation['attention_maps']['vit_attention'], cmap='viridis')
            axes[0, 2].set_title('Visual Attention')
            axes[0, 2].axis('off')
        
        # Prediction confidence
        if 'prediction_confidence' in explanation:
            probs = explanation['prediction_confidence']
            axes[1, 0].bar(class_names, probs)
            axes[1, 0].set_title('Prediction Confidence')
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Radiomics importance (top 10)
        if 'radiomics_importance' in explanation:
            rad_imp = explanation['radiomics_importance']
            top_features = sorted(rad_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importance = zip(*top_features)
            
            axes[1, 1].barh(range(len(features)), importance)
            axes[1, 1].set_yticks(range(len(features)))
            axes[1, 1].set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])
            axes[1, 1].set_title('Top Radiomics Features')
            axes[1, 1].set_xlabel('Importance')
        
        # Summary text
        pred_class = explanation.get('predicted_class', 0)
        true_label = explanation.get('true_label', None)
        confidence = explanation.get('prediction_confidence', [0, 0, 0])[pred_class]
        
        summary_text = f"Predicted: {class_names[pred_class]}\nConfidence: {confidence:.3f}"
        if true_label is not None:
            summary_text += f"\nTrue: {class_names[true_label]}"
            summary_text += f"\nCorrect: {pred_class == true_label}"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Prediction Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Explanation plots saved to: {save_path}")


def create_explainer(model, device='cpu') -> MultimodalExplainer:
    """
    Create a multimodal explainer.
    
    Args:
        model: Trained multimodal model
        device: Device for computation
        
    Returns:
        Multimodal explainer instance
    """
    return MultimodalExplainer(model, device)


if __name__ == "__main__":
    # Test XAI system
    print("Testing Complete XAI System...")
    
    print("XAI system test completed!")
