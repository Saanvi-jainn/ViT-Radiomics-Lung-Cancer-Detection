"""
Medical Image Augmentation for Lung Cancer Classification
Focus on CLAHE and contrast enhancement for subtle texture differences
Critical for Benign vs Malignant classification
"""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random


class CLAHETransform:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) for medical images"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), probability=0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.probability = probability
        
    def __call__(self, img):
        if random.random() > self.probability:
            return img
            
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Apply CLAHE to each channel
        if len(img_np.shape) == 3:
            # RGB image
            img_clahe = np.zeros_like(img_np)
            for i in range(3):
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                img_clahe[:, :, i] = clahe.apply(img_np[:, :, i])
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_clahe = clahe.apply(img_np)
        
        return Image.fromarray(img_clahe)


class RandomContrast:
    """Random contrast adjustment for medical images"""
    
    def __init__(self, lower=0.8, upper=1.2, probability=0.5):
        self.lower = lower
        self.upper = upper
        self.probability = probability
        
    def __call__(self, img):
        if random.random() > self.probability:
            return img
            
        # Convert to tensor for contrast adjustment
        img_tensor = transforms.functional.to_tensor(img)
        
        # Apply random contrast
        contrast_factor = random.uniform(self.lower, self.upper)
        img_tensor = transforms.functional.adjust_contrast(img_tensor, contrast_factor)
        
        # Convert back to PIL
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return transforms.functional.to_pil_image(img_tensor)


class RandomBrightness:
    """Random brightness adjustment for medical images"""
    
    def __init__(self, lower=0.9, upper=1.1, probability=0.5):
        self.lower = lower
        self.upper = upper
        self.probability = probability
        
    def __call__(self, img):
        if random.random() > self.probability:
            return img
            
        # Convert to tensor for brightness adjustment
        img_tensor = transforms.functional.to_tensor(img)
        
        # Apply random brightness
        brightness_factor = random.uniform(self.lower, self.upper)
        img_tensor = transforms.functional.adjust_brightness(img_tensor, brightness_factor)
        
        # Convert back to PIL
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return transforms.functional.to_pil_image(img_tensor)


class RandomZoom:
    """Random zoom/crop for medical images (more effective than flips)"""
    
    def __init__(self, zoom_range=(0.9, 1.1), probability=0.5):
        self.zoom_range = zoom_range
        self.probability = probability
        
    def __call__(self, img):
        if random.random() > self.probability:
            return img
            
        w, h = img.size
        zoom_factor = random.uniform(*self.zoom_range)
        
        # Calculate new dimensions
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        
        # Resize and center crop back to original size
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Center crop
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img_cropped = img_resized.crop((left, top, left + w, top + h))
        
        return img_cropped


class GaussianNoise:
    """Add Gaussian noise for robustness"""
    
    def __init__(self, mean=0, std=0.01, probability=0.3):
        self.mean = mean
        self.std = std
        self.probability = probability
        
    def __call__(self, img):
        if random.random() > self.probability:
            return img
            
        # Convert to tensor
        img_tensor = transforms.functional.to_tensor(img)
        
        # Add Gaussian noise
        noise = torch.randn_like(img_tensor) * self.std + self.mean
        img_tensor = img_tensor + noise
        
        # Clamp and convert back
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return transforms.functional.to_pil_image(img_tensor)


def get_medical_augmentation_transforms(image_size=224, is_training=True):
    """
    Get medical-specific augmentation transforms.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply augmentations
        
    Returns:
        Transform pipeline
    """
    
    if is_training:
        # Strong augmentation for Stage 2 (Benign vs Malignant)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            
            # Medical-specific augmentations
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8), probability=0.7),
            RandomContrast(lower=0.8, upper=1.3, probability=0.6),
            RandomBrightness(lower=0.9, upper=1.1, probability=0.5),
            RandomZoom(zoom_range=(0.9, 1.1), probability=0.5),
            GaussianNoise(std=0.01, probability=0.3),
            
            # Standard augmentations (reduced probability)
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            
            # Final processing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_strong_stage2_transforms(image_size=224):
    """
    Get strong augmentation specifically for Stage 2 (Benign vs Malignant).
    This focuses on texture and contrast differences.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        
        # Strong medical augmentations for texture differences
        CLAHETransform(clip_limit=3.0, tile_grid_size=(8, 8), probability=0.8),
        RandomContrast(lower=0.7, upper=1.4, probability=0.7),
        RandomBrightness(lower=0.85, upper=1.15, probability=0.6),
        RandomZoom(zoom_range=(0.85, 1.15), probability=0.6),
        GaussianNoise(std=0.015, probability=0.4),
        
        # Minimal geometric augmentations
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=5),
        
        # Final processing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform
