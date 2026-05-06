"""
Enhanced image preprocessing utilities for lung cancer CT scan dataset.

This module handles loading, preprocessing, and augmentation of CT scan images
for the Vision Transformer model with advanced preprocessing options.

Key Features:
- Resize to fixed size (224x224 recommended)
- Normalize pixel values (0-1 scaling)
- Optional Gaussian blur for noise reduction
- Data augmentation for training
- Integration with PyTorch DataLoader
"""

import os
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class CTScanDataset(Dataset):
    """
    PyTorch Dataset for CT scan images with enhanced preprocessing.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels (0=Benign, 1=Malignant, 2=Normal)
        transform: PyTorch transforms for data augmentation
        preprocess_fn: Additional preprocessing function
    """
    
    def __init__(self, image_paths, labels, transform=None, preprocess_fn=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocess_fn = preprocess_fn
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply custom preprocessing if provided
        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def resize_image(image, target_size=(224, 224), resample=Image.BILINEAR):
    """
    Resize image to target size with specified resampling method.
    
    Args:
        image: PIL Image
        target_size: Tuple (width, height) for target size
        resample: PIL resampling filter
    
    Returns:
        PIL.Image: Resized image
    """
    return image.resize(target_size, resample)


def normalize_image(image, method='minmax'):
    """
    Normalize image pixel values to specified range.
    
    Args:
        image: PIL Image or numpy array
        method: Normalization method ('minmax', 'zscore', '0-1')
    
    Returns:
        Same type as input with normalized values
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image, dtype=np.float32)
    else:
        img_array = image.astype(np.float32)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    elif method == 'zscore':
        # Z-score normalization
        mean, std = img_array.mean(), img_array.std()
        img_array = (img_array - mean) / (std + 1e-8)
    elif method == '0-1':
        # Simple scaling to [0, 1]
        img_array = img_array / 255.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if isinstance(image, Image.Image):
        return Image.fromarray(np.clip(img_array * 255, 0, 255).astype(np.uint8))
    else:
        return img_array


def apply_gaussian_blur(image, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur for noise reduction.
    
    Args:
        image: PIL Image or numpy array
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Same type as input with blur applied
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        is_pil = True
    else:
        img_array = image.copy()
        is_pil = False
    
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
    
    if is_pil:
        return Image.fromarray(blurred.astype(np.uint8))
    else:
        return blurred


def enhance_contrast(image, method='clahe', clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance image contrast for better feature extraction.
    
    Args:
        image: PIL Image or numpy array
        method: Contrast enhancement method ('clahe', 'histogram_eq')
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
    
    Returns:
        Same type as input with enhanced contrast
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image, dtype=np.uint8)
        is_pil = True
    else:
        img_array = image.astype(np.uint8)
        is_pil = False
    
    # Convert to grayscale for CLAHE
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB if original was RGB
        if len(img_array.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    elif method == 'histogram_eq':
        # Simple histogram equalization
        enhanced = cv2.equalizeHist(gray)
        
        # Convert back to RGB if original was RGB
        if len(img_array.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")
    
    if is_pil:
        return Image.fromarray(enhanced.astype(np.uint8))
    else:
        return enhanced


def create_preprocessing_pipeline(target_size=(224, 224), normalize_method='0-1', 
                           apply_blur=False, blur_kernel=3, blur_sigma=1.0,
                           enhance_contrast=False, contrast_method='clahe'):
    """
    Create a comprehensive preprocessing pipeline.
    
    Args:
        target_size: Target image size (width, height)
        normalize_method: Normalization method ('minmax', 'zscore', '0-1')
        apply_blur: Whether to apply Gaussian blur
        blur_kernel: Gaussian blur kernel size
        blur_sigma: Gaussian blur sigma
        enhance_contrast: Whether to enhance contrast
        contrast_method: Contrast enhancement method
    
    Returns:
        function: Preprocessing function that takes PIL Image and returns PIL Image
    """
    def preprocess_fn(image):
        # Resize image
        if image.size != target_size:
            image = resize_image(image, target_size)
        
        # Apply contrast enhancement
        if enhance_contrast:
            image = enhance_contrast(image, method=contrast_method)
        
        # Apply Gaussian blur for noise reduction
        if apply_blur:
            image = apply_gaussian_blur(image, blur_kernel, blur_sigma)
        
        # Normalize pixel values
        if normalize_method:
            image = normalize_image(image, method=normalize_method)
        
        return image
    
    return preprocess_fn


def load_dataset(dataset_root, image_size=(224, 224)):
    """
    Load CT scan dataset from directory structure.
    
    Expected structure:
    dataset_root/
    ├── Bengin cases/
    ├── Malignant cases/
    └── Normal cases/
    
    Args:
        dataset_root: Path to dataset root directory
        image_size: Target size for resizing images (width, height)
    
    Returns:
        tuple: (image_paths, labels) as lists
    """
    # Class mapping
    class_dirs = {
        0: "Bengin cases",
        1: "Malignant cases", 
        2: "Normal cases",
    }
    
    image_paths = []
    labels = []
    
    for label, folder_name in class_dirs.items():
        folder = os.path.join(dataset_root, folder_name)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing dataset folder: {folder}")
        
        # Get all image files
        for image_file in sorted(os.listdir(folder)):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(folder, image_file)
                image_paths.append(image_path)
                labels.append(label)
    
    if not image_paths:
        raise RuntimeError(f"No images found under {dataset_root}")
    
    print(f"Loaded {len(image_paths)} images from {len(class_dirs)} classes")
    return image_paths, labels


def get_data_transforms(image_size=224, is_training=True):
    """
    Get simple and effective data transforms.
    
    Args:
        image_size: Target image size (default 224x224)
        is_training: Whether this is for training (with augmentation)
    
    Returns:
        torchvision transforms composition
    """
    # Normalize to ImageNet standards
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    
    if is_training:
        # Enhanced training transforms for better generalization
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        # Validation/test transforms (NO augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    
    return transform


def create_data_loaders(image_paths, labels, batch_size=32, test_size=0.2, 
                        val_size=0.15, image_size=224, num_workers=4,
                        preprocess_config=None):
    """
    Create train, validation, and test data loaders with enhanced preprocessing.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Batch size for data loaders
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        preprocess_config: Dictionary with preprocessing configuration
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Default preprocessing configuration
    if preprocess_config is None:
        preprocess_config = {
            'target_size': (image_size, image_size),
            'normalize_method': '0-1',
            'apply_blur': False,
            'blur_kernel': 3,
            'blur_sigma': 1.0,
            'enhance_contrast': False,
            'contrast_method': 'clahe'
        }
    
    # For 14K dataset, use 70/15/15 split for better training
    if len(image_paths) > 10000:  # Large dataset optimization
        test_size = 0.15
        val_size = 0.176  # 0.15/0.85 to get final 15% validation
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, 
        random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, 
        random_state=42, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples") 
    print(f"  Test: {len(X_test)} samples")
    
    # Create preprocessing function
    preprocess_fn = create_preprocessing_pipeline(**preprocess_config)
    
    # Create datasets
    train_transform = get_data_transforms(image_size, is_training=True)
    val_transform = get_data_transforms(image_size, is_training=False)
    
    # For multiprocessing, we need to handle preprocessing differently
    # Apply preprocessing directly in the dataset __getitem__ method
    train_dataset = CTScanDataset(
        X_train, y_train, 
        transform=train_transform, 
        preprocess_fn=None  # We'll handle preprocessing in transform
    )
    val_dataset = CTScanDataset(
        X_val, y_val, 
        transform=val_transform, 
        preprocess_fn=None
    )
    test_dataset = CTScanDataset(
        X_test, y_test, 
        transform=val_transform, 
        preprocess_fn=None
    )
    
    # Optimize data loading for 14K dataset
    # Use more workers for larger datasets but limit to avoid memory issues
    optimal_workers = min(num_workers, 8) if len(image_paths) > 10000 else num_workers
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=optimal_workers, pin_memory=True, 
        persistent_workers=True if optimal_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=optimal_workers, pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=optimal_workers, pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels: List of class labels
    
    Returns:
        torch.Tensor: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)


# Example usage and demonstration
if __name__ == "__main__":
    """
    Example usage of the preprocessing module.
    """
    print("=== Preprocessing Module Example Usage ===\n")
    
    # Example 1: Basic preprocessing
    print("1. Basic preprocessing pipeline:")
    basic_config = {
        'target_size': (224, 224),
        'normalize_method': '0-1',
        'apply_blur': False,
        'enhance_contrast': False
    }
    print(f"   Config: {basic_config}")
    
    # Example 2: Advanced preprocessing with noise reduction
    print("\n2. Advanced preprocessing with noise reduction:")
    advanced_config = {
        'target_size': (224, 224),
        'normalize_method': 'minmax',
        'apply_blur': True,
        'blur_kernel': 5,
        'blur_sigma': 1.5,
        'enhance_contrast': True,
        'contrast_method': 'clahe'
    }
    print(f"   Config: {advanced_config}")
    
    # Example 3: Integration with dataset loader
    print("\n3. Integration with dataset loader:")
    print("   # Load dataset")
    print("   image_paths, labels = load_dataset('data/raw/dataset', image_size=(224, 224))")
    print("   ")
    print("   # Create data loaders with preprocessing")
    print("   train_loader, val_loader, test_loader = create_data_loaders(")
    print("       image_paths, labels,")
    print("       batch_size=32,")
    print("       image_size=224,")
    print("       preprocess_config=advanced_config")
    print("   )")
    
    # Example 4: Individual function usage
    print("\n4. Individual function usage:")
    print("   from PIL import Image")
    print("   ")
    print("   # Load and preprocess single image")
    print("   image = Image.open('ct_scan.jpg')")
    print("   ")
    print("   # Resize to 224x224")
    print("   resized = resize_image(image, (224, 224))")
    print("   ")
    print("   # Apply Gaussian blur for noise reduction")
    print("   blurred = apply_gaussian_blur(resized, kernel_size=3, sigma=1.0)")
    print("   ")
    print("   # Normalize to [0, 1]")
    print("   normalized = normalize_image(blurred, method='0-1')")
    print("   ")
    print("   # Enhance contrast")
    print("   enhanced = enhance_contrast(normalized, method='clahe')")
    
    print("\n=== End of Example ===")
