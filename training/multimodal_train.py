"""
Multimodal Training Pipeline for Lung Cancer Classification.

This module implements the complete training pipeline that combines:
1. Image preprocessing (resized, normalized, augmented)
2. Radiomics feature extraction (27 hand-crafted features)
3. ViT feature extraction (128-dimensional visual features)
4. Multimodal fusion with proper train/validation splits
5. Loss function and optimizer for fusion model

Data Flow:
Images → Preprocessing → Radiomics Features (27) ──┐
Images → Preprocessing → ViT Features (128) ──┤ Fusion → Classification
                                             │
                                       Combined Features (155)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from typing import Tuple, List, Dict, Any
from PIL import Image

# Import custom modules
from models.vit_model import create_vit_model
from models.multimodal_fusion_complete import create_multimodal_model
from utils.preprocessing import load_dataset, get_data_transforms
from utils.extract_radiomics_complete import extract_radiomics_for_dataset


class MultimodalDataset(Dataset):
    """
    Custom dataset for multimodal learning with lazy loading.
    
    This dataset handles:
    - Image paths for lazy loading
    - Pre-computed radiomics features
    - Labels for classification
    - On-the-fly ViT feature extraction
    """
    
    def __init__(self, image_paths: List[str], radiomics_features: np.ndarray, 
                 labels: np.ndarray, vit_model: nn.Module = None,
                 transform=None, extract_vit_features: bool = True):
        """
        Initialize multimodal dataset.
        
        Args:
            image_paths: List of image file paths
            radiomics_features: Pre-computed radiomics features (n_samples, 27)
            labels: Class labels (n_samples,)
            vit_model: ViT model for feature extraction
            transform: Image transformations
            extract_vit_features: Whether to extract ViT features on-the-fly
        """
        self.image_paths = image_paths
        self.radiomics_features = torch.FloatTensor(radiomics_features)
        self.labels = torch.LongTensor(labels)
        self.vit_model = vit_model
        self.transform = transform
        self.extract_vit_features = extract_vit_features
        
        # Pre-extract ViT features if model is provided and not extracting on-the-fly
        if vit_model is not None and not extract_vit_features:
            self.vit_features = self._preextract_vit_features()
        else:
            self.vit_features = None
    
    def _preextract_vit_features(self) -> torch.Tensor:
        """Pre-extract ViT features for all images."""
        print("Pre-extracting ViT features...")
        vit_features_list = []
        
        for i, image_path in enumerate(self.image_paths):
            if (i + 1) % 100 == 0 or i == len(self.image_paths) - 1:
                print(f"  Processed {i + 1}/{len(self.image_paths)} images")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Extract ViT features
            with torch.no_grad():
                vit_feat = self.vit_model.extract_features(image.unsqueeze(0))
                vit_features_list.append(vit_feat.squeeze(0))
        
        return torch.stack(vit_features_list)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get radiomics features and label
        radiomics = self.radiomics_features[idx]
        label = self.labels[idx]
        
        # Get ViT features
        if self.vit_features is not None:
            # Pre-extracted features
            vit_features = self.vit_features[idx]
        elif self.extract_vit_features:
            # Extract on-the-fly
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            with torch.no_grad():
                vit_features = self.vit_model.extract_features(image.unsqueeze(0)).squeeze(0)
        else:
            vit_features = torch.zeros(128)  # Default ViT feature dimension
        
        return {
            'vit_features': vit_features,
            'radiomics_features': radiomics,
            'label': label,
            'image_path': self.image_paths[idx]
        }


class MultimodalTrainer:
    """
    Trainer class for multimodal fusion model.
    
    Handles training, validation, and evaluation with proper loss functions
    and optimizers for the fusion model.
    """
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize trainer.
        
        Args:
            model: Multimodal fusion model
            device: Training device (CPU/GPU)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function: CrossEntropyLoss with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler: Reduce on plateau with less aggressive decay
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.8, 
            patience=8,
            min_lr=1e-5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        print(f"MultimodalTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Loss function: CrossEntropyLoss")
        print(f"  Optimizer: AdamW")
        print(f"  Scheduler: ReduceLROnPlateau")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train model for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        epoch_total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Extract data from batch (tuple format)
            vit_features, radiomics_features, labels = batch
            vit_features = vit_features.to(self.device)
            radiomics_features = radiomics_features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, features = self.model(vit_features, radiomics_features)
            
            # Handle different loss types
            if hasattr(self.criterion, '__class__') and 'Asymmetric' in self.criterion.__class__.__name__:
                # Asymmetric loss only needs logits and labels
                loss_output = self.criterion(logits, labels)
                total_loss = loss_output
                ce_loss = total_loss  # For display purposes
                center_loss = torch.tensor(0.0, device=self.device)
            else:
                # CombinedLoss needs logits, features, and labels
                loss_output = self.criterion(logits, features, labels)
                if isinstance(loss_output, tuple):  # CombinedLoss returns tuple
                    total_loss, ce_loss, center_loss = loss_output
                else:
                    total_loss = loss_output
                    ce_loss = total_loss
                    center_loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            epoch_total_loss += total_loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'ce_loss': f'{ce_loss.item():.4f}',
                'center_loss': f'{center_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = epoch_total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model for one epoch.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        epoch_total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                # Extract data from batch (tuple format)
                vit_features, radiomics_features, labels = batch
                vit_features = vit_features.to(self.device)
                radiomics_features = radiomics_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, features = self.model(vit_features, radiomics_features)
                
                # Handle different loss types
                if hasattr(self.criterion, '__class__') and 'Asymmetric' in self.criterion.__class__.__name__:
                    # Asymmetric loss only needs logits and labels
                    loss_output = self.criterion(logits, labels)
                    total_loss = loss_output
                    ce_loss = total_loss  # For display purposes
                    center_loss = torch.tensor(0.0, device=self.device)
                else:
                    # CombinedLoss needs logits, features, and labels
                    loss_output = self.criterion(logits, features, labels)
                    if isinstance(loss_output, tuple):  # CombinedLoss returns tuple
                        total_loss, ce_loss, center_loss = loss_output
                    else:
                        total_loss = loss_output
                        ce_loss = total_loss
                        center_loss = torch.tensor(0.0, device=self.device)
                
                # Statistics
                epoch_total_loss += total_loss.item()
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'total_loss': f'{total_loss.item():.4f}',
                    'ce_loss': f'{ce_loss.item():.4f}',
                    'center_loss': f'{center_loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = epoch_total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, save_path: str = None) -> Dict[str, List[float]]:
        """
        Train the multimodal fusion model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_path: Path to save best model
        
        Returns:
            dict: Training history
        """
        print("Computing class weights for imbalanced dataset...")
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        all_labels = np.array(train_loader.dataset.labels)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(all_labels), 
            y=all_labels
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Apply reduced manual weighting for better class balance
        # Conservative boost for underrepresented classes
        balanced_weights = class_weights.clone()
        
        # Handle different number of classes (2 for Stage 1, 3 for Stage 2)
        if len(balanced_weights) == 2:
            # Stage 1: Normal vs Abnormal
            balanced_weights[0] *= 1.2  # Conservative boost for Normal class
            print("Stage 1: Normal vs Abnormal - boosting Normal class")
        else:
            # Stage 2: Benign vs Malignant (or original 3-class)
            balanced_weights[0] *= 1.1  # Conservative boost for Benign class
            if len(balanced_weights) > 2:
                balanced_weights[2] *= 1.2  # Conservative boost for Normal class
            print("Stage 2: Benign vs Malignant - boosting Benign class")
        
        print(f"Original class weights: {class_weights.cpu().numpy()}")
        print(f"Reduced manual weights: {balanced_weights.cpu().numpy()}")
        print("Using conservative weighting to avoid over-emphasis")
        
        # Check if this is Stage 2 by examining the unique labels in dataset
        unique_labels = torch.unique(torch.tensor(train_loader.dataset.labels))
        
        # Stage 1 has labels [0, 1] (Normal=0, Abnormal=1)
        # Stage 2 has labels [0, 1] (Benign=0, Malignant=1) but we need to differentiate
        # We'll use the data distribution to determine the stage
        
        # Stage 2 typically has equal benign/malignant (1500 each)
        # Stage 1 has imbalanced normal/abnormal (1500 normal, 3000 abnormal)
        label_counts = torch.bincount(torch.tensor(train_loader.dataset.labels))
        
        if len(label_counts) == 2 and label_counts[0] == label_counts[1]:  # Equal distribution = Stage 2
            from utils.asymmetric_loss import WeightedAsymmetricLoss
            self.criterion = WeightedAsymmetricLoss(
                malignant_weight=3.0,  # Heavy penalty for missing malignant
                benign_weight=1.0,
                gamma_neg=4
            )
            print("Using WeightedAsymmetricLoss for Stage 2 (Benign vs Malignant)")
        else:  # Stage 1 or other multi-class
            from utils.center_loss import CombinedLoss
            
            # Get feature dimension from model
            with torch.no_grad():
                dummy_vit = torch.randn(1, 256).to(self.device)  # Match reduced ViT dim
                dummy_rad = torch.randn(1, 46).to(self.device)
                _, dummy_features = self.model(dummy_vit, dummy_rad)
                feat_dim = dummy_features.size(1)
            
            self.criterion = CombinedLoss(
                num_classes=len(unique_labels),
                feat_dim=feat_dim,
                ce_weight=1.0,
                center_weight=0.001  # Reduced to prevent explosion
            )
            print(f"Using CombinedLoss (CrossEntropy + CenterLoss) with feat_dim={feat_dim}")
        
        print(f"\n=== Training Multimodal Fusion Model ===")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'history': self.history
                    }, save_path)
                    print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
        
        return self.history


def create_multimodal_data_loaders(dataset_path: str, batch_size: int = 32, 
                                 test_size: float = 0.2, val_size: float = 0.1,
                                 image_size: int = 224, num_workers: int = 4,
                                 preextract_vit_features: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create multimodal data loaders with proper train/validation/test splits.
    
    Args:
        dataset_path: Path to dataset root
        batch_size: Batch size for data loaders
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        image_size: Target image size
        num_workers: Number of worker processes
        preextract_vit_features: Whether to pre-extract ViT features
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("=== Creating Multimodal Data Loaders ===\n")
    
    # Step 1: Load image paths and labels
    print("1. Loading dataset...")
    image_paths, labels = load_dataset(dataset_path, image_size=(image_size, image_size))
    print(f"   Loaded {len(image_paths)} images from {len(set(labels))} classes")
    
    # Step 2: Extract radiomics features
    print("\n2. Extracting radiomics features...")
    radiomics_features, feature_names = extract_radiomics_for_dataset(image_paths)
    print(f"   Extracted {len(feature_names)} radiomics features")
    print(f"   Radiomics features shape: {radiomics_features.shape}")
    
    # Step 3: Create ViT model for feature extraction
    print("\n3. Creating ViT model...")
    vit_model = create_vit_model(
        img_size=image_size,
        patch_size=16,
        embed_dim=128,
        n_heads=8,
        n_layers=8,
        n_classes=3,
        dropout=0.1
    )
    vit_model.eval()
    
    # Step 4: Split dataset
    print("\n4. Splitting dataset...")
    
    # First split: separate test set
    X_paths_temp, X_paths_test, y_temp, y_test, X_rad_temp, X_rad_test = train_test_split(
        image_paths, labels, radiomics_features,
        test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation
    X_paths_train, X_paths_val, y_train, y_val, X_rad_train, X_rad_val = train_test_split(
        X_paths_temp, y_temp, X_rad_temp,
        test_size=val_size, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_paths_train)} samples")
    print(f"   Validation: {len(X_paths_val)} samples")
    print(f"   Test: {len(X_paths_test)} samples")
    
    # Step 5: Create transforms
    print("\n5. Creating data transforms...")
    train_transform = get_data_transforms(image_size, is_training=True)
    val_transform = get_data_transforms(image_size, is_training=False)
    
    # Step 6: Create datasets
    print("\n6. Creating datasets...")
    
    train_dataset = MultimodalDataset(
        X_paths_train, X_rad_train, y_train,
        vit_model=vit_model, transform=train_transform,
        extract_vit_features=not preextract_vit_features
    )
    
    val_dataset = MultimodalDataset(
        X_paths_val, X_rad_val, y_val,
        vit_model=vit_model, transform=val_transform,
        extract_vit_features=not preextract_vit_features
    )
    
    test_dataset = MultimodalDataset(
        X_paths_test, X_rad_test, y_test,
        vit_model=vit_model, transform=val_transform,
        extract_vit_features=not preextract_vit_features
    )
    
    # Step 7: Create data loaders
    print("\n7. Creating data loaders...")
    
    # Implement oversampling for minority classes
    from torch.utils.data import WeightedRandomSampler
    
    # Calculate class weights for sampling
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_dataset.labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   Data loaders created successfully")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


def visualize_data_flow():
    """
    Create a visualization of the multimodal data flow.
    """
    print("\n=== Multimodal Data Flow Visualization ===\n")
    
    # Create flow diagram
    flow_steps = [
        "1. INPUT IMAGES",
        "   ↓",
        "2. PREPROCESSING",
        "   - Resize to 224×224",
        "   - Normalize pixel values",
        "   - Apply data augmentation (training only)",
        "   ↓",
        "3. FEATURE EXTRACTION (PARALLEL)",
        "   ├── PATH A: RADIOMICS",
        "   │   - Extract 27 hand-crafted features",
        "   │   - Mean, std, min, max intensity",
        "   │   - Texture features (LBP, histogram)",
        "   │   - Shape features (contours, compactness)",
        "   │   - Statistical features (skewness, kurtosis)",
        "   │   └→ Radiomics Features (27-dim)",
        "   │",
        "   └── PATH B: VISION TRANSFORMER",
        "       - Patch embedding (16×16 patches)",
        "       - Positional encoding",
        "       - 8 transformer blocks",
        "       - Multi-head attention (8 heads)",
        "       - Global average pooling",
        "       └→ ViT Features (128-dim)",
        "   ↓",
        "4. FEATURE FUSION",
        "   - Concatenate: [ViT(128) + Radiomics(27)] = 155-dim",
        "   - FC Layer 1: 155 → 64 (ReLU + BatchNorm + Dropout)",
        "   - FC Layer 2: 64 → 64 (ReLU + BatchNorm + Dropout)",
        "   - Output Layer: 64 → 3 (classes)",
        "   ↓",
        "5. CLASSIFICATION",
        "   - Output logits for [Benign, Malignant, Normal]",
        "   - Loss: CrossEntropyLoss",
        "   - Optimizer: AdamW",
        "   - Learning Rate: 0.001 (with scheduler)",
        "   ↓",
        "6. TRAINING LOOP",
        "   - Forward pass → Loss → Backward pass → Update",
        "   - Validation after each epoch",
        "   - Early stopping on validation accuracy",
        "   - Model checkpointing for best validation accuracy"
    ]
    
    for step in flow_steps:
        print(step)
    
    print("\n=== Key Data Shapes ===")
    print("Input Image: (3, 224, 224)")
    print("Radiomics Features: (batch_size, 27)")
    print("ViT Features: (batch_size, 128)")
    print("Fused Features: (batch_size, 155)")
    print("Hidden Features: (batch_size, 64)")
    print("Output Logits: (batch_size, 3)")
    print("Predictions: (batch_size,) - class indices")
    
    print("\n=== Loss Function & Optimizer ===")
    print("Loss: CrossEntropyLoss()")
    print("  - Suitable for multi-class classification")
    print("  - Combines LogSoftmax + NLLLoss")
    print("  - Handles class imbalance implicitly")
    print("")
    print("Optimizer: AdamW")
    print("  - Adam with decoupled weight decay")
    print("  - Learning rate: 0.001")
    print("  - Weight decay: 1e-4 (L2 regularization)")
    print("  - Beta1: 0.9, Beta2: 0.999")
    print("")
    print("Scheduler: ReduceLROnPlateau")
    print("  - Reduces LR when validation accuracy plateaus")
    print("  - Factor: 0.5 (reduce by 50%)")
    print("  - Patience: 5 epochs")


def main():
    """
    Main function to run the complete multimodal training pipeline.
    """
    print("=" * 60)
    print("MULTIMODAL LUNG CANCER CLASSIFICATION")
    print("=" * 60)
    
    # Show data flow
    visualize_data_flow()
    
    # Configuration
    config = {
        'dataset_path': 'data/raw/dataset',
        'batch_size': 32,
        'test_size': 0.2,
        'val_size': 0.1,
        'image_size': 224,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'preextract_vit_features': True
    }
    
    print(f"\n=== Training Configuration ===")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_multimodal_data_loaders(
        dataset_path=config['dataset_path'],
        batch_size=config['batch_size'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        preextract_vit_features=config['preextract_vit_features']
    )
    
    # Create multimodal model
    print(f"\n=== Creating Multimodal Fusion Model ===")
    vit_config = {
        'img_size': config['image_size'],
        'patch_size': 16,
        'embed_dim': 256,
        'n_heads': 8,
        'n_layers': 12,
        'n_classes': 3,
        'dropout': 0.1
    }
    fusion_model = create_multimodal_model(
        vit_config=vit_config,
        rad_dim=100,
        num_classes=3,
        fusion_type='attention'
    )
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = MultimodalTrainer(
        model=fusion_model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    print(f"\n=== Starting Training ===")
    save_path = 'outputs/models/multimodal_fusion_best.pth'
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        save_path=save_path
    )
    
    # Plot training history
    plot_training_history(history, save_path='outputs/plots/multimodal_training_history.png')
    
    print(f"\n=== Training Complete ===")
    print(f"Best model saved to: {save_path}")
    print(f"Training history plotted to: outputs/plots/multimodal_training_history.png")


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
