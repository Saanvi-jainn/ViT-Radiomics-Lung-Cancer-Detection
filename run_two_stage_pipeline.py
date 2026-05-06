"""
Two-Stage Pipeline for Lung Cancer Detection
Stage 1: Normal vs Abnormal (Benign + Malignant)
Stage 2: Benign vs Malignant (only on abnormal samples)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from utils.preprocessing import load_dataset, get_data_transforms
from utils.extract_radiomics_complete import extract_radiomics_for_dataset
from models.multimodal_fusion_complete import create_multimodal_model
from training.multimodal_train import MultimodalTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

class TwoStagePipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for both stages."""
        print("=" * 80)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("=" * 80)
        
        # Load original dataset
        self.image_paths, self.labels = load_dataset(self.config['data_path'])
        print(f"Loaded {len(self.image_paths)} images")
        
        # Extract radiomics features
        from utils.extract_radiomics_complete import extract_radiomics_for_dataset
        self.radiomics_features, self.feature_names = extract_radiomics_for_dataset(self.image_paths)
        print(f"Extracted {self.radiomics_features.shape[1]} radiomics features")
        
        # Create Stage 1 labels (Normal=0, Abnormal=1)
        self.stage1_labels = np.array([0 if label == 2 else 1 for label in self.labels])
        
        # Create Stage 2 dataset (only abnormal samples)
        abnormal_indices = np.where(self.stage1_labels == 1)[0]
        self.stage2_indices = abnormal_indices
        self.stage2_image_paths = [self.image_paths[i] for i in abnormal_indices]
        self.stage2_radiomics = self.radiomics_features[abnormal_indices]
        self.stage2_labels = np.array([self.labels[i] for i in abnormal_indices if self.labels[i] != 2])
        
        # Convert Stage 2 labels (Benign=0, Malignant=1)
        self.stage2_labels = np.array([0 if label == 0 else 1 for label in self.stage2_labels])
        
        print(f"Stage 1 - Normal: {np.sum(self.stage1_labels == 0)}, Abnormal: {np.sum(self.stage1_labels == 1)}")
        print(f"Stage 2 - Benign: {np.sum(self.stage2_labels == 0)}, Malignant: {np.sum(self.stage2_labels == 1)}")
        
    def train_stage1_model(self):
        """Train Stage 1: Normal vs Abnormal classifier."""
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING STAGE 1 MODEL (Normal vs Abnormal)")
        print("=" * 80)
        
        # Create model for Stage 1 (2 classes)
        stage1_config = self.config.copy()
        stage1_config['num_classes'] = 2
        
        # Create ViT model for Stage 1 (2 classes: Normal vs Abnormal)
        self.stage1_model = create_multimodal_model(
            vit_config={
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 256,  # Smaller for Stage 1
                'n_heads': 8,     # Match reduced ViT
                'n_layers': 8,    # Match reduced ViT
                'n_classes': 2,
                'dropout': 0.1
            },
            rad_dim=46,
            num_classes=2,
            fusion_type=stage1_config['fusion_type']
        ).to(self.device)
        
        # Implement focal loss for better class imbalance handling
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss(reduction='none')
                
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        # Create trainer for Stage 1 with focal loss
        self.stage1_trainer = MultimodalTrainer(
            self.stage1_model,
            self.device,
            learning_rate=1e-5,  # Critical for fine-tuning ViT
            weight_decay=stage1_config['weight_decay']
        )
        
        # Replace criterion with focal loss + knowledge distillation
        class DistillationLoss(nn.Module):
            def __init__(self, alpha=0.3, temperature=4.0):
                super(DistillationLoss, self).__init__()
                self.alpha = alpha  # Weight for distillation loss
                self.temperature = temperature
                self.focal_loss = FocalLoss(alpha=1, gamma=2)
                self.kl_loss = nn.KLDivLoss(reduction='batchmean')
                
            def forward(self, student_logits, teacher_logits, labels):
                # Standard classification loss
                classification_loss = self.focal_loss(student_logits, labels)
                
                # Knowledge distillation loss
                student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
                teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
                
                distillation_loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
                
                # Combine losses
                total_loss = (1 - self.alpha) * classification_loss + self.alpha * distillation_loss
                return total_loss
        
        # Create data loaders for Stage 1
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from sklearn.model_selection import train_test_split
        
        # Split data for Stage 1
        train1_idx, val1_idx, train1_labels, val1_labels = train_test_split(
            range(len(self.image_paths)), self.stage1_labels,
            test_size=0.2, random_state=42, stratify=self.stage1_labels
        )
        
        train1_paths = [self.image_paths[i] for i in train1_idx]
        val1_paths = [self.image_paths[i] for i in val1_idx]
        train1_rad = self.radiomics_features[train1_idx]
        val1_rad = self.radiomics_features[val1_idx]
        
        # Create radiomics teacher model after data is available
        from utils.radiomics_ensemble import train_radiomics_classifier
        teacher_classifier, _ = train_radiomics_classifier(
            train1_rad, train1_labels,
            model_type='random_forest',
            save_path=self.output_dir / 'radiomics_teacher.pkl'
        )
        
        self.stage1_trainer.criterion = DistillationLoss(alpha=0.3, temperature=4.0)
        self.stage1_trainer.teacher_classifier = teacher_classifier  # Store teacher for distillation
        
        # Add learning rate scheduler for better convergence
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        self.stage1_trainer.scheduler = ReduceLROnPlateau(
            self.stage1_trainer.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Create ViT model for feature extraction (same as training)
        from models.vit_model import create_vit_model
        vit_model = create_vit_model(
            img_size=224,
            patch_size=16,
            embed_dim=256,  # Reduced from 384
            n_heads=8,     # Reduced from 12
            n_layers=8,    # Reduced from 16
            n_classes=2
        ).to(self.device)
        
        # Improved fine-tuning strategy for larger ViT
        vit_model = self.stage1_model.vit_model
        total_layers = len(vit_model.transformer_blocks)
        
        # Freeze all layers first
        for param in vit_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last 5 layers for fine-tuning
        for i in range(max(0, total_layers - 5), total_layers):
            for param in vit_model.transformer_blocks[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze the final classification layers
        for param in vit_model.head.parameters():
            param.requires_grad = True
        
        # Create enhanced medical-specific transforms for ViT
        from utils.preprocessing import get_data_transforms
        import torchvision.transforms as transforms
        from PIL import ImageEnhance
        import cv2
        import numpy as np
        
        class MedicalTransforms:
            def __init__(self, image_size=224, is_training=True):
                self.image_size = image_size
                self.is_training = is_training
                
            def apply_clahe(self, image):
                """Apply CLAHE for better contrast enhancement"""
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    # Convert to LAB color space for better CLAHE
                    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return Image.fromarray(img_array)
            
            def __call__(self, image):
                # Apply CLAHE first
                image = self.apply_clahe(image)
                
                if self.is_training:
                    # Medical-safe augmentations
                    if np.random.random() > 0.5:
                        enhancer = ImageEnhance.Contrast(image)
                        image = enhancer.enhance(np.random.uniform(0.8, 1.2))
                    
                    if np.random.random() > 0.5:
                        enhancer = ImageEnhance.Brightness(image)
                        image = enhancer.enhance(np.random.uniform(0.9, 1.1))
                    
                    # Slight zoom/crop
                    if np.random.random() > 0.5:
                        width, height = image.size
                        zoom = np.random.uniform(0.9, 1.0)
                        new_width, new_height = int(width * zoom), int(height * zoom)
                        left = (width - new_width) // 2
                        top = (height - new_height) // 2
                        image = image.crop((left, top, left + new_width, top + new_height))
                        image = image.resize((width, height))
                # Standard transforms
                base_transforms = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                return base_transforms(image)
        
        train_transform = MedicalTransforms(image_size=224, is_training=True)
        val_transform = MedicalTransforms(image_size=224, is_training=False)
        
        # Pre-compute ViT features for faster training
        from PIL import Image
        print("Pre-computing ViT features for Stage 1...")
        train1_vit_features = []
        val1_vit_features = []
        
        # Pre-compute training features
        for path in tqdm(train1_paths, desc="Extracting train ViT features"):
            image = Image.open(path).convert('RGB')
            if train_transform:
                image = train_transform(image)
            with torch.no_grad():
                features = vit_model.extract_features(image.unsqueeze(0)).squeeze(0)
            train1_vit_features.append(features.cpu().numpy())
        
        # Pre-compute validation features
        for path in tqdm(val1_paths, desc="Extracting val ViT features"):
            image = Image.open(path).convert('RGB')
            if val_transform:
                image = val_transform(image)
            with torch.no_grad():
                features = vit_model.extract_features(image.unsqueeze(0)).squeeze(0)
            val1_vit_features.append(features.cpu().numpy())
        
        train1_vit_features = np.array(train1_vit_features)
        val1_vit_features = np.array(val1_vit_features)
        print(f"ViT features pre-computed: {train1_vit_features.shape}")
        
        # Create fast dataset class with pre-computed features
        from torch.utils.data import Dataset
        
        class FastMultimodalDataset(Dataset):
            def __init__(self, radiomics_features, vit_features, labels):
                self.radiomics_features = torch.FloatTensor(radiomics_features)
                self.vit_features = torch.FloatTensor(vit_features)
                self.labels = torch.LongTensor(labels)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return self.vit_features[idx], self.radiomics_features[idx], self.labels[idx]
        
        # Create datasets with pre-computed features
        train1_dataset = FastMultimodalDataset(train1_rad, train1_vit_features, train1_labels)
        val1_dataset = FastMultimodalDataset(val1_rad, val1_vit_features, val1_labels)
        
        # Create data loaders
        train1_loader = DataLoader(train1_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val1_loader = DataLoader(val1_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Train Stage 1 with early stopping and more epochs
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(30):  # More epochs with early stopping
            # Train
            self.stage1_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for vit_features, rad_features, labels in train1_loader:
                vit_features = vit_features.to(self.device)
                rad_features = rad_features.to(self.device)
                labels = labels.to(self.device)
                
                self.stage1_trainer.optimizer.zero_grad()
                
                # Get teacher predictions from radiomics classifier
                teacher_probs = self.stage1_trainer.teacher_classifier.predict_proba(rad_features.cpu().numpy())
                teacher_logits = torch.FloatTensor(teacher_probs).to(self.device)
                
                # Get student predictions from ViT
                logits = self.stage1_model.forward_logits_only(vit_features, rad_features)
                
                # Use distillation loss
                loss = self.stage1_trainer.criterion(logits, teacher_logits, labels)
                
                loss.backward()
                self.stage1_trainer.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            train_loss /= len(train1_loader)
            
            # Validate
            self.stage1_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for vit_features, rad_features, labels in val1_loader:
                    vit_features = vit_features.to(self.device)
                    rad_features = rad_features.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Get teacher predictions from radiomics classifier
                    teacher_probs = self.stage1_trainer.teacher_classifier.predict_proba(rad_features.cpu().numpy())
                    teacher_logits = torch.FloatTensor(teacher_probs).to(self.device)
                    
                    # Get student predictions from ViT
                    logits = self.stage1_model.forward_logits_only(vit_features, rad_features)
                    
                    # Use distillation loss
                    loss = self.stage1_trainer.criterion(logits, teacher_logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss /= len(val1_loader)
            
            # Update scheduler
            self.stage1_trainer.scheduler.step(val_acc)
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.stage1_model.state_dict(),
                    'optimizer_state_dict': self.stage1_trainer.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.output_dir / 'stage1_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        print(f"Best Stage 1 Validation Accuracy: {best_val_acc:.2f}%")
        
        # Create and save Stage 1 ensemble
        from utils.radiomics_ensemble import create_ensemble
        
        # Train radiomics classifier for Stage 1
        from utils.radiomics_ensemble import train_radiomics_classifier
        rad_classifier, rad_train_acc = train_radiomics_classifier(
            self.radiomics_features, self.stage1_labels,
            model_type='random_forest',
            save_path=self.output_dir / 'stage1_radiomics_classifier.pkl'
        )
        
        # Create Stage 1 ensemble with optimized weights
        stage1_ensemble = create_ensemble(
            vit_model=self.stage1_model,
            radiomics_classifier=rad_classifier,
            vit_weight=0.3,  # 30% ViT weight
            radiomics_weight=0.7  # 70% radiomics weight
        )
        
        # Save Stage 1 ensemble
        import pickle
        with open(self.output_dir / 'stage1_ensemble.pkl', 'wb') as f:
            pickle.dump(stage1_ensemble, f)
        
        return {'best_val_acc': best_val_acc}
    
    def train_stage2_model(self):
        """Train Stage 2: Benign vs Malignant classifier."""
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING STAGE 2 MODEL (Benign vs Malignant)")
        print("=" * 80)
        
        # Create model for Stage 2 (2 classes)
        stage2_config = self.config.copy()
        stage2_config['num_classes'] = 2
        
        self.stage2_model = create_multimodal_model(
            vit_config={
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 256,  # Match reduced ViT
                'n_heads': 8,     # Match reduced ViT
                'n_layers': 8,    # Match reduced ViT
                'n_classes': 2,
                'dropout': 0.1
            },
            rad_dim=46,
            num_classes=2,
            fusion_type=stage2_config['fusion_type']
        ).to(self.device)
        
        # Create trainer for Stage 2 with fine-tuning learning rate
        self.stage2_trainer = MultimodalTrainer(
            self.stage2_model,
            self.device,
            learning_rate=1e-5,  # Critical for fine-tuning ViT
            weight_decay=stage2_config['weight_decay']
        )
        
        # Create data loaders for Stage 2
        from sklearn.model_selection import train_test_split
        # Split data for Stage 2
        train2_idx, val2_idx, train2_labels, val2_labels = train_test_split(
            range(len(self.stage2_image_paths)), self.stage2_labels,
            test_size=0.2, random_state=42, stratify=self.stage2_labels
        )
        
        train2_paths = [self.stage2_image_paths[i] for i in train2_idx]
        val2_paths = [self.stage2_image_paths[i] for i in val2_idx]
        train2_rad = self.stage2_radiomics[train2_idx]
        val2_rad = self.stage2_radiomics[val2_idx]
        
        # Create smaller ViT model for Stage 2
        from models.vit_model import create_vit_model
        vit_model2 = create_vit_model(
            img_size=224,
            patch_size=16,
            embed_dim=256,  # Reduced from 384
            n_heads=8,     # Reduced from 12
            n_layers=8,    # Reduced from 16
            n_classes=2
        ).to(self.device)
        
        # Fine-tune last 3 layers of ViT for Stage 2 (critical for benign vs malignant)
        # Freeze all layers first
        for param in vit_model2.parameters():
            param.requires_grad = False
        
        # Unfreeze last 3 transformer layers
        for param in vit_model2.transformer_blocks[-3:].parameters():
            param.requires_grad = True
        
        # Unfreeze classification head
        for param in vit_model2.head.parameters():
            param.requires_grad = True
        
        # Create strong medical augmentation transforms for Stage 2
        from utils.medical_augmentation import get_strong_stage2_transforms, get_medical_augmentation_transforms
        train_transform2 = get_strong_stage2_transforms(image_size=224)  # Strong augmentation for benign vs malignant
        val_transform2 = get_medical_augmentation_transforms(image_size=224, is_training=False)  # No augmentation for validation
        
        # Pre-compute ViT features for Stage 2
        from PIL import Image
        print("Pre-computing ViT features for Stage 2...")
        train2_vit_features = []
        val2_vit_features = []
        
        # Pre-compute training features
        for path in tqdm(train2_paths, desc="Extracting Stage 2 train ViT features"):
            image = Image.open(path).convert('RGB')
            if train_transform2:
                image = train_transform2(image)
            with torch.no_grad():
                features = vit_model2.extract_features(image.unsqueeze(0)).squeeze(0)
            train2_vit_features.append(features.cpu().numpy())
        
        # Pre-compute validation features
        for path in tqdm(val2_paths, desc="Extracting Stage 2 val ViT features"):
            image = Image.open(path).convert('RGB')
            if val_transform2:
                image = val_transform2(image)
            with torch.no_grad():
                features = vit_model2.extract_features(image.unsqueeze(0)).squeeze(0)
            val2_vit_features.append(features.cpu().numpy())
        
        train2_vit_features = np.array(train2_vit_features)
        val2_vit_features = np.array(val2_vit_features)
        print(f"Stage 2 ViT features pre-computed: {train2_vit_features.shape}")
        
        # Define FastMultimodalDataset class for Stage 2
        from torch.utils.data import Dataset
        class FastMultimodalDataset(Dataset):
            def __init__(self, radiomics_features, vit_features, labels):
                self.radiomics_features = torch.FloatTensor(radiomics_features)
                self.vit_features = torch.FloatTensor(vit_features)
                self.labels = torch.LongTensor(labels)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return self.vit_features[idx], self.radiomics_features[idx], self.labels[idx]
        
        # Create datasets with pre-computed features
        train2_dataset = FastMultimodalDataset(train2_rad, train2_vit_features, train2_labels)
        val2_dataset = FastMultimodalDataset(val2_rad, val2_vit_features, val2_labels)
        
        # Create balanced data loaders for Stage 2 using WeightedRandomSampler
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calculate class weights for balanced sampling
        class_weights = compute_class_weight('balanced', classes=np.unique(train2_labels), y=train2_labels)
        sample_weights = class_weights[train2_labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        # Create balanced sampler
        balanced_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Use balanced sampling for Stage 2 training
        train2_loader = DataLoader(train2_dataset, batch_size=self.config['batch_size'], sampler=balanced_sampler)
        
        # Regular loader for validation
        val2_loader = DataLoader(val2_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Train Stage 2 with radiomics ensemble
        print("\n" + "="*60)
        print("TRAINING RADIOMICS CLASSIFIER FOR STAGE 2")
        print("="*60)
        
        from utils.radiomics_ensemble import train_radiomics_classifier, create_ensemble
        
        # Train radiomics classifier separately
        rad_classifier, rad_train_acc = train_radiomics_classifier(
            train2_rad, train2_labels, 
            model_type='random_forest',
            save_path=self.output_dir / 'stage2_radiomics_classifier.pkl'
        )
        
        # Train ViT model as before
        stage2_history = self.stage2_trainer.train(
            train2_loader, val2_loader,
            epochs=self.config['epochs'],
            save_path=self.output_dir / 'stage2_model.pth'
        )
        
        # Create ensemble with optimized 20% ViT + 80% Radiomics for Stage 2
        self.stage2_ensemble = create_ensemble(
            vit_model=self.stage2_model,
            radiomics_classifier=rad_classifier,
            vit_weight=0.2,  # 20% ViT weight (optimized)
            radiomics_weight=0.8  # 80% radiomics weight (optimized)
        )
        
        # Evaluate ensemble on validation set
        print("\n" + "="*60)
        print("EVALUATING ENSEMBLE ON VALIDATION SET")
        print("="*60)
        
        val_metrics = self.stage2_ensemble.evaluate(
            val2_vit_features, val2_rad, val2_labels
        )
        
        print(f"ViT Validation Accuracy: {val_metrics['vit_accuracy']:.4f}")
        print(f"Radiomics Validation Accuracy: {val_metrics['radiomics_accuracy']:.4f}")
        print(f"Ensemble Validation Accuracy: {val_metrics['ensemble_accuracy']:.4f}")
        
        # Save ensemble
        with open(self.output_dir / 'stage2_ensemble.pkl', 'wb') as f:
            import pickle
            pickle.dump(self.stage2_ensemble, f)
        
        # Save Stage 2 results
        self.plot_training_history(stage2_history, self.output_dir / 'stage2_training_history.png')
        
        return stage2_history
        
    
    def evaluate_two_stage_pipeline(self):
        """Evaluate complete two-stage pipeline."""
        print("\n" + "=" * 80)
        print("STEP 4: EVALUATING TWO-STAGE PIPELINE")
        print("=" * 80)
        
        # Load trained models
        stage1_checkpoint = torch.load(self.output_dir / 'stage1_model.pth')
        self.stage1_model.load_state_dict(stage1_checkpoint['model_state_dict'])
        
        stage2_checkpoint = torch.load(self.output_dir / 'stage2_model.pth')
        self.stage2_model.load_state_dict(stage2_checkpoint['model_state_dict'])
        
        # Get test data splits
        from sklearn.model_selection import train_test_split
        
        # Stage 1 test split
        train1_idx, test1_idx, _, _ = train_test_split(
            range(len(self.image_paths)), self.stage1_labels, 
            test_size=0.2, random_state=42, stratify=self.stage1_labels
        )
        
        # Stage 2 test split (only abnormal samples)
        train2_idx, test2_idx, _, _ = train_test_split(
            range(len(self.stage2_image_paths)), self.stage2_labels,
            test_size=0.2, random_state=42, stratify=self.stage2_labels
        )
        
        # Evaluate Stage 1
        stage1_test_paths = [self.image_paths[i] for i in test1_idx]
        stage1_test_labels = self.stage1_labels[test1_idx]
        stage1_test_rad = self.radiomics_features[test1_idx]
        
        # Pre-compute ViT features for Stage 1 test
        from utils.preprocessing import get_data_transforms
        from PIL import Image
        from models.vit_model import create_vit_model
        from torch.utils.data import DataLoader, Dataset
        
        # Define FastMultimodalDataset class
        class FastMultimodalDataset(Dataset):
            def __init__(self, radiomics_features, vit_features, labels):
                self.radiomics_features = torch.FloatTensor(radiomics_features)
                self.vit_features = torch.FloatTensor(vit_features)
                self.labels = torch.LongTensor(labels)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return self.vit_features[idx], self.radiomics_features[idx], self.labels[idx]
        
        val_transform = get_data_transforms(image_size=224, is_training=False)
        
        print("Pre-computing Stage 1 test ViT features...")
        stage1_test_vit_features = []
        for path in tqdm(stage1_test_paths, desc="Extracting Stage 1 test ViT features"):
            image = Image.open(path).convert('RGB')
            if val_transform:
                image = val_transform(image)
            with torch.no_grad():
                vit_model = create_vit_model(img_size=224, patch_size=16, embed_dim=256, n_heads=8, n_layers=8, n_classes=2).to(self.device)
                features = vit_model.extract_features(image.unsqueeze(0)).squeeze(0)
            stage1_test_vit_features.append(features.cpu().numpy())
        
        stage1_test_vit_features = np.array(stage1_test_vit_features)
        
        # Create test dataset for Stage 1
        test1_dataset = FastMultimodalDataset(stage1_test_rad, stage1_test_vit_features, stage1_test_labels)
        test1_loader = DataLoader(test1_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Load Stage 1 ensemble
        with open(self.output_dir / 'stage1_ensemble.pkl', 'rb') as f:
            import pickle
            stage1_ensemble = pickle.load(f)
        
        # Evaluate Stage 1 ensemble
        stage1_correct = 0
        stage1_total = 0
        
        with torch.no_grad():
            for vit_features, rad_features, labels in test1_loader:
                vit_features = vit_features.to(self.device)
                rad_features = rad_features.cpu().numpy()
                labels = labels.cpu().numpy()
                
                # Get ensemble predictions
                ensemble_probs = stage1_ensemble.predict_proba(
                    vit_features.cpu().numpy(), rad_features
                )
                ensemble_pred = np.argmax(ensemble_probs, axis=1)
                
                stage1_correct += np.sum(ensemble_pred == labels)
                stage1_total += len(labels)
        
        stage1_acc = (stage1_correct / stage1_total) * 100
        print(f"Stage 1 Test Accuracy: {stage1_acc:.2f}%")
        
        # Evaluate Stage 2
        stage2_test_labels = self.stage2_labels[test2_idx]
        stage2_test_rad = self.stage2_radiomics[test2_idx]
        stage2_test_paths = [self.stage2_image_paths[i] for i in test2_idx]
        
        # Pre-compute ViT features for Stage 2 test
        print("Pre-computing Stage 2 test ViT features...")
        stage2_test_vit_features = []
        for path in tqdm(stage2_test_paths, desc="Extracting Stage 2 test ViT features"):
            image = Image.open(path).convert('RGB')
            if val_transform:
                image = val_transform(image)
            with torch.no_grad():
                vit_model2 = create_vit_model(img_size=224, patch_size=16, embed_dim=256, n_heads=8, n_layers=8, n_classes=2).to(self.device)
                features = vit_model2.extract_features(image.unsqueeze(0)).squeeze(0)
            stage2_test_vit_features.append(features.cpu().numpy())
        
        stage2_test_vit_features = np.array(stage2_test_vit_features)
        
        # Load Stage 2 ensemble
        with open(self.output_dir / 'stage2_ensemble.pkl', 'rb') as f:
            self.stage2_ensemble = pickle.load(f)
        
        # Evaluate Stage 2 ensemble
        val_metrics = self.stage2_ensemble.evaluate(
            stage2_test_vit_features, stage2_test_rad, stage2_test_labels
        )
        
        stage2_acc = val_metrics['ensemble_accuracy'] * 100
        print(f"Stage 2 Test Accuracy: {stage2_acc:.2f}%")
        
        # Calculate combined accuracy
        combined_accuracy = (stage1_acc + stage2_acc) / 2
        
        return combined_accuracy, f"Stage 1: {stage1_acc:.2f}%, Stage 2: {stage2_acc:.2f}%"
    
    def predict_stage2(self, image_path, radiomics_features):
        """Predict using Stage 2 model."""
        self.stage2_model.eval()
        with torch.no_grad():
            # Get ViT features (simplified for this example)
            vit_features = torch.randn(1, 384).to(self.device)  # Placeholder
            rad_tensor = torch.FloatTensor(radiomics_features).unsqueeze(0).to(self.device)
            
            logits = self.stage2_model(vit_features, rad_tensor)
            pred = torch.argmax(logits, dim=1).item()
            return pred
    
    def plot_training_history(self, history, save_path):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Malignant', 'Normal'],
                   yticklabels=['Benign', 'Malignant', 'Normal'])
        plt.title('Two-Stage Pipeline Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def run_complete_pipeline(self):
        """Run the complete two-stage pipeline."""
        print("🎯 OPTIMIZED TWO-STAGE LUNG CANCER DETECTION PIPELINE")
        print("=" * 80)
        print("Stage 1: Normal vs Abnormal (30% ViT + 70% Radiomics)")
        print("Stage 2: Benign vs Malignant (20% ViT + 80% Radiomics)")
        print("=" * 80)
        print("Configuration: Optimized ViT-Radiomics Fusion Weights")
        print("-" * 80)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Train Stage 1 model
        stage1_history = self.train_stage1_model()
        
        # Step 3: Train Stage 2 model
        stage2_history = self.train_stage2_model()
        
        # Step 4: Evaluate complete pipeline
        final_accuracy, final_report = self.evaluate_two_stage_pipeline()
        
        return final_accuracy, final_report

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Lung Cancer Detection Pipeline')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/balanced_dataset', 
                       help='Path to dataset directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=60, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, 
                       help='Weight decay')
    
    # Model arguments
    parser.add_argument('--embed-dim', type=int, default=384,
                       help='Embedding dimension')
    parser.add_argument('--n-heads', type=int, default=12, 
                       help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=16, 
                       help='Number of transformer layers')
    parser.add_argument('--fusion-type', type=str, default='concat', 
                       choices=['attention', 'concat', 'gated'],
                       help='Type of fusion mechanism')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/two_stage_pipeline',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'embed_dim': args.embed_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'fusion_type': args.fusion_type,
        'output_dir': args.output_dir,
        'device': args.device
    }
    
    # Create and run pipeline
    pipeline = TwoStagePipeline(config)
    accuracy, report = pipeline.run_complete_pipeline()
    
    print(f"\n🎉 TWO-STAGE PIPELINE COMPLETED!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()
