"""
Complete multimodal + XAI lung cancer detection pipeline.
Integrates radiomics, multimodal fusion, and explainability for comprehensive analysis.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from utils.preprocessing import load_dataset, create_data_loaders, get_data_transforms
from utils.extract_radiomics_complete import extract_radiomics_for_dataset, RadiomicsExtractor
from models.multimodal_fusion_complete import create_multimodal_model
from utils.xai_complete import create_explainer
from training.multimodal_train import MultimodalTrainer


class CompleteMultimodalPipeline:
    """
    Complete pipeline for multimodal lung cancer detection with XAI.
    """
    
    def __init__(self, config):
        """
        Initialize the complete pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize components
        self.radiomics_extractor = None
        self.model = None
        self.explainer = None
        self.trainer = None
        
        # Data
        self.image_paths = None
        self.labels = None
        self.radiomics_features = None
        self.feature_names = None
        
        print("Complete Multimodal + XAI Pipeline Initialized")
    
    def load_and_preprocess_data(self):
        """Load and preprocess all data including radiomics features."""
        print("\n" + "="*80)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*80)
        
        # Load image paths and labels
        print("Loading image paths and labels...")
        self.image_paths, self.labels = load_dataset(
            self.config['data_path'], 
            image_size=self.config['image_size']
        )
        
        print(f"Loaded {len(self.image_paths)} images from {len(np.unique(self.labels))} classes")
        
        # Extract radiomics features using optimized function
        print("Extracting radiomics features...")
        from utils.extract_radiomics_complete import extract_radiomics_for_dataset
        
        # Use the optimized extraction function with parallel processing and caching
        self.radiomics_features, self.feature_names = extract_radiomics_for_dataset(self.image_paths)
        
        print(f"Extracted {self.radiomics_features.shape[1]} radiomics features")
        print(f"Radiomics features shape: {self.radiomics_features.shape}")
        
        # Create data splits
        print("Creating train/val/test splits...")
        
        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels, train_val_rad, test_rad = train_test_split(
            self.image_paths, self.labels, self.radiomics_features,
            test_size=0.15, random_state=42, stratify=self.labels
        )
        
        # Second split: train vs val
        train_paths, val_paths, train_labels, val_labels, train_rad, val_rad = train_test_split(
            train_val_paths, train_val_labels, train_val_rad,
            test_size=0.1765, random_state=42, stratify=train_val_labels
        )
        
        print(f"Data splits:")
        print(f"  Train: {len(train_paths)} samples")
        print(f"  Validation: {len(val_paths)} samples")
        print(f"  Test: {len(test_paths)} samples")
        
        # Store splits
        self.data_splits = {
            'train': (train_paths, train_labels, train_rad),
            'val': (val_paths, val_labels, val_rad),
            'test': (test_paths, test_labels, test_rad)
        }
        
        return self.data_splits
    
    def create_model(self):
        """Create the multimodal fusion model."""
        print("\n" + "="*80)
        print("STEP 2: CREATING MULTIMODAL MODEL")
        print("="*80)
        
        # Model configuration
        vit_config = {
            'img_size': self.config['image_size'],
            'patch_size': self.config['patch_size'],
            'embed_dim': self.config['embed_dim'],
            'n_heads': self.config['n_heads'],
            'n_layers': self.config['n_layers'],
            'n_classes': 3,
            'dropout': 0.1
        }
        
        # Create multimodal model
        self.model = create_multimodal_model(
            vit_config=vit_config,
            rad_dim=self.radiomics_features.shape[1],
            num_classes=3,
            fusion_type=self.config['fusion_type']
        )
        
        self.model.to(self.device)
        
        print(f"Multimodal model created:")
        print(f"  ViT config: {vit_config}")
        print(f"  Radiomics dimension: {self.radiomics_features.shape[1]}")
        print(f"  Fusion type: {self.config['fusion_type']}")
        
        return self.model
    
    def train_model(self):
        """Train the multimodal model."""
        print("\n" + "="*80)
        print("STEP 3: TRAINING MULTIMODAL MODEL")
        print("="*80)
        
        # Create trainer
        self.trainer = MultimodalTrainer(
            self.model, 
            self.device,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Create data loaders
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        
        class MultimodalDataset(Dataset):
            def __init__(self, image_paths, labels, radiomics, vit_features, transform):
                self.image_paths = image_paths
                self.labels = labels
                self.radiomics = torch.FloatTensor(radiomics)
                self.vit_features = torch.FloatTensor(vit_features)  # Pre-extracted
                self.transform = transform
                
            def __len__(self):
                return len(self.image_paths)
                
            def __getitem__(self, idx):
                # Use pre-extracted ViT features - no image loading needed!
                vit_features = self.vit_features[idx]
                radiomics = self.radiomics[idx]
                label = self.labels[idx]
                return vit_features, radiomics, label
        
        transform = get_data_transforms(self.config['image_size'], is_training=True)
        val_transform = get_data_transforms(self.config['image_size'], is_training=False)
        
        train_paths, train_labels, train_rad = self.data_splits['train']
        val_paths, val_labels, val_rad = self.data_splits['val']
        
        # PRE-EXTRACT VIT FEATURES - This is the key optimization!
        print("Pre-extracting ViT features to speed up training...")
        train_vit_features = self._extract_vit_features_batch(train_paths, transform)
        val_vit_features = self._extract_vit_features_batch(val_paths, val_transform)
        print("ViT features extracted successfully!")
        
        train_dataset = MultimodalDataset(train_paths, train_labels, train_rad, train_vit_features, transform)
        val_dataset = MultimodalDataset(val_paths, val_labels, val_rad, val_vit_features, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        
        # Train model
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Batch size: {self.config['batch_size']}")
        
        start_time = time.time()
        
        save_path = os.path.join(self.config['output_dir'], "best_multimodal_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['epochs'],
            save_path=save_path
        )
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
        
        # Store history for plotting
        self.history = history
        
        return history
    
    def _extract_vit_features_batch(self, image_paths, transform):
        """Extract ViT features for a batch of images once with caching."""
        from models.vit_model import create_vit_model
        from PIL import Image
        import hashlib
        import pickle
        from pathlib import Path
        
        # Create cache directory
        cache_dir = Path("vit_features_cache")
        cache_dir.mkdir(exist_ok=True)
        
        vit_model = create_vit_model(
            img_size=self.config['image_size'],
            patch_size=self.config['patch_size'],
            embed_dim=self.config['embed_dim'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers'],
            n_classes=3,
            dropout=0.1
        )
        vit_model.to(self.device)
        vit_model.eval()
        
        features_list = []
        cached_count = 0
        
        print(f"Extracting ViT features for {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            # Generate cache key
            cache_key = hashlib.md5(str(image_path).encode()).hexdigest()
            cache_path = cache_dir / f"{cache_key}.pkl"
            
            # Check cache first
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        vit_feat = pickle.load(f)
                    features_list.append(vit_feat)
                    cached_count += 1
                    continue
                except:
                    pass  # Fall back to extraction
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            if transform:
                image = transform(image)
            
            # Extract ViT features
            with torch.no_grad():
                vit_feat = vit_model.extract_features(image.unsqueeze(0).to(self.device))
                vit_feat = vit_feat.squeeze(0).cpu()
                
                # Cache the result
                with open(cache_path, 'wb') as f:
                    pickle.dump(vit_feat, f)
                
                features_list.append(vit_feat)
            
            if (i + 1) % 500 == 0 or i == len(image_paths) - 1:
                print(f"  Processed {i + 1}/{len(image_paths)} images (cached: {cached_count})")
        
        print(f"ViT features cached: {len(list(cache_dir.glob('*.pkl')))} files")
        return torch.stack(features_list)
    
    def setup_explainer(self):
        """Setup the XAI explainer."""
        print("\n" + "="*80)
        print("STEP 4: SETUP XAI EXPLAINER")
        print("="*80)
        
        # Load best model
        best_model_path = os.path.join(self.config['output_dir'], "best_multimodal_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Best model loaded for XAI analysis")
        
        # Create explainer
        self.explainer = create_explainer(self.model, self.device)
        print("XAI explainer created successfully")
        
        return self.explainer
    
    def evaluate_with_explanations(self):
        """Evaluate model with comprehensive XAI explanations."""
        print("\n" + "="*80)
        print("STEP 5: EVALUATION WITH XAI EXPLANATIONS")
        print("="*80)
        
        # Get test data
        test_paths, test_labels, test_rad = self.data_splits['test']
        
        # Extract ViT features for test set
        print("Extracting ViT features for test set...")
        transform = get_data_transforms(self.config['image_size'], is_training=False)
        test_vit_features = self._extract_vit_features_batch(test_paths, transform)
        
        # Evaluate model
        print("Evaluating model on test set...")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for i, (path, label, rad_features) in enumerate(zip(test_paths, test_labels, test_rad)):
                # Use pre-extracted ViT features
                vit_tensor = test_vit_features[i].unsqueeze(0).to(self.device)
                rad_tensor = torch.FloatTensor(rad_features).unsqueeze(0).to(self.device)
                
                # Predict
                outputs = self.model(vit_tensor, rad_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend([label])
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Generate explanations for first few samples
                if i < 5:
                    print(f"Generating explanation for sample {i+1}...")
                    # Load original image for explanation
                    from PIL import Image
                    image = Image.open(path).convert('RGB')
                    transform = get_data_transforms(self.config['image_size'], is_training=False)
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    explanation = self.explainer.generate_complete_explanation(
                        image_tensor, rad_tensor, predicted[0].item(), label
                    )
                    
                    # Save explanation plots
                    save_path = os.path.join(
                        self.config['output_dir'], 
                        f'explanations', 
                        f'explanation_sample_{i+1}.png'
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    self.explainer.save_explanation_plots(
                        explanation, image_tensor, save_path, 
                        class_names=['Benign', 'Malignant', 'Normal']
                    )
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        class_names = ['Benign', 'Malignant', 'Normal']
        
        print('\n' + '='*80)
        print('FINAL RESULTS WITH XAI')
        print('='*80)
        print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        
        print('\nDetailed Classification Report:')
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        print(report)
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names)
        plt.title('Confusion Matrix - Multimodal + XAI Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_save_path = os.path.join(self.config['output_dir'], 'confusion_matrix.png')
        plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to: {cm_save_path}')
        plt.close()
        
        # Generate training history plots if available
        if hasattr(self, 'history') and self.history:
            self.plot_training_history()
        
        # Generate per-class performance metrics
        self.plot_class_performance(all_labels, all_predictions, class_names)
        
        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        
        # Convert test data to tensors
        test_images = []
        test_rad_tensors = []
        
        for path in test_paths[:50]:  # Sample for feature importance
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image)
            test_images.append(image_tensor)
        
        test_images = torch.stack(test_images)
        test_rad_tensors = torch.FloatTensor(test_rad[:50])
        test_labels_tensor = torch.LongTensor(test_labels[:50])
        
        # Feature importance
        feature_importance = self.explainer.feature_importance_analysis(
            test_images, test_rad_tensors, test_labels_tensor
        )
        
        if 'rad_importance' in feature_importance:
            print("\nTop 10 Most Important Radiomics Features:")
            importance_scores = feature_importance['rad_importance']
            top_indices = np.argsort(importance_scores)[-10:][::-1]
            
            for i, idx in enumerate(top_indices):
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
                print(f"  {i+1}. {feature_name}: {importance_scores[idx]:.4f}")
        
        print('\n' + '='*80)
        print('MULTIMODAL + XAI PIPELINE COMPLETED')
        print('='*80)
        print(f'Final Accuracy: {accuracy*100:.2f}%')
        print('XAI explanations saved to: outputs/explanations/')
        print('Feature importance analysis completed')
        
        return accuracy, report, feature_importance
    
    def plot_training_history(self):
        """Plot training history graphs."""
        if not hasattr(self, 'history') or not self.history:
            return
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot training & validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy', color='blue')
        plt.plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        if 'learning_rate' in self.history:
            plt.plot(self.history['learning_rate'], label='Learning Rate', color='green')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save training history plot
        history_save_path = os.path.join(self.config['output_dir'], 'training_history.png')
        plt.savefig(history_save_path, dpi=300, bbox_inches='tight')
        print(f'Training history plot saved to: {history_save_path}')
        plt.close()
    
    def plot_class_performance(self, true_labels, pred_labels, class_names):
        """Plot per-class performance metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Calculate per-class metrics
        precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
        recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)
        
        # Create performance DataFrame
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=class_names)
        
        # Plot performance metrics
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Per-Class Performance Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metrics')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save performance plot
        perf_save_path = os.path.join(self.config['output_dir'], 'class_performance.png')
        plt.savefig(perf_save_path, dpi=300, bbox_inches='tight')
        print(f'Class performance plot saved to: {perf_save_path}')
        plt.close()
        
        # Print detailed metrics
        print("\nPer-Class Performance Summary:")
        print("="*50)
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  Precision: {precision[i]:.3f}")
            print(f"  Recall: {recall[i]:.3f}")
            print(f"  F1-Score: {f1[i]:.3f}")
        print("="*50)
    
    def run_complete_pipeline(self):
        """Run the complete multimodal + XAI pipeline."""
        print("STARTING COMPLETE MULTIMODAL + XAI PIPELINE")
        print("="*80)
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Create model
        self.create_model()
        
        # Step 3: Train model
        self.train_model()
        
        # Step 4: Setup explainer
        self.setup_explainer()
        
        # Step 5: Evaluate with explanations
        accuracy, report, feature_importance = self.evaluate_with_explanations()
        
        return accuracy, report, feature_importance


def main():
    parser = argparse.ArgumentParser(description='Complete Multimodal + XAI Lung Cancer Detection')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/balanced_dataset', 
                       help='Path to dataset directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, 
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, 
                       help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--image-size', type=int, default=224, 
                       help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16, 
                       help='Patch size for ViT')
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
    parser.add_argument('--output-dir', type=str, default='outputs/balanced_dataset_experiment',
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
        'patience': args.patience,
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'fusion_type': args.fusion_type,
        'output_dir': args.output_dir,
        'device': args.device
    }
    
    print("COMPLETE MULTIMODAL + XAI LUNG CANCER DETECTION")
    print("="*80)
    print("Features:")
    print("  - Radiomics feature extraction")
    print("  - Multimodal fusion (ViT + Radiomics)")
    print("  - Cross-modal attention")
    print("  - Grad-CAM explanations")
    print("  - Feature importance analysis")
    print("  - SHAP explanations")
    print("="*80)
    
    # Create and run pipeline
    pipeline = CompleteMultimodalPipeline(config)
    accuracy, report, feature_importance = pipeline.run_complete_pipeline()
    
    print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
