"""
Radiomics + ViT Ensemble for Medical Classification
Train radiomics separately and ensemble with ViT predictions
Final prediction = 0.7 * ViT + 0.3 * Radiomics
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os


class RadiomicsClassifier:
    """
    Separate radiomics classifier for texture-based analysis.
    """
    
    def __init__(self, model_type='random_forest', n_estimators=100):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.trained = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, radiomics_features, labels):
        """
        Train radiomics classifier.
        
        Args:
            radiomics_features: [n_samples, n_features]
            labels: [n_samples]
        """
        # Standardize features
        radiomics_scaled = self.scaler.fit_transform(radiomics_features)
        
        # Train model
        self.model.fit(radiomics_scaled, labels)
        self.trained = True
        
        # Get training accuracy
        train_pred = self.model.predict(radiomics_scaled)
        train_acc = np.mean(train_pred == labels)
        
        print(f"Radiomics {self.model_type} training accuracy: {train_acc:.4f}")
        
        return train_acc
    
    def predict_proba(self, radiomics_features):
        """
        Get prediction probabilities.
        
        Args:
            radiomics_features: [n_samples, n_features]
            
        Returns:
            probabilities: [n_samples, n_classes]
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        radiomics_scaled = self.scaler.transform(radiomics_features)
        return self.model.predict_proba(radiomics_scaled)
    
    def predict(self, radiomics_features):
        """
        Get predictions.
        
        Args:
            radiomics_features: [n_samples, n_features]
            
        Returns:
            predictions: [n_samples]
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        radiomics_scaled = self.scaler.transform(radiomics_features)
        return self.model.predict(radiomics_scaled)
    
    def save(self, path):
        """Save model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'trained': self.trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path):
        """Load model and scaler."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.trained = model_data['trained']


class ViTRadiomicsEnsemble:
    """
    Ensemble ViT and Radiomics predictions with radiomics as primary feature.
    Default: Final prediction = 0.2 * ViT + 0.8 * Radiomics
    """
    
    def __init__(self, vit_model, radiomics_classifier, vit_weight=0.2, radiomics_weight=0.8):
        self.vit_model = vit_model
        self.radiomics_classifier = radiomics_classifier
        self.vit_weight = vit_weight
        self.radiomics_weight = radiomics_weight
        
        # Ensure weights sum to 1
        total_weight = vit_weight + radiomics_weight
        self.vit_weight /= total_weight
        self.radiomics_weight /= total_weight
    
    def predict_proba(self, vit_features, radiomics_features):
        """
        Get ensemble prediction probabilities.
        
        Args:
            vit_features: [n_samples, vit_dim]
            radiomics_features: [n_samples, rad_dim]
            
        Returns:
            ensemble_probs: [n_samples, n_classes]
        """
        # Get ViT predictions
        self.vit_model.eval()
        with torch.no_grad():
            vit_features_tensor = torch.FloatTensor(vit_features)
            radiomics_tensor = torch.FloatTensor(radiomics_features)
            
            # Handle multimodal model that expects both inputs
            if hasattr(self.vit_model, 'forward_logits_only'):
                vit_logits = self.vit_model.forward_logits_only(vit_features_tensor, radiomics_tensor)
            else:
                # Standard forward pass
                vit_logits, _ = self.vit_model(vit_features_tensor, radiomics_tensor)
            
            vit_probs = torch.softmax(vit_logits, dim=1).numpy()
        
        # Get Radiomics predictions
        rad_probs = self.radiomics_classifier.predict_proba(radiomics_features)
        
        # Ensemble predictions
        ensemble_probs = (self.vit_weight * vit_probs + 
                         self.radiomics_weight * rad_probs)
        
        return ensemble_probs
    
    def predict(self, vit_features, radiomics_features):
        """
        Get ensemble predictions.
        
        Args:
            vit_features: [n_samples, vit_dim]
            radiomics_features: [n_samples, rad_dim]
            
        Returns:
            predictions: [n_samples]
        """
        ensemble_probs = self.predict_proba(vit_features, radiomics_features)
        return np.argmax(ensemble_probs, axis=1)
    
    def evaluate(self, vit_features, radiomics_features, labels):
        """
        Evaluate ensemble performance.
        
        Args:
            vit_features: [n_samples, vit_dim]
            radiomics_features: [n_samples, rad_dim]
            labels: [n_samples]
            
        Returns:
            metrics: dict with evaluation metrics
        """
        # Get ensemble predictions
        ensemble_preds = self.predict(vit_features, radiomics_features)
        ensemble_probs = self.predict_proba(vit_features, radiomics_features)
        
        # Calculate accuracy
        accuracy = np.mean(ensemble_preds == labels)
        
        # Detailed metrics
        cm = confusion_matrix(labels, ensemble_preds)
        report = classification_report(labels, ensemble_preds, digits=4, output_dict=True)
        
        # Individual model accuracies
        self.vit_model.eval()
        with torch.no_grad():
            vit_features_tensor = torch.FloatTensor(vit_features)
            radiomics_tensor = torch.FloatTensor(radiomics_features)
            
            # Handle multimodal model that expects both inputs
            if hasattr(self.vit_model, 'forward_logits_only'):
                vit_logits = self.vit_model.forward_logits_only(vit_features_tensor, radiomics_tensor)
            else:
                # Standard forward pass
                vit_logits, _ = self.vit_model(vit_features_tensor, radiomics_tensor)
            vit_preds = torch.argmax(vit_logits, dim=1).numpy()
        
        vit_acc = np.mean(vit_preds == labels)
        rad_acc = np.mean(self.radiomics_classifier.predict(radiomics_features) == labels)
        
        metrics = {
            'ensemble_accuracy': accuracy,
            'vit_accuracy': vit_acc,
            'radiomics_accuracy': rad_acc,
            'confusion_matrix': cm,
            'classification_report': report,
            'ensemble_probs': ensemble_probs
        }
        
        return metrics


def train_radiomics_classifier(radiomics_features, labels, model_type='random_forest', save_path=None):
    """
    Train a radiomics classifier.
    
    Args:
        radiomics_features: [n_samples, n_features]
        labels: [n_samples]
        model_type: 'random_forest', 'svm', or 'logistic'
        save_path: Path to save the trained model
        
    Returns:
        trained RadiomicsClassifier
    """
    classifier = RadiomicsClassifier(model_type=model_type)
    train_acc = classifier.train(radiomics_features, labels)
    
    if save_path:
        classifier.save(save_path)
        print(f"Radiomics classifier saved to: {save_path}")
    
    return classifier, train_acc


def create_ensemble(vit_model, radiomics_classifier, vit_weight=0.7, radiomics_weight=0.3):
    """
    Create ViT + Radiomics ensemble.
    
    Args:
        vit_model: Trained ViT model
        radiomics_classifier: Trained radiomics classifier
        vit_weight: Weight for ViT predictions
        radiomics_weight: Weight for radiomics predictions
        
    Returns:
        ViTRadiomicsEnsemble
    """
    return ViTRadiomicsEnsemble(
        vit_model=vit_model,
        radiomics_classifier=radiomics_classifier,
        vit_weight=vit_weight,
        radiomics_weight=radiomics_weight
    )
