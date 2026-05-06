"""
Balanced Batch Sampler for Medical Classification
Ensures equal representation of each class in every batch
Critical for Benign vs Malignant classification
"""

import torch
from torch.utils.data import Sampler, BatchSampler
import numpy as np
from collections import defaultdict


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler that ensures equal samples from each class.
    
    Args:
        labels: List of labels for each sample
        batch_size: Total batch size (must be even for 2-class balance)
        shuffle: Whether to shuffle within classes
    """
    
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
        
        # Validate batch size
        num_classes = len(self.class_indices)
        if batch_size % num_classes != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by number of classes {num_classes}")
        
        self.samples_per_class = batch_size // num_classes
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
    def __iter__(self):
        # Shuffle indices within each class
        if self.shuffle:
            for class_idx in self.class_indices:
                np.random.shuffle(self.class_indices[class_idx])
        
        # Create balanced batches
        batches = []
        for batch_idx in range(self.num_batches):
            batch = []
            for class_idx in sorted(self.class_indices.keys()):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(self.class_indices[class_idx][start_idx:end_idx])
            
            if len(batch) == self.batch_size:
                batches.append(batch)
        
        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)
        
        # Flatten to single list of indices
        for batch in batches:
            yield from batch
    
    def __len__(self):
        return self.num_batches * self.batch_size


def create_balanced_dataloader(dataset, labels, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a balanced data loader.
    
    Args:
        dataset: PyTorch dataset
        labels: Labels for each sample in dataset
        batch_size: Batch size (must be even for 2-class balance)
        shuffle: Whether to shuffle
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader with balanced sampling
    """
    balanced_sampler = BalancedBatchSampler(labels, batch_size, shuffle)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=balanced_sampler,
        num_workers=num_workers
    )


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler that maintains class distribution in each batch.
    More flexible than balanced sampling for datasets with unequal class sizes.
    """
    
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        
        # Calculate total number of batches
        self.total_samples = len(labels)
        self.num_batches = (self.total_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        # Shuffle indices within each class
        if self.shuffle:
            for class_idx in self.class_indices:
                np.random.shuffle(self.class_indices[class_idx])
        
        # Create iterators for each class
        class_iterators = {}
        for class_idx, indices in self.class_indices.items():
            class_iterators[class_idx] = iter(indices)
        
        # Create batches with stratified sampling
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Calculate class proportions for this batch
            remaining_samples = self.total_samples - batch_idx * self.batch_size
            current_batch_size = min(self.batch_size, remaining_samples)
            
            # Sample from each class proportionally
            for class_idx in self.class_indices:
                class_size = len(self.class_indices[class_idx])
                class_proportion = class_size / self.total_samples
                class_batch_size = max(1, int(current_batch_size * class_proportion))
                
                # Add samples from this class
                for _ in range(class_batch_size):
                    try:
                        batch.append(next(class_iterators[class_idx]))
                    except StopIteration:
                        # Restart iterator if we run out
                        if self.shuffle:
                            np.random.shuffle(self.class_indices[class_idx])
                        class_iterators[class_idx] = iter(self.class_indices[class_idx])
                        batch.append(next(class_iterators[class_idx]))
            
            # Ensure batch size is correct
            while len(batch) < current_batch_size:
                # Add random samples to fill batch
                random_class = np.random.choice(list(self.class_indices.keys()))
                try:
                    batch.append(next(class_iterators[random_class]))
                except StopIteration:
                    if self.shuffle:
                        np.random.shuffle(self.class_indices[random_class])
                    class_iterators[random_class] = iter(self.class_indices[random_class])
                    batch.append(next(class_iterators[random_class]))
            
            # Shuffle batch order
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield from batch[:current_batch_size]
    
    def __len__(self):
        return self.total_samples
