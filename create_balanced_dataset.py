"""
Create balanced dataset with 1,500 samples per class using random sampling.
"""

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def create_balanced_dataset():
    """Create balanced dataset with 1,500 images per class."""
    
    # Configuration
    source_dir = Path("data/raw/Dataset")
    target_dir = Path("data/balanced_dataset")
    samples_per_class = 1500
    
    # Class mapping (original -> balanced names)
    class_mapping = {
        "Bengin cases": "Bengin cases",
        "Malignant cases": "Malignant cases", 
        "Normal cases": "Normal cases"
    }
    
    print("🎯 Creating Balanced Dataset")
    print("=" * 50)
    print(f"Samples per class: {samples_per_class}")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    # Create target directories
    target_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    for original_class, balanced_class in class_mapping.items():
        print(f"📁 Processing {original_class} -> {balanced_class}")
        
        # Source and target paths
        source_path = source_dir / original_class
        target_path = target_dir / balanced_class
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Get all JPG images
        image_files = list(source_path.glob("*.jpg"))
        print(f"   Found {len(image_files)} images")
        
        # Random sampling
        if len(image_files) <= samples_per_class:
            selected_files = image_files
            print(f"   Using all {len(selected_files)} images (less than target)")
        else:
            selected_files = random.sample(image_files, samples_per_class)
            print(f"   Randomly selected {len(selected_files)} images")
        
        # Copy files
        copied_count = 0
        for img_file in tqdm(selected_files, desc=f"   Copying {balanced_class}"):
            target_file = target_path / img_file.name
            shutil.copy2(img_file, target_file)
            copied_count += 1
        
        total_copied += copied_count
        print(f"   ✅ Copied {copied_count} images")
        print()
    
    print("=" * 50)
    print(f"🎉 Balanced Dataset Created!")
    print(f"Total images copied: {total_copied}")
    print(f"Dataset location: {target_dir}")
    
    # Verify the balanced dataset
    print("\n📊 Dataset Verification:")
    for balanced_class in class_mapping.values():
        class_path = target_dir / balanced_class
        count = len(list(class_path.glob("*.jpg")))
        print(f"   {balanced_class}: {count} images")
    
    return target_dir

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create balanced dataset
    balanced_dataset_path = create_balanced_dataset()
    
    print(f"\n✅ Ready to train on balanced dataset!")
    print(f"Update dataset path in run_multimodal_xai_complete.py to:")
    print(f"   {balanced_dataset_path}")
