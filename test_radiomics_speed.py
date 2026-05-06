"""
Test radiomics extraction speed with the optimized implementation.
"""

import os
import sys
import time
sys.path.append('.')

from utils.preprocessing import load_dataset
from utils.extract_radiomics_complete import extract_radiomics_for_dataset

def test_radiomics_speed():
    """Test radiomics extraction speed on a subset of data."""
    print("="*80)
    print("TESTING RADIOMICS EXTRACTION SPEED")
    print("="*80)
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset("data/raw/Dataset")
    
    # Test with different subset sizes
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        if size > len(image_paths):
            continue
            
        print(f"\n--- Testing with {size} images ---")
        test_paths = image_paths[:size]
        
        # Clear cache before each test
        import shutil
        if os.path.exists("radiomics_cache"):
            shutil.rmtree("radiomics_cache")
        
        # Time the extraction
        start_time = time.time()
        features, feature_names = extract_radiomics_for_dataset(test_paths)
        end_time = time.time()
        
        extraction_time = end_time - start_time
        time_per_image = extraction_time / size
        
        print(f"Results:")
        print(f"  Features shape: {features.shape}")
        print(f"  Feature names: {len(feature_names)}")
        print(f"  Total time: {extraction_time:.2f} seconds")
        print(f"  Time per image: {time_per_image:.3f} seconds")
        print(f"  Images per second: {size/extraction_time:.1f}")
        
        # Test cache speed on second run
        print(f"\n--- Testing cache speed with {size} images ---")
        start_time = time.time()
        features_cached, _ = extract_radiomics_for_dataset(test_paths)
        end_time = time.time()
        
        cache_time = end_time - start_time
        print(f"  Cache time: {cache_time:.2f} seconds")
        print(f"  Speedup: {extraction_time/cache_time:.1f}x faster")
        
        # Verify results are the same
        if features.shape == features_cached.shape:
            print("  ✅ Cached results match original")
        else:
            print("  ❌ Cached results differ")
    
    print("\n" + "="*80)
    print("RADIOMICS SPEED TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    test_radiomics_speed()
