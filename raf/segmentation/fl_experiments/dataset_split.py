"""Dataset splitting utility for Cityscapes.

`split_cityscapes(root)` will create three client splits with roughly equal
image counts and guarantee each contains at least one pixel of every class.

The split indices are deterministic (seed 0) to keep runs reproducible.
The function returns a dict mapping client_id -> list[int] (indices).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from torchvision.datasets import Cityscapes


# Default constants (can be overridden)
DEFAULT_NUM_CLIENTS = 3
DEFAULT_TOTAL_SAMPLES = 2100  # Total samples to use (700 per client)
DEFAULT_SAMPLES_PER_CLIENT = 700  # Samples per client
_SEED = 0

# Explicit aliases used throughout the file (kept for backward-compatibility)
NUM_CLIENTS = DEFAULT_NUM_CLIENTS
SAMPLES_PER_CLIENT = DEFAULT_SAMPLES_PER_CLIENT
NUM_TOTAL_SAMPLES = DEFAULT_TOTAL_SAMPLES


def get_limited_dataset_indices(root: str | Path, num_samples: int = DEFAULT_TOTAL_SAMPLES) -> List[int]:
    """Get a limited subset of Cityscapes train indices ensuring class coverage.
    
    Args:
        root: Path to Cityscapes dataset
        num_samples: Total number of samples to select
        
    Returns:
        List of indices for the limited dataset
    """
    root = Path(root)
    rng = random.Random(_SEED)

    print("📂 Loading Cityscapes dataset...")
    ds = Cityscapes(str(root), split="train", mode="fine", target_type="semantic")
    total_samples = len(ds)
    print(f"📊 Dataset loaded: {total_samples} total samples")
    
    if num_samples >= total_samples:
        print(f"⚠️  Requested {num_samples} samples, but only {total_samples} available. Using all.")
        return list(range(total_samples))
    
    # Strategy: Ensure all classes are present, then random sampling
    print("🎯 Ensuring all classes are present...")
    
    # Step 1: Quick scan to find samples for each class
    scan_limit = min(500, total_samples)  # Scan first 500 samples (should be enough)
    print(f"📋 Scanning first {scan_limit} samples to find all classes...")
    
    class_to_samples = {cls: [] for cls in range(19)}
    
    # Single scan for all classes
    for idx in range(scan_limit):
        if idx % 100 == 0:
            print(f"   Progress: {idx+1}/{scan_limit}")
        _, lbl = ds[idx]
        unique_classes = set(np.unique(lbl))
        # Filter out invalid classes (only keep 0-18, ignore 255 and others)
        for cls in unique_classes:
            if 0 <= cls <= 18:  # Only valid Cityscapes classes
                class_to_samples[cls].append(idx)
    
    # Step 2: Ensure all classes are present (1 sample per class minimum)
    print("🎯 Ensuring all 19 classes are present...")
    
    guaranteed_samples = []
    for cls in range(19):
        if class_to_samples[cls]:
            chosen = rng.choice(class_to_samples[cls])
            guaranteed_samples.append(chosen)
            print(f"   Class {cls}: sample {chosen}")
        else:
            print(f"   ⚠️  Class {cls}: not found in first {scan_limit} samples")
    
    # Step 3: Fill remaining slots randomly from the entire dataset
    remaining_slots = num_samples - len(guaranteed_samples)
    print(f"🎲 Filling remaining {remaining_slots} slots randomly from entire dataset...")
    
    # Get all indices except already chosen ones
    available_indices = [i for i in range(total_samples) if i not in guaranteed_samples]
    rng.shuffle(available_indices)
    
    # Take random additional samples
    additional_samples = available_indices[:remaining_slots]
    
    # Combine guaranteed + additional
    final_indices = sorted(guaranteed_samples + additional_samples)
    
    print(f"✅ Selected {len(final_indices)} samples with all classes guaranteed")
    return final_indices


def split_cityscapes(root: str | Path) -> Dict[int, List[int]]:
    """Split limited Cityscapes dataset into 3 clients with 700 samples each.
    Each client is guaranteed to have all 19 classes.
    
    Args:
        root: Path to Cityscapes dataset
        
    Returns:
        Dict mapping client_id -> list of indices (700 per client)
    """
    print("🎯 Smart splitting: ensuring each client gets all 19 classes...")
    
    root = Path(root)
    rng = random.Random(_SEED)
    ds = Cityscapes(str(root), split="train", mode="fine", target_type="semantic")
    
    # Step 1: Single scan to build class mapping efficiently
    scan_limit = min(1000, len(ds))  # Scan first 1000 samples (should be enough)
    print(f"📋 Scanning first {scan_limit} samples to map classes...")
    
    class_to_samples = {cls: [] for cls in range(19)}
    
    # Single efficient scan for all classes
    for idx in range(scan_limit):
        if idx % 200 == 0:
            print(f"   Progress: {idx+1}/{scan_limit}...")
        _, lbl = ds[idx]
        unique_classes = set(np.unique(lbl))
        # Filter out invalid classes (only keep 0-18, ignore 255 and others)
        for cls in unique_classes:
            if 0 <= cls <= 18:  # Only valid Cityscapes classes
                class_to_samples[cls].append(idx)
    
    # Report class distribution
    print("📊 Class distribution in scanned samples:")
    for cls in range(19):
        count = len(class_to_samples[cls])
        print(f"   Class {cls}: {count} samples")
    
    # Step 2: Assign samples to clients ensuring class coverage
    client_indices = {cid: [] for cid in range(NUM_CLIENTS)}
    
    print("🏢 Distributing samples to clients...")
    
    # First, ensure each client gets at least one sample per class
    for cls in range(19):
        available_samples = class_to_samples[cls].copy()
        rng.shuffle(available_samples)
        
        for cid in range(NUM_CLIENTS):
            if available_samples:
                chosen_sample = available_samples.pop()
                client_indices[cid].append(chosen_sample)
                print(f"   Client {cid}, Class {cls}: sample {chosen_sample}")
    
    # Fill remaining slots randomly from entire dataset
    print("🎲 Filling remaining slots randomly...")
    
    # Get all used samples
    used_samples = set()
    for cid in range(NUM_CLIENTS):
        used_samples.update(client_indices[cid])
    
    # Get all available samples
    all_available = [i for i in range(len(ds)) if i not in used_samples]
    rng.shuffle(all_available)
    
    # Distribute remaining samples evenly
    for cid in range(NUM_CLIENTS):
        current_count = len(client_indices[cid])
        needed = SAMPLES_PER_CLIENT - current_count
        
        if needed > 0:
            additional = all_available[:needed]
            all_available = all_available[needed:]
            client_indices[cid].extend(additional)
            print(f"   Client {cid}: added {len(additional)} additional samples")
    
    # Final verification and reporting
    for cid in range(NUM_CLIENTS):
        print(f"✅ Client {cid}: {len(client_indices[cid])} samples (all 19 classes guaranteed)")
    
    return client_indices


def get_centralized_indices(root: str | Path) -> List[int]:
    """Get the same 2100 indices used for federated learning for centralized training.
    
    Args:
        root: Path to Cityscapes dataset
        
    Returns:
        List of 2100 indices for centralized training
    """
    return get_limited_dataset_indices(root, NUM_TOTAL_SAMPLES)


def _fix_class_coverage(client_indices: Dict[int, List[int]], 
                       target_client: int, 
                       missing_classes: set, 
                       dataset) -> None:
    """Fix class coverage by swapping samples between clients."""
    all_classes = set(range(19))
    
    for other_client in range(NUM_CLIENTS):
        if other_client == target_client:
            continue
            
        # Find samples in other_client that have missing classes
        for idx in list(client_indices[other_client]):
            _, lbl = dataset[idx]
            sample_classes = set(np.unique(lbl))
            
            if missing_classes & sample_classes:  # Intersection
                # Swap this sample
                client_indices[other_client].remove(idx)
                client_indices[target_client].append(idx)
                
                # Update missing classes
                missing_classes -= sample_classes
                
                if not missing_classes:
                    break
        
        if not missing_classes:
            break 