"""
Checkpoint utilities for saving and resuming training
with improved logging
"""
import pickle
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np


def save_checkpoint(data, filename, metadata=None, verbose=True):
    """
    Save checkpoint data with metadata and detailed logging
    
    Args:
        data: Dictionary with checkpoint data
        filename: Path to save checkpoint
        metadata: Additional metadata to save
        verbose: Whether to print detailed logs
    """
    checkpoint_dir = Path(filename).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'data': data,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
        file_size = f.tell()
    
    # Get readable size
    size_str = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/1024/1024:.1f}MB"
    
    # Get keys from data for logging
    data_keys = list(data.keys()) if isinstance(data, dict) else ['data']
    
    print(f"[CHECKPOINT] SAVED: {Path(filename).name}", flush=True)
    print(f"[CHECKPOINT]   Location: {filename}", flush=True)
    print(f"[CHECKPOINT]   Size: {size_str}", flush=True)
    print(f"[CHECKPOINT]   Data keys: {data_keys}", flush=True)
    if metadata:
        print(f"[CHECKPOINT]   Metadata: {metadata}", flush=True)
    print(f"[CHECKPOINT]   Timestamp: {checkpoint['timestamp']}", flush=True)
    print(f"[CHECKPOINT]   Checkpoint saved successfully", flush=True)
    
    return filename


def load_checkpoint(filename, verbose=True):
    """
    Load checkpoint data with detailed logging
    
    Returns:
        Tuple of (data, metadata) or (None, None) if not found
    """
    if not Path(filename).exists():
        if verbose:
            print(f"[CHECKPOINT] No checkpoint found: {filename}", flush=True)
        return None, None
    
    try:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        data = checkpoint.get('data')
        metadata = checkpoint.get('metadata')
        timestamp = checkpoint.get('timestamp', 'unknown')
        version = checkpoint.get('version', 'unknown')
        
        if verbose:
            print(f"[CHECKPOINT] LOADED: {Path(filename).name}", flush=True)
            print(f"[CHECKPOINT]   Location: {filename}", flush=True)
            print(f"[CHECKPOINT]   Version: {version}", flush=True)
            print(f"[CHECKPOINT]   Created: {timestamp}", flush=True)
            if metadata:
                print(f"[CHECKPOINT]   Metadata: {metadata}", flush=True)
            if isinstance(data, dict):
                print(f"[CHECKPOINT]   Data keys: {list(data.keys())}", flush=True)
            print(f"[CHECKPOINT]   Checkpoint loaded successfully", flush=True)
        
        return data, metadata
        
    except Exception as e:
        print(f"[CHECKPOINT] Failed to load {filename}: {e}", flush=True)
        return None, None


def get_latest_checkpoint(checkpoint_dir, prefix="", verbose=False):
    """
    Get the latest checkpoint file in a directory with logging
    
    Args:
        checkpoint_dir: Directory to search
        prefix: Optional prefix to filter files
        verbose: Whether to print detailed logs
    
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        if verbose:
            print(f"[CHECKPOINT] Directory not found: {checkpoint_dir}", flush=True)
        return None
    
    pattern = f"{prefix}*.pkl" if prefix else "*.pkl"
    files = list(checkpoint_dir.glob(pattern))
    
    if not files:
        if verbose:
            print(f"[CHECKPOINT] No checkpoints found in {checkpoint_dir} (pattern: {pattern})", flush=True)
        return None
    
    # Sort by modification time
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest = files[0]
    
    if verbose:
        print(f"[CHECKPOINT] Latest checkpoint: {latest.name}", flush=True)
        print(f"[CHECKPOINT]   Modified: {datetime.fromtimestamp(latest.stat().st_mtime).isoformat()}", flush=True)
        print(f"[CHECKPOINT]   Size: {latest.stat().st_size/1024:.1f}KB", flush=True)
    
    return latest


def checkpoint_exists(checkpoint_dir, prefix=""):
    """Check if any checkpoint exists in directory"""
    return get_latest_checkpoint(checkpoint_dir, prefix) is not None


def get_checkpoint_summary(checkpoint_dir):
    """Get a summary of all checkpoints in a directory"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return {'total': 0, 'files': [], 'total_size': 0}
    
    files = list(checkpoint_dir.glob("*.pkl"))
    if not files:
        return {'total': 0, 'files': [], 'total_size': 0}
    
    total_size = sum(f.stat().st_size for f in files)
    
    return {
        'total': len(files),
        'files': [{'name': f.name, 'size': f.stat().st_size, 'modified': f.stat().st_mtime} for f in files],
        'total_size': total_size,
        'total_size_str': f"{total_size/1024:.1f}KB" if total_size < 1024*1024 else f"{total_size/1024/1024:.1f}MB"
    }
