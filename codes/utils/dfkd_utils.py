"""
Data-Free Knowledge Distillation (DFKD) utilities for Federated Learning.

This module contains all the necessary utilities for implementing DFKD in the FL setting:
- Generator networks adapted for different datasets
- Image handling and memory management utilities
- FastMetaSynthesizer for synthetic data generation
- Loss functions and synthesis utilities
- Data structures for handling synthetic data efficiently

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
import random
import time
import copy
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from kornia import augmentation
from utils.model_utils import *

# ===========================
# Dataset Configuration - Using centralized transforms
# ===========================

from utils.data_utils import get_transform_manager


# ===========================
# Adaptive Beta Calculation for both simple KD and DFKD
# ===========================

def _stepwise_decay(config, staleness):
    """
    Step-wise decay function.
    Divide the staleness axis into four thresholds based on a base value.
    A good value for kd_settings['adaptive_beta_factor'] is 1.5.
    """
    kd_settings = config[config['kd_augmentation'] + '_settings']
    # Get adaptive beta parameters
    adaptive_factor = kd_settings['adaptive_beta_factor']
    nb_active_jobs = config[config['algorithm'] + '_settings']['nb_of_active_jobs']
    buffer_size = config[config['algorithm'] + '_settings']['buffer_size']
    
    # Calculate base threshold
    base_threshold = adaptive_factor * nb_active_jobs / buffer_size
    
    # Define thresholds
    threshold_1 = base_threshold / 4
    threshold_2 = base_threshold / 2
    threshold_3 = base_threshold * 3 / 4
    threshold_4 = base_threshold
    
    # Determine beta based on staleness
    if staleness < threshold_1:
        return 0.0  # Skip KD
    elif staleness < threshold_2:
        return 0.25
    elif staleness < threshold_3:
        return 0.5
    elif staleness < threshold_4:
        return 0.75
    else:
        return 1.0


def _exponential_decay(config, staleness):
    """
    Exponential decay function: β = 1 - exp(-λ * staleness)
    A good value for kd_settings['adaptive_beta_factor'] is 0.005.
    """
    kd_settings = config[config['kd_augmentation'] + '_settings']
    lambda_param = kd_settings['adaptive_beta_factor']
    
    beta = 1.0 - math.exp(-lambda_param * staleness)
    return min(1.0, max(0.0, beta))


def _linear_decay(config, staleness):
    """
    Linear decay function: β = min(1.0, γ * staleness)
    A good value for kd_settings['adaptive_beta_factor'] is 0.005.
    """
    kd_settings = config[config['kd_augmentation'] + '_settings']
    gamma = kd_settings['adaptive_beta_factor']
    
    beta = gamma * staleness
    return min(1.0, max(0.0, beta))



def _cosine_decay(config, staleness):
    """
    Cosine decay with clipping function:
    If staleness ≤ 2 * s_max: β = 1 - 0.5 * (1 + cos(π * staleness / (2 * s_max)))
    Else: β = 1.0

    A good value for kd_settings['adaptive_beta_factor'] is 75.
    """
    kd_settings = config[config['kd_augmentation'] + '_settings']
    s_max = kd_settings['adaptive_beta_factor']
    
    if staleness <= 2 * s_max:
        beta = 1.0 - 0.5 * (1.0 + math.cos(math.pi * staleness / (2.0 * s_max)))
    else:
        beta = 1.0
    
    return min(1.0, max(0.0, beta))


def calculate_adaptive_beta(config, staleness):
    """
    Calculate adaptive beta based on staleness and configuration.
    
    Args:
        staleness (int): The staleness of the update in terms of rounds
        
    Returns:
        float: The calculated beta value
    """
    if config['kd_augmentation'] is not None:
        kd_settings = config[config['kd_augmentation'] + '_settings']
    elif config['async_downweighting']:
        config = copy.deepcopy(config)
        config['kd_augmentation'] = 'dfkd'
        kd_settings = config['dfkd_settings'] # Use DFKD settings for async downweighting
    else:
        raise ValueError("No KD augmentation or async downweighting specified")
        
    if isinstance(kd_settings['beta'], (float, int)) or not isinstance(kd_settings['beta'], str):
        return kd_settings['beta']
    elif kd_settings['beta'] == "adaptive" or kd_settings['beta'] == "cosine":
        return _cosine_decay(config, staleness)
    elif kd_settings['beta'] == "exponential":
        return _exponential_decay(config, staleness)
    elif kd_settings['beta'] == "linear":
        return _linear_decay(config, staleness)
    elif kd_settings['beta'] == "stepwise":
        return _stepwise_decay(config, staleness)
    else:
        raise ValueError(f"Invalid beta type: {kd_settings['beta']}")


# ===========================
# Generator Network
# ===========================

class DFKDGenerator(nn.Module):
    """
    Data-Free Knowledge Distillation Generator adapted for different datasets.
    Uses additive class embeddings for class-conditional generation.
    """
    def __init__(self, dataset_name, nz=256, ngf=64):
        super(DFKDGenerator, self).__init__()
        
        # Get dataset specifications
        specs = get_transform_manager(dataset_name).get_specs()
        self.num_classes = specs['num_classes']
        self.nc = specs['channels']
        self.img_size = specs['img_size']

        self.params = (nz, ngf, self.img_size, self.nc, self.num_classes)
        self.init_size = self.img_size // 4
        self.nz = nz
        self.dataset_name = dataset_name  # Store dataset_name for cloning
        
        # First linear layer takes nz input (since we sum z + class_embeddings)
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        # Conv blocks with regular BatchNorm
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, self.nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, z, y, class_embeddings, class_embedding_weight=1.0):
        """
        Args:
            z: noise vector (batch, nz)
            y: class labels (batch,)
            class_embeddings: trainable class embeddings (num_classes, emb_dim)
            class_embedding_weight: weight for class embeddings in the sum
        """
        # Get class embeddings for each sample
        y_emb = class_embeddings[y]  # (batch, emb_dim)
        
        # Weighted additive conditioning: sum z and weighted class embeddings
        z_y = z + class_embedding_weight * y_emb  # (batch, nz) - requires emb_dim == nz
        
        # First linear layer
        out = self.l1(z_y)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        
        img = self.conv_blocks(out)
        
        return img

    def clone(self):
        """Create a copy of the generator"""
        # Extract constructor parameters from stored params
        nz, ngf, _, _, _ = self.params
        
        # Create new instance with same parameters
        clone = DFKDGenerator(
            dataset_name=self.dataset_name,
            nz=nz,
            ngf=ngf
        )
        
        # Copy the state from the original
        clone.load_state_dict(self.state_dict())
        
        # Move to same device as original
        clone = clone.to(next(self.parameters()).device)
        
        return clone

# ===========================
# Loss Functions
# ===========================

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    """KL divergence loss function"""
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T*T)

class KLDiv(nn.Module):
    """KL divergence loss module"""
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

# ===========================
# Image Handling Utilities
# ===========================


def save_image_grid(imgs, output, max_images=64, grid_size=None, labels=None):
    """Save images as a grid subplot for visualization with optional class labels"""
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().clamp(0, 1).cpu()
    
    # Handle labels
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
    
    # Limit number of images
    if len(imgs) > max_images:
        # Randomly sample indices instead of taking first ones
        indices = torch.randperm(len(imgs))[:max_images]
        imgs = imgs[indices]
        if labels is not None:
            labels = labels[indices]
    
    n_imgs = len(imgs)
    if n_imgs == 0:
        return
    
    # Calculate grid size
    if grid_size is None:
        cols = int(math.ceil(math.sqrt(n_imgs)))
        rows = int(math.ceil(n_imgs / cols))
    else:
        rows, cols = grid_size
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Convert axes to flat list for easy indexing
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot images
    for idx in range(rows * cols):
        ax = axes_flat[idx]
        
        if idx < n_imgs:
            img = imgs[idx]
            if img.dim() == 3 and img.shape[0] in [1, 3]:  # CHW format
                img = img.permute(1, 2, 0)
            if img.shape[-1] == 1:  # Grayscale
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            # Add class label as title if available
            if labels is not None and idx < len(labels):
                class_idx = labels[idx]
                ax.set_title(f'Class {class_idx}', fontsize=10, fontweight='bold', pad=2)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

# ===========================
# Synthetic Data Management - Simplified Implementation
# ===========================

class InMemorySyntheticDataset(Dataset):
    """
    Simplified in-memory dataset for synthetic images that maintains efficient label indexing.
    Combines dataset storage with efficient distribution-based sampling.
    """
    def __init__(self, images=None, labels=None, dataset_name=None):
        self.data = images if images is not None else []
        self.labels = labels if labels is not None else []
        
        # Efficient label-to-indices mapping for fast sampling
        self.label_to_indices = {}
        self._build_label_index()
        

        self.transform = None
        
    
    def _build_label_index(self):
        """Build the label-to-indices mapping from current labels."""
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label = int(label) if isinstance(label, torch.Tensor) else label
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    
    def add_samples(self, new_images, new_labels):
        """Add new images and labels efficiently - O(k) time where k = number of new samples."""
        if isinstance(new_images, torch.Tensor):
            new_images = new_images.detach().cpu()
            for i in range(new_images.shape[0]):
                self.data.append(new_images[i])
        else:
            self.data.extend(new_images)
        
        # Add labels and update index efficiently
        assert new_labels is not None, "Labels must be provided"
        start_idx = len(self.labels)
        
        if isinstance(new_labels, torch.Tensor):
            new_labels = new_labels.detach().cpu()
            for i in range(new_labels.shape[0]):
                label = int(new_labels[i])
                self.labels.append(label)
                # Update index in O(1) per new sample
                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(start_idx + i)
        else:
            for i, label in enumerate(new_labels):
                label = int(label) if isinstance(label, torch.Tensor) else label
                self.labels.append(label)
                # Update index in O(1) per new sample
                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(start_idx + i)
    
    def clear(self):
        """Clear all images and labels"""
        self.data = []
        self.labels = []
        self.label_to_indices = {}
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx] if idx < len(self.labels) else -1
        
        # Return raw synthetic data without any transforms
        # Transforms will be applied later in synthesis/KD phases
        if not isinstance(img, torch.Tensor):
            raise ValueError(f"Expected tensor image, got {type(img)}. Synthetic data should always be tensors.")
        
        # Convert label to tensor if it's not already
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
            
        return img, label
    
    def __len__(self):
        return len(self.data)
    
    def save_to_file(self, filepath):
        """Save dataset to torch file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'images': self.data,
            'labels': self.labels,
            'length': len(self.data)
        }, filepath)
    
    @classmethod
    def load_from_file(cls, filepath, dataset_name=None):
        """Load dataset from torch file with automatic transform setup"""
        assert os.path.exists(filepath), f"File {filepath} does not exist"
        
        data = torch.load(filepath, map_location='cpu')
        images = data.get('images', [])
        labels = data.get('labels', [])
        return cls(images=images, labels=labels, dataset_name=dataset_name)


def sample_batch_indices(dataset, label_distribution, batch_size):
    """
    Efficiently sample batch indices according to label distribution.
    
    Args:
        dataset: InMemorySyntheticDataset with label_to_indices mapping
        label_distribution: Dict[label, float] or tensor of probabilities
        batch_size: Number of samples to draw
    
    Returns:
        List[int]: Batch indices shuffled randomly
        
    Time Complexity: O(M + B) where M = distinct labels, B = batch_size
    """
    if len(dataset) == 0:
        return []
    
    # Handle both dict and tensor inputs
    if isinstance(label_distribution, dict):
        labels = list(label_distribution.keys())
        probs = torch.tensor([label_distribution[l] for l in labels], dtype=torch.float)
    else:
        # Assume tensor input with indices as labels
        labels = list(range(len(label_distribution)))
        probs = label_distribution.clone().detach() if isinstance(label_distribution, torch.Tensor) else torch.tensor(label_distribution, dtype=torch.float)
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Sample label indices using multinomial - O(M) time

    label_indices = torch.multinomial(probs, batch_size, replacement=True)
    
    # Count occurrences of each label - O(B) time
    from collections import Counter
    label_counts = Counter(label_indices.tolist())
    
    # Collect sample indices for each label - total O(B) time
    batch_indices = []
    for label_idx, count in label_counts.items():
        label = labels[label_idx]
        
        # Get available indices for this label
        available_indices = dataset.label_to_indices.get(label, [])
        
        if available_indices:
            # Sample 'count' indices from available ones
            if len(available_indices) >= count:
                # Random selection without replacement if we have enough
                selected = torch.randperm(len(available_indices))[:count]
                batch_indices.extend([available_indices[i] for i in selected])
            else:
                # Sample with replacement if we don't have enough
                selected = torch.randint(0, len(available_indices), (count,))
                batch_indices.extend([available_indices[i] for i in selected])
        else:
            # Fallback: random indices if label not found
            fallback_indices = torch.randint(0, len(dataset), (count,))
            batch_indices.extend(fallback_indices.tolist())
    
    # Shuffle the final batch - O(B) time
    if batch_indices:
        shuffled_order = torch.randperm(len(batch_indices))
        batch_indices = [batch_indices[i] for i in shuffled_order]
    
    return batch_indices[:batch_size]  # Ensure exact batch size


def sample_multiple_batches_optimized(dataset, distributions, batch_size, num_batches_per_dist):
    """
    Efficiently sample multiple batches for multiple distributions at once.
    
    This optimization avoids repeated multinomial sampling, counting, and shuffling
    by batching operations across all needed samples.
    
    Args:
        dataset: InMemorySyntheticDataset with label_to_indices mapping
        distributions: List of distribution tensors/arrays
        batch_size: Size of each batch
        num_batches_per_dist: Number of batches to sample per distribution
    
    Returns:
        List of (images, labels) tuples organized as:
        [dist0_batch0, dist0_batch1, ..., dist1_batch0, dist1_batch1, ...]
        
    Time Complexity: O(N*M + N*B) where N=total_batches, M=labels, B=batch_size
    vs. old approach: O(N*(M + B + shuffle)) - significant savings on multinomial calls
    """
    if len(dataset) == 0:
        return []
    
    # Prepare all distributions and labels
    all_probs = []
    labels_list = []
    
    for dist in distributions:
        if isinstance(dist, dict):
            labels = list(dist.keys())
            probs = torch.tensor([dist[l] for l in labels], dtype=torch.float)
        else:
            labels = list(range(len(dist)))
            probs = dist.clone().detach() if isinstance(dist, torch.Tensor) else torch.tensor(dist, dtype=torch.float)
        
        # Normalize probabilities
        probs = probs / probs.sum()
        all_probs.append(probs)
        labels_list.append(labels)
    
    # Pre-sample all label indices at once - major optimization!
    total_samples_needed = len(distributions) * num_batches_per_dist * batch_size
    all_batch_results = []
    
    # Process each distribution
    for dist_idx, (probs, labels) in enumerate(zip(all_probs, labels_list)):
        # Sample all batches for this distribution at once
        total_samples_for_dist = num_batches_per_dist * batch_size
        
        all_label_indices = torch.multinomial(probs, total_samples_for_dist, replacement=True)
        
        # Count all labels at once
        from collections import Counter
        all_label_counts = Counter(all_label_indices.tolist())
        
        # Pre-generate all sample indices for this distribution
        all_sample_indices = []
        for label_idx, count in all_label_counts.items():
            label = labels[label_idx]
            available_indices = dataset.label_to_indices.get(label, [])
            
            if available_indices:
                if len(available_indices) >= count:
                    # Sample without replacement
                    selected = torch.randperm(len(available_indices))[:count]
                    selected_indices = [available_indices[i] for i in selected]
                else:
                    # Sample with replacement
                    selected = torch.randint(0, len(available_indices), (count,))
                    selected_indices = [available_indices[i] for i in selected]
            else:
                # Fallback to random indices
                selected_indices = torch.randint(0, len(dataset), (count,)).tolist()
            
            all_sample_indices.extend(selected_indices)
        
        # Shuffle all indices once
        if all_sample_indices:
            shuffled_order = torch.randperm(len(all_sample_indices))
            all_sample_indices = [all_sample_indices[i] for i in shuffled_order]
        
        # Split into individual batches
        for batch_idx in range(num_batches_per_dist):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = all_sample_indices[start_idx:end_idx]
            
            # Load actual data for this batch
            if batch_indices:
                images = []
                labels = []
                for idx in batch_indices:
                    img, label = dataset[idx]
                    images.append(img)
                    labels.append(label)
                
                if images:
                    batch_data = (torch.stack(images), torch.tensor(labels, dtype=torch.long))
                    all_batch_results.append(batch_data)
                else:
                    all_batch_results.append((None, None))
            else:
                all_batch_results.append((None, None))
    
    return all_batch_results


class SimplifiedMemoryImagePool(object):
    """
    Simplified memory pool using the efficient dataset and sampling approach.
    """
    
    def __init__(self, dataset_name, exp_dir, max_images=None, vis_freq=10):
        self.dataset_name = dataset_name
        self.exp_dir = os.path.abspath(exp_dir)
        self.max_images = max_images
        self.vis_freq = vis_freq
        self.iteration = 0
        
        # Create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Single efficient dataset with built-in label indexing
        self.dataset = InMemorySyntheticDataset(dataset_name=dataset_name)
        
        # Dataset file path
        self.dataset_path = os.path.join(self.exp_dir, 'synthetic_dataset.pth')
    
    def add(self, imgs, current_round, targets=None):
        """Add images and labels to the pool"""
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().cpu()
        
        # Add to dataset efficiently - O(k) time
        self.dataset.add_samples(imgs, targets)
        
        # Apply max_images limit if specified
        if self.max_images is not None and len(self.dataset) > self.max_images:
            # Keep only the most recent images and labels
            excess = len(self.dataset) - self.max_images
            self.dataset.data = self.dataset.data[excess:]
            self.dataset.labels = self.dataset.labels[excess:]
            # Rebuild index after truncation (still O(max_images))
            self.dataset._build_label_index()
        
        # Save visualization periodically
        if self.iteration % self.vis_freq == 0:
            vis_path = os.path.join(
                self.exp_dir, 'dfkd_visualizations', f'round_{current_round}.png'
            )
            # Move to CPU for visualization to free GPU memory
            cpu_imgs = imgs.detach().cpu() if isinstance(imgs, torch.Tensor) else imgs
            cpu_targets = targets.detach().cpu() if isinstance(targets, torch.Tensor) else targets
            save_image_grid(cpu_imgs, vis_path, max_images=64, labels=cpu_targets)
        
            # Save dataset to file (overwrite)
            self.dataset.save_to_file(self.dataset_path)
        
        self.iteration += 1
    
    def sample_batch(self, label_distribution, batch_size):
        """
        Sample a batch according to the given label distribution.
        
        Args:
            label_distribution: Dict[label, float] or tensor of probabilities
            batch_size: Number of samples to draw
            
        Returns:
            Tuple of (images, labels) tensors
        """
        # Get batch indices efficiently
        indices = sample_batch_indices(self.dataset, label_distribution, batch_size)
        
        if not indices:
            return None, None
        
        # Load actual data
        images = []
        labels = []
        for idx in indices:
            img, label = self.dataset[idx]
            images.append(img)
            labels.append(label)
        
        if images:
            return torch.stack(images), torch.tensor(labels, dtype=torch.long)
        else:
            return None, None
    
    def get_dataset(self):
        """Get a copy of the dataset for backward compatibility"""
        return self.dataset

    
    def load_initial_dataset(self, dataset_path):
        """Load initial dataset from file"""
        assert os.path.exists(dataset_path), f"File {dataset_path} does not exist"
        initial_dataset = InMemorySyntheticDataset.load_from_file(
            dataset_path, 
            dataset_name=self.dataset_name
        )
        self.dataset.add_samples(initial_dataset.data, initial_dataset.labels)
        print(f"Loaded {len(initial_dataset)} initial images with labels from {dataset_path}")



class DataIter(object):
    """Data iterator utility"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data

# ===========================
# Synthesis Utilities
# ===========================

class SynthesisMomentumManager:
    """
    External manager for synthesis momentum state.
    Keeps all synthesis-related state separate from the model.
    """
    def __init__(self):
        self.momentum_state = {}  # Maps StatTracker id to (mean_mmt, var_mmt)
    
    def get_momentum(self, stat_tracker):
        """Get momentum state for a StatTracker module"""
        tracker_id = id(stat_tracker)
        return self.momentum_state.get(tracker_id, None)
    
    def set_momentum(self, stat_tracker, mean_mmt, var_mmt):
        """Set momentum state for a StatTracker module"""
        tracker_id = id(stat_tracker)
        self.momentum_state[tracker_id] = (mean_mmt.detach().clone(), var_mmt.detach().clone())
    
    def update_momentum(self, stat_tracker, mean, var, mmt_rate=0.9):
        """Update momentum state for a StatTracker module"""
        tracker_id = id(stat_tracker)
        
        if tracker_id not in self.momentum_state:
            # Initialize momentum (detached to avoid gradient conflicts)
            self.momentum_state[tracker_id] = (mean.detach().clone(), var.detach().clone())
        else:
            mean_mmt, var_mmt = self.momentum_state[tracker_id]
            # Detach inputs to avoid gradient graph conflicts between iterations
            new_mean_mmt = mmt_rate * mean_mmt + (1 - mmt_rate) * mean.detach()
            new_var_mmt = mmt_rate * var_mmt + (1 - mmt_rate) * var.detach()
            self.momentum_state[tracker_id] = (new_mean_mmt, new_var_mmt)
    
    def clear(self):
        """Clear all momentum state"""
        self.momentum_state.clear()

def get_stat_trackers(model):
    """
    Get all StatTracker modules from any model.
    
    Args:
        model: Any model that may contain StatTracker modules
    
    Returns:
        List of StatTracker modules
    """
    stat_trackers = []
    for module in model.modules():
        if isinstance(module, StatTracker):
            stat_trackers.append(module)
    return stat_trackers

def compute_feature_loss(model, batch_features, momentum_manager=None):
    """
    Compute feature loss for synthesis by comparing running stats to current batch stats.
    Uses pre-computed features to avoid redundant forward passes.
    
    Args:
        model: Any model with StatTracker modules
        batch_features: Pre-computed features from model(inputs, return_features=True)
        momentum_manager: Optional SynthesisMomentumManager for momentum-mixed targets
    
    Returns:
        Total feature loss (scalar tensor)
    """
    stat_trackers = get_stat_trackers(model)
    
    # Initialize with proper tensor type and device
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Ensure we have the expected number of features
    if len(stat_trackers) != len(batch_features):
        raise ValueError(f"Mismatch: {len(stat_trackers)} StatTrackers but {len(batch_features)} batch features")
    
    for i, stat_tracker in enumerate(stat_trackers):
        mean, var = batch_features[i]
        
        # Compute target stats (running vs momentum-mixed)
        if momentum_manager is None:
            target_mean = stat_tracker.running_mean
            target_var = stat_tracker.running_var
        else:
            momentum_state = momentum_manager.get_momentum(stat_tracker)
            if momentum_state is None:
                target_mean = stat_tracker.running_mean
                target_var = stat_tracker.running_var
            else:
                mean_mmt, var_mmt = momentum_state
                target_mean = mean_mmt
                target_var = var_mmt
        
        # L2 norm distances
        loss = (target_var - var).norm(2) + (target_mean - mean).norm(2)
        total_loss = total_loss + loss
    
    return total_loss

def update_stat_tracker_momentum(model, batch_features, momentum_manager, mmt_rate=0.9):
    """
    Update momentum for StatTracker modules using pre-computed batch statistics.
    This should only be called during synthesis.
    Uses pre-computed features to avoid redundant forward passes.
    
    Args:
        model: Any model with StatTracker modules
        batch_features: Pre-computed features from model(inputs, return_features=True)
        momentum_manager: SynthesisMomentumManager to store momentum state
        mmt_rate: Momentum mixing rate for synthesis
    """
    stat_trackers = get_stat_trackers(model)
    
    for i, stat_tracker in enumerate(stat_trackers):
        if i < len(batch_features):
            mean, var = batch_features[i]
            momentum_manager.update_momentum(stat_tracker, mean, var, mmt_rate)

# ===========================
# Meta-Learning Utilities
# ===========================

def reptile_grad(src, tar):
    """Apply REPTILE gradient update"""
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data)#, alpha=67) 

def fomaml_grad(src, tar):
    """Apply First-Order MAML gradient update"""
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)

def reset_l0(model):
    """Reset the first layer of the generator"""
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

# ===========================
# FastMetaSynthesizer
# ===========================

class FastMetaSynthesizer:
    """
    Fast Meta-Learning based synthetic data generation for federated learning.
    Adapted for FL context with teacher model distribution handling.
    """
    def __init__(self, generator, config):
        """
        Initialize the synthesizer.
        
        Args:
            generator: Generator network 
            config: Configuration object with DFKD settings
        """
        # Get dataset specifications and DFKD settings
        dataset_name = config['dataset']
        self.dataset_specs = get_transform_manager(dataset_name).get_specs()
        dfkd_config = config['dfkd_settings']

        # Core components
        self.generator = generator
        
        # Dataset and model properties
        self.num_classes = self.dataset_specs['num_classes']
        self.img_size = (self.dataset_specs['channels'], 
                        self.dataset_specs['img_size'], 
                        self.dataset_specs['img_size'])
        
        # Synthesis parameters
        self.nz = dfkd_config['nz']
        self.iterations = dfkd_config['g_steps']
        self.lr_g = dfkd_config['lr_g']
        self.lr_z = dfkd_config['lr_z']
        self.synthesis_batch_size = dfkd_config['synth_batch_size']
        self.sample_batch_size = dfkd_config['kd_batch_size']
        
        # Loss weights
        self.adv = dfkd_config['adv']
        self.bn = dfkd_config['bn']
        self.oh = dfkd_config['oh']
        self.kl_uniform = dfkd_config['kl_uniform']
        self.diversity_loss_weight = dfkd_config['diversity_loss_weight']
        self.class_embedding_weight = dfkd_config['class_embedding_weight']
        
        # Training parameters
        self.bn_mmt = dfkd_config['bn_mmt']
        self.ismaml = dfkd_config['is_maml']
        self.reset_l0_flag = dfkd_config['reset_l0']
        self.ep_start = dfkd_config['warmup_rounds']
        
        # Experiment settings
        self.exp_dir = config['out_dir']
        
        # Setup data management - simplified approach
        self.data_pool = SimplifiedMemoryImagePool(
            dataset_name=dataset_name,
            exp_dir=self.exp_dir, 
            max_images=dfkd_config['max_images'], 
            vis_freq=dfkd_config['vis_freq']
        )
        
        # Create augmentation pipeline using centralized kornia transforms
        self.aug = get_transform_manager(dataset_name).get_transforms(mode='train', format='kornia', normalize=True)
        
        # Training state
        self.ep = 0
        self.prev_z = None
        
        # External momentum manager for synthesis (completely separate from model)
        self.momentum_manager = SynthesisMomentumManager()
        
        # Loss tracking windows for all losses
        self.loss_window_size = 20
        self.bn_loss_window = []
        self.oh_loss_window = []
        self.adv_loss_window = []
        self.kl_uniform_loss_window = []
        self.diversity_loss_window = []
        self.kd_loss_window = []
        
        # Setup device
        self.device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        self.generator = self.generator.to(self.device).train()
        
        self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                                 self.lr_g, betas=[0.5, 0.999], eps=torch.finfo(torch.float32).tiny)  # eps=0 gave 0/0=NaN for never-updated elements

    def _compute_teacher_weights(self, teacher_list):
        """
        Compute normalized weights w_k(j) for each teacher k and label j.
        
        Args:
            teacher_list: List of (teacher, distribution) tuples
            
        Returns:
            weights: Tensor of shape [num_teachers, num_classes] containing w_k(j)
            q_values: Tensor of shape [num_teachers, num_classes] containing q_k(j)
        """
        num_teachers = len(teacher_list)
        
        # Extract teacher distributions (q_k values) - ensure they're already on device
        distributions = []
        for _, distribution in teacher_list:
            if not isinstance(distribution, torch.Tensor):
                distribution = torch.tensor(distribution, dtype=torch.float32, device=self.device)
            elif distribution.device != self.device:
                distribution = distribution.to(self.device)
            distributions.append(distribution)
        
        q_values = torch.stack(distributions)  # [num_teachers, num_classes]
        
        # Compute normalized weights: w_k(j) = q_k(j) / sum_t q_t(j)
        sum_q = q_values.sum(dim=0, keepdim=True)  # [1, num_classes]
        # Avoid division by zero
        sum_q = torch.clamp(sum_q, min=1e-8)
        weights = q_values / sum_q  # [num_teachers, num_classes]
        
        return weights, q_values
    
    def _stabilize_loss(self, loss, loss_window, loss_weight):
        """
        Stabilize any loss by tracking a window of recent values and zeroing out
        losses that are inf, nan, or outliers to prevent extreme loss spikes.
        
        Args:
            loss: Current loss value
            loss_window: List tracking recent loss values
            loss_weight: Loss weight coefficient (if <= 0, return loss as is)
            
        Returns:
            Stabilized loss value
        """
        if loss_weight <= 0:  # If loss is not used, return as is
            return loss
            
        loss_value = loss.item()
        
        # Check for inf, nan, or extreme values
        if torch.isnan(loss) or torch.isinf(loss):
            # Return a zero tensor that properly preserves gradients by using the original loss
            return loss * 0.0  # Multiplying by 0 preserves the computation graph
        
        # If we have enough history, check for outliers
        if len(loss_window) >= 10:  # Need at least 10 values for stable mean
            window_mean = sum(loss_window) / len(loss_window)
            threshold = 3 * window_mean
            
            # If current loss is too large, zero it out
            if loss_value > threshold and window_mean > 0:
                # Return a zero tensor that properly preserves gradients
                return loss * 0.0  # Multiplying by 0 preserves the computation graph
        
        # Add current loss to window (only if it's a reasonable value)
        loss_window.append(loss_value)
        
        # Keep window size limited
        if len(loss_window) > self.loss_window_size:
            loss_window.pop(0)  # Remove oldest value
            
        return loss
    
    def synthesize(self, teacher_list, student, current_round, targets=None):
        """
        Synthesize data using the given teacher list.
        
        Args:
            teacher_list: List of (teacher, distribution) tuples
            student: Student model to train (passed as argument)
            targets: Optional target labels
            
        Returns:
            Dict containing synthesized results and synthesis info
        """
        start = time.time()
        
        self.ep += 1
        student.eval()
        
        num_teachers = len(teacher_list)
        
        best_cost = 1e6
        
        # Track synthesis losses
        synthesis_losses = []
        loss_components = {'bn': [], 'oh': [], 'adv': [], 'kl_uniform': [], 'diversity': [], 'total': []}
        
        best_inputs = None

        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        
        if targets is None:
            # Sample targets from combined distribution of all teachers
            # Compute combined distribution as mean of all teacher distributions
            distributions_on_device = [distribution.to(self.device) for _, distribution in teacher_list]
            combined_distribution = torch.stack(distributions_on_device).mean(dim=0)
            targets = torch.multinomial(combined_distribution, self.synthesis_batch_size, replacement=True)
        else:
            targets = targets.to(self.device)
        
        # Create trainable class embeddings for each class (similar to z)
        # Use the same dimension as nz for addition to work
        emb_dim = self.nz  # Must match nz for z + class_embeddings to work
        class_embeddings = torch.randn(size=(self.num_classes, emb_dim), device=self.device).requires_grad_()
        
        # Compute teacher weights for this batch
        teacher_weights, q_values = self._compute_teacher_weights(teacher_list)

        fast_generator = self.generator.clone().to(self.device)  # Ensure cloned generator is on GPU

        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z},
            {'params': [class_embeddings], 'lr': self.lr_z}  # Optimize class embeddings with same lr as z
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            # Generator forward pass
            inputs = fast_generator(z, targets, class_embeddings, self.class_embedding_weight)
            inputs_aug = self.aug(inputs)  # crop and normalize
            if it == 0:
                originalMeta = inputs
            
            #############################################
            # Weighted Multi-Teacher Loss Computation (Batch-wise)
            #############################################
            
            # Get all teacher outputs and features
            teacher_outputs = []
            teacher_features = []
            
            # Process all teachers in parallel where possible
            for teacher_idx, (teacher, _) in enumerate(teacher_list):
                # CORRECT FIX: Teacher parameters are frozen (requires_grad=False) so no gradients
                # are computed for them, but gradient flow through operations is preserved
                # This allows gradients to flow back to inputs_aug and generator
                t_out, t_feat = teacher(inputs_aug, return_features=True)

                teacher_outputs.append(t_out)
                teacher_features.append(t_feat)
            
            # Stack teacher outputs: [num_teachers, batch_size, num_classes]
            teacher_outputs = torch.stack(teacher_outputs, dim=0)
            
            # Get weights for all samples: [batch_size, num_teachers]  
            sample_weights = teacher_weights[:, targets].T  # [batch_size, num_teachers]
            sample_q_values = q_values[:, targets].T  # [batch_size, num_teachers]
            
            # Feature loss: sum_m w_m(y) * feature_loss(m(z), m)
            loss_bn = torch.zeros(1, device=self.device, requires_grad=True)
            if self.bn > 0:
                total_loss_bn = torch.tensor(0.0, device=self.device, requires_grad=True)
                for teacher_idx, (teacher, _) in enumerate(teacher_list):
                    if teacher_features[teacher_idx]:  # Only if features are available
                        feature_loss = compute_feature_loss(teacher, teacher_features[teacher_idx], self.momentum_manager)
                        # Weight by sum of all sample weights for this teacher
                        teacher_weight_sum = sample_weights[:, teacher_idx].sum()
                        total_loss_bn = total_loss_bn + teacher_weight_sum * feature_loss
                loss_bn = total_loss_bn / self.synthesis_batch_size
                
                # Stabilize BN loss to prevent extreme values
                loss_bn = self._stabilize_loss(loss_bn, self.bn_loss_window, self.bn)
            
            # Cross entropy loss: sum_m w_m(y) * cross_entropy_loss(m(z), y)
            # Compute CE loss for each teacher: [num_teachers, batch_size]
            ce_losses = torch.zeros(len(teacher_list), self.synthesis_batch_size, device=self.device)
            for teacher_idx in range(len(teacher_list)):
                ce_losses[teacher_idx] = F.cross_entropy(
                    teacher_outputs[teacher_idx], targets, reduction='none'
                )
            
            # Apply weights: [batch_size, num_teachers] * [num_teachers, batch_size] -> [batch_size, num_teachers]
            weighted_ce_losses = sample_weights * ce_losses.T
            loss_oh = weighted_ce_losses.sum() / self.synthesis_batch_size
            
            # Stabilize OH loss
            loss_oh = self._stabilize_loss(loss_oh, self.oh_loss_window, self.oh)
            
            # KL-from-Uniform loss: sum_m q_m(y) * KL(m(z), uniform)
            loss_kl_uniform = torch.tensor(0.0, device=self.device, requires_grad=True)
            if self.kl_uniform > 0:
                uniform_target = torch.ones(self.num_classes, device=self.device) / self.num_classes
                uniform_target = uniform_target.unsqueeze(0).expand(self.synthesis_batch_size, -1)  # [batch_size, num_classes]
                
                # Compute KL divergence for each teacher: [num_teachers, batch_size]
                kl_losses = torch.zeros(len(teacher_list), self.synthesis_batch_size, device=self.device)
                for teacher_idx in range(len(teacher_list)):
                    teacher_probs = F.softmax(teacher_outputs[teacher_idx], dim=1)
                    kl_losses[teacher_idx] = F.kl_div(
                        teacher_probs.log(), uniform_target, reduction='none'
                    ).sum(dim=1)
                
                # Apply q_values weights: [batch_size, num_teachers] * [num_teachers, batch_size] -> [batch_size, num_teachers]
                weighted_kl_losses = ((1-sample_q_values)**5) * kl_losses.T
                loss_kl_uniform = weighted_kl_losses.sum() / self.synthesis_batch_size
                
                # Stabilize KL uniform loss
                loss_kl_uniform = self._stabilize_loss(loss_kl_uniform, self.kl_uniform_loss_window, self.kl_uniform)
            
            # Diversity Loss: weighted pairwise L2 distance between teacher features of the same class
            loss_diversity = torch.tensor(0.0, device=self.device, requires_grad=True)
            if self.diversity_loss_weight > 0:
                # Get unique classes in the current batch
                unique_classes = targets.unique()
                total_weighted_div = 0.0
                
                for class_j in unique_classes:
                    # Find all indices where targets == class_j
                    class_indices = (targets == class_j).nonzero(as_tuple=True)[0]
                    
                    if len(class_indices) > 1:  # Skip if only one sample of this class
                        # Compute average teacher_weights for this class across all teachers
                        q_bar_j = teacher_weights[:, class_j].mean()
                        
                        # Compute pairwise distances for each teacher separately
                        div_sum_j = 0.0
                        num_pairs_total = 0
                        
                        # For each teacher, compute pairwise distances between samples of class j
                        for teacher_idx in range(len(teacher_list)):
                            # teacher_features[teacher_idx] is a list of (mean, var) tuples from all layers
                            layer_features = teacher_features[teacher_idx]  # [(mean1, var1), (mean2, var2), ...]
                            
                            # Select middle features (2 mean, 2 var pairs = 4 variables total)
                            num_layers = len(layer_features)
                            middle_features = []
                            
                            if num_layers >= 2:
                                # Find middle indices for 2 (mean, var) pairs
                                if num_layers == 2:
                                    indices = [0, 1]  # Use both if only 2 layers
                                else:
                                    # Select 2 middle layers
                                    mid = num_layers // 2
                                    if num_layers % 2 == 0:
                                        indices = [mid - 1, mid]  # Even number of layers
                                    else:
                                        indices = [mid - 1, mid + 1] if mid > 0 else [mid, mid + 1]  # Odd number
                                
                                # Collect the 4 variables (2 means + 2 vars)
                                for idx in indices:
                                    if idx < num_layers:
                                        mean, var = layer_features[idx]
                                        middle_features.append(mean)  # Shape: [C]
                                        middle_features.append(var)   # Shape: [C]
                            else:
                                # Fallback: use available features
                                for mean, var in layer_features:
                                    middle_features.append(mean)
                                    middle_features.append(var)
                            
                            # Concatenate into a single feature vector
                            teacher_feat = torch.cat(middle_features, dim=0)  # Shape: [total_middle_features]
                            
                            class_features = teacher_feat[class_indices]  # [num_class_samples]
                            
                            # Vectorized computation of pairwise distances within this teacher's features
                            n_samples = class_features.shape[0]
                            if n_samples > 1:
                                # Ensure class_features is 2D
                                if class_features.dim() == 1:
                                    class_features = class_features.unsqueeze(1)  # Make it [n_samples, 1]
                                
                                # Expand to [n_samples, 1, D] and [1, n_samples, D] for broadcasting
                                feat_i = class_features.unsqueeze(1)  # [n_samples, 1, D]
                                feat_j = class_features.unsqueeze(0)  # [1, n_samples, D]
                                
                                # Compute squared L2 distances: [n_samples, n_samples]
                                diff = feat_i - feat_j
                                squared_dists = (diff ** 2).sum(dim=-1)  # Use -1 to sum over last dimension
                                
                                # Extract upper triangular part (i < j) to avoid double counting
                                upper_tri_mask = torch.triu(torch.ones(n_samples, n_samples, device=self.device), diagonal=1).bool()
                                pairwise_dists = squared_dists[upper_tri_mask]
                                
                                # Sum the distances for this teacher
                                div_sum_j += pairwise_dists.sum()
                                num_pairs_total += len(pairwise_dists)
                        
                        # Average over all pairs from all teachers
                        if num_pairs_total > 0:
                            div_sum_j = div_sum_j / num_pairs_total
                        
                        # Weight by average teacher_weights for this class
                        weighted_div_j = q_bar_j * div_sum_j
                        total_weighted_div += weighted_div_j
                
                loss_diversity = total_weighted_div
                
                # Stabilize diversity loss
                loss_diversity = self._stabilize_loss(loss_diversity, self.diversity_loss_window, self.diversity_loss_weight)
            
            # Adversarial loss: sum_m w_m(y) * adv_loss(m(z), student(z))
            loss_adv = torch.tensor(0.0, device=self.device, requires_grad=True)
            if self.adv > 0 and (self.ep >= self.ep_start):
                student_output = student(inputs_aug)  # [batch_size, num_classes]
                
                # Compute adversarial loss for each teacher: [num_teachers, batch_size]
                adv_losses = torch.zeros(len(teacher_list), self.synthesis_batch_size, device=self.device)
                for teacher_idx in range(len(teacher_list)):
                    teacher_output = teacher_outputs[teacher_idx]  # [batch_size, num_classes]
                    
                    # Check if predictions match
                    s_pred = student_output.max(1)[1]
                    t_pred = teacher_output.max(1)[1]
                    mask = (s_pred == t_pred).float()
                    
                    # Compute KL divergence
                    kl_loss = kldiv(student_output, teacher_output, reduction='none').sum(dim=1)
                    adv_losses[teacher_idx] = -(kl_loss * mask)
                
                # Apply weights: [batch_size, num_teachers] * [num_teachers, batch_size] -> [batch_size, num_teachers]
                weighted_adv_losses = sample_weights * adv_losses.T
                loss_adv = weighted_adv_losses.sum() / self.synthesis_batch_size
                
                # Stabilize adversarial loss
                loss_adv = self._stabilize_loss(loss_adv, self.adv_loss_window, self.adv)
            
            # Combine losses and optimization step
            loss = (self.bn * loss_bn + 
                   self.oh * loss_oh + 
                   self.adv * loss_adv + 
                   self.kl_uniform * loss_kl_uniform + 
                   self.diversity_loss_weight * loss_diversity)

            # Track best inputs
            with torch.no_grad():
                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_inputs = inputs.data.clone()
            
            # Track losses
            synthesis_losses.append(loss.item())
            loss_components['bn'].append(loss_bn.item())
            loss_components['oh'].append(loss_oh.item()) 
            loss_components['adv'].append(loss_adv.item())
            loss_components['kl_uniform'].append(loss_kl_uniform.item())
            loss_components['diversity'].append(loss_diversity.item())
            loss_components['total'].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            
            if self.ismaml:
                if it == 0: 
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1): 
                    self.meta_optimizer.step()
                else:
                    optimizer.step()  # Still need to step the fast optimizer for MAML
            else:
                optimizer.step()
        
        # Update external momentum manager (no model state modified)
        if self.bn_mmt != 0 and best_inputs is not None:
            # Update momentum for all teachers using best inputs
            # Teacher parameters are already frozen, so this is efficient
            for teacher, _ in teacher_list:
                with torch.no_grad():  # No gradients needed for momentum update
                    _, best_features = teacher(self.aug(best_inputs), return_features=True)
                if best_features:  # Only if features are available
                    update_stat_tracker_momentum(teacher, best_features, self.momentum_manager, mmt_rate=self.bn_mmt)

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            if all(torch.isfinite(p.grad).all() for p in self.generator.parameters()):  # skip if NaN/inf grads hit the fast generator
                self.meta_optimizer.step()


        student.train()
        self.prev_z = {"z": z, "class_embeddings": class_embeddings, "targets": targets}

        # Use best inputs if available, otherwise use current inputs
        if best_inputs is None:
            best_inputs = inputs.data

        # Add to simplified pool with labels (handles visualization and saving automatically)
        if torch.isfinite(best_inputs).all():  # never pool NaN/inf images (they would poison every later KD step)
            self.data_pool.add(best_inputs, current_round, targets)
        
        end = time.time()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        synthesis_info = {
            'losses': synthesis_losses,
            'loss_components': loss_components,
            'best_cost': best_cost,
            'synthesis_time': end - start
        }
        
        return {"synthetic": best_inputs}, synthesis_info
    
    def sample(self, distribution=None):
        """
        Sample data from the synthesized dataset.
        
        Args:
            distribution: If provided, sample according to this distribution.
                         If None, use uniform sampling.
        
        Returns:
            Tuple of (images, labels)
        """
        # Sample from specific distribution using simplified approach
        if distribution is None:
            distribution = torch.ones(self.num_classes) / self.num_classes
        return self.data_pool.sample_batch(distribution, self.sample_batch_size)

## ===========================
# DFKD Operations
## ===========================

def perform_dfkd_with_buffer(current_round, kd_buffer, synthesizer, config, global_model_params, clients, teacher_models=None, student_model=None, wandb_run=None, synthesize_this_round=True):
    """
    Perform Data-Free Knowledge Distillation using buffered teacher models.
    
    Required imports (to be added where this function is called):
    - from utils.model_utils import create_model, reset_model_state, load_model_params, get_model_params, get_model_diff
    
    Args:
        current_round: Current federated learning round
        kd_buffer: Buffer containing (model_params, client_id) tuples
        synthesizer: FastMetaSynthesizer instance
        config: Configuration dictionary
        global_model_params: Current global model parameters
        clients: List of client objects
        teacher_models: Optional list of pre-created teacher models
        student_model: Optional pre-created student model
        wandb_run: Optional wandb run object for logging metrics
        synthesize_this_round: Whether to perform generator training and synthesis this round.
                               If False, only KD training with existing synthetic data is performed.
                               Default is True to preserve backward compatibility.
        
    Returns:
        Updated student model parameters
    """

    assert kd_buffer is not None and len(kd_buffer) > 0, f"KD buffer must have at least 1 model but got {len(kd_buffer)} models. Check the config or the code!"
    # Extract teacher model parameters from buffer
    teacher_models_params = [model_params for model_params, _ in kd_buffer]
    # Get KD settings from config
    kd_settings = config['dfkd_settings']
    batch_size = kd_settings['kd_batch_size']  # kd batch size
    lr = kd_settings['lr']  # Learning rate for student
    T = kd_settings['T']  # Temperature for knowledge distillation

    warmup_rounds = kd_settings['warmup_rounds']  # Warmup rounds before adversarial loss
    kd_nb_of_iters = kd_settings['kd_nb_of_iters']  # Number of total steps in each KD run

    device = config['train_device']
    
    client_map = {client.client_id: client for client in clients}        
    teacher_client_class_probs = []
    # Extract class probabilities for all teacher models (same order as teacher_models_params)
    # All clients are in the kd_buffer in the correct order
    for _, client_id in kd_buffer:
        client = client_map[client_id]
        client_probs = client.class_proportions
        teacher_client_class_probs.append(client_probs)

    teacher_models_to_use = []
    for i, teacher_params in enumerate(teacher_models_params):
        assert teacher_models is not None, "Teacher models must be provided for DFKD"
        if teacher_models is None:
            teacher_model = create_model(config=config)
        else:
            assert len(teacher_models) == len(teacher_models_params), f"Number of teacher models ({len(teacher_models)}) must match number of teacher model parameters ({len(teacher_models_params)}). Check the config or the code!"
            teacher_model = teacher_models[i]
            reset_model_state(teacher_model, reset_norm_stats=teacher_model.has_running_stats)  # Comprehensive reset
        teacher_model = load_model_params(teacher_model, teacher_params, device)
        teacher_model.eval()
        # Turn off gradients for teacher models
        if next(teacher_model.parameters()).requires_grad:
            for p in teacher_model.parameters():
                p.requires_grad = False
        teacher_models_to_use.append(teacher_model)

    assert student_model is not None, "Student model must be provided for DFKD"
    # Use provided student model or create new one
    if student_model is None:
        student_model = create_model(config=config)
    else:
        reset_model_state(student_model, reset_norm_stats=student_model.has_running_stats)
    student_model = load_model_params(student_model, global_model_params, device)
    # For synthesis, eval mode and turn off gradients
    student_model.eval()
    if next(student_model.parameters()).requires_grad:
        for p in student_model.parameters():
            p.requires_grad = False

    criterion = KLDiv(T=T, reduction='sum')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-5)

    teacher_models_and_distributions = [(teacher_model, torch.tensor(distribution, dtype=torch.float32)) for teacher_model, distribution in zip(teacher_models_to_use, teacher_client_class_probs)]
    
    # Conditionally perform synthesis based on synthesize_this_round
    if synthesize_this_round:
        vis_results, synthesis_info = synthesizer.synthesize(teacher_models_and_distributions, student_model, current_round, targets=None)
        
        # Log synthesis metrics (best cost and final components)
        syn_components = synthesis_info['loss_components']
        best_cost = synthesis_info['best_cost']
        syn_time = synthesis_info['synthesis_time']
        
        # Log best synthesis losses in one line
        logging.info(f"Round {current_round}: DFKD synthesis - Best Cost: {best_cost:.4f}, BN: {syn_components['bn'][-1]:.4f}, OH: {syn_components['oh'][-1]:.4f}, ADV: {syn_components['adv'][-1]:.4f}, KL: {syn_components['kl_uniform'][-1]:.4f}, DIV: {syn_components['diversity'][-1]:.4f}, Time: {syn_time:.2f}s")
        
        # Log to wandb if enabled and wandb_run provided
        if wandb_run is not None and config.get("wandb_flag", False):
            wandb_metrics = {
                'dfkd_synthesis/best_cost': best_cost,
                'dfkd_synthesis/bn_loss': syn_components['bn'][-1],
                'dfkd_synthesis/oh_loss': syn_components['oh'][-1],
                'dfkd_synthesis/adv_loss': syn_components['adv'][-1],
                'dfkd_synthesis/kl_uniform_loss': syn_components['kl_uniform'][-1],
                'dfkd_synthesis/diversity_loss': syn_components['diversity'][-1],
                'dfkd_synthesis/time': syn_time,
                'round': current_round
            }
            wandb_run.log(wandb_metrics)
    else:
        # Skip synthesis, use existing synthetic data for KD-only
        generate_freq = config['dfkd_settings'].get('generate_freq', 1)
        logging.info(f"Round {current_round}: Skipping DFKD synthesis (generate_freq={generate_freq}), performing KD-only with existing synthetic data")

    if current_round > warmup_rounds:

        # For KD training, train mode and turn on gradients if not already done
        student_model.train()
        if not next(student_model.parameters()).requires_grad:
            for p in student_model.parameters():
                p.requires_grad = True
        
        # Knowledge Distillation Training Phase
        total_loss = 0
        total_samples = 0
        step_losses = []
        
        start_time = time.time()
        
        # Pre-sample all batches at once to avoid repeated sampling overhead
        # Convert teacher distributions to the format expected by the optimized sampler
        teacher_distributions = [torch.tensor(probs, dtype=torch.float32) for probs in teacher_client_class_probs]
        
        # # Use classical optimized method instead
        all_batches = sample_multiple_batches_optimized(
            dataset=synthesizer.data_pool.dataset,
            distributions=teacher_distributions,
            batch_size=batch_size,
            num_batches_per_dist=kd_nb_of_iters
        )
        

        # Organize batches: [teacher0_batch0, teacher0_batch1, ..., teacher1_batch0, ...]
        # We need to reorganize to: [iteration0_teacher0, iteration0_teacher1, ..., iteration1_teacher0, ...]
        organized_batches = []
        num_teachers = len(teacher_models_to_use)
  
        for kd_iter in range(kd_nb_of_iters):
            iter_batches = []
            for teacher_idx in range(num_teachers):
                batch_idx = teacher_idx * kd_nb_of_iters + kd_iter
                if batch_idx < len(all_batches):
                    iter_batches.append(all_batches[batch_idx])
                else:
                    iter_batches.append((None, None))
            organized_batches.append(iter_batches)
        
        for kd_iter in range(kd_nb_of_iters):
            
            # Initialize loss for this iteration
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            iteration_batch_size = 0
            
            # Get pre-sampled batches for this iteration
            iter_batches = organized_batches[kd_iter]
            
            # For each teacher and its pre-sampled batch
            for teacher_idx, (teacher_model, (images, labels)) in enumerate(zip(teacher_models_to_use, iter_batches)):
                
                if images is None or images.size(0) == 0:
                    continue  # Skip if no data available
                
                # Move to device
                if images.device != device:
                    images = images.to(device, non_blocking=True)
                if labels.device != device:
                    labels = labels.to(device, non_blocking=True)
                
                # Apply augmentation transforms (same as synthesizer uses)
                with torch.no_grad():
                    images = synthesizer.aug(images)
                
                # Student forward pass
                s_out = student_model(images.detach())
                
                # Teacher forward pass
                with torch.no_grad():
                    t_out = teacher_model(images)
                
                # Loss computation
                teacher_loss = criterion(s_out, t_out.detach())
                
                # Add to accumulated loss
                loss = loss + teacher_loss
                iteration_batch_size += images.size(0)
            
            if iteration_batch_size > 0:
                loss = loss / iteration_batch_size

                # Stabilize KD loss
                loss = synthesizer._stabilize_loss(loss, synthesizer.kd_loss_window, 1)
            
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss = total_loss + loss.item()
                total_samples += iteration_batch_size
                step_losses.append(loss.item())
        
        # Calculate training metrics and log KD results
        if len(step_losses) > 0:
            avg_loss = total_loss / len(step_losses)
            min_loss = min(step_losses)
            max_loss = max(step_losses)
            training_time = time.time() - start_time
            
            # Log KD training final losses in one line
            logging.info(f"Round {current_round}: DFKD KD training - Avg Loss: {avg_loss:.4f}, Min Loss: {min_loss:.4f}, Max Loss: {max_loss:.4f}, Steps: {len(step_losses)}, Time: {training_time:.2f}s")
            
            # Log to wandb if enabled and wandb_run provided
            if wandb_run is not None and config.get("wandb_flag", False):
                kd_metrics = {
                    'dfkd_kd/avg_loss': avg_loss,
                    'dfkd_kd/min_loss': min_loss,
                    'dfkd_kd/max_loss': max_loss,
                    'dfkd_kd/num_steps': len(step_losses),
                    'dfkd_kd/time': training_time,
                    'round': current_round
                }
                wandb_run.log(kd_metrics)
        else:
            logging.warning(f"Round {current_round}: No successful KD training steps completed")
    
    else:
        logging.info(f"Round {current_round}: Skipping KD training (warmup phase, round <= {warmup_rounds})")
    
    # Get the distilled model parameters
    distilled_model_params = get_model_params(student_model, config["model_device"])
    
    # If server_lr is negative, return the distilled model parameters
    # Else, return the difference between the distilled and current model
    if config['server_lr'] < 0:
        return distilled_model_params
    
    # Calculate the KD update (difference between distilled and global model)
    kd_update = get_model_diff(distilled_model_params, global_model_params)
    
    return kd_update
