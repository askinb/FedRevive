"""
Data-Free Knowledge Distillation (DFKD) utilities for Text Classification Tasks.

This module contains utilities for implementing DFKD in FL setting for text tasks:
- Text-specific data structures for synthetic embeddings
- FastTextMetaSynthesizer for synthetic embedding generation using prompt vectors
- Loss functions and synthesis utilities for transformer models
- Text-specific momentum management and feature extraction

MAIN CLASSES:
=============
Core Data Structures:
- SyntheticTextDataset: Handles synthetic embeddings with attention masks
- MemoryEfficientTextPool: Memory-efficient storage for synthetic text data
- Text sampling utilities for distribution-aware sampling

Synthesis Components:
- FastTextMetaSynthesizer: Main synthesizer with multi-teacher support for text
- TextSynthesisMomentumManager: External momentum management for synthesis
- StatTrackerText: Statistics tracker for transformer models

USAGE:
======
Similar to dfkd_utils.py but adapted for text data with embeddings instead of images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os
import numpy as np
import random
import time
import logging
import math
import copy
from collections import defaultdict, Counter
from utils.model_utils import *
from utils.synthetic_text_data_utils import SyntheticNewsDataset 
from utils.synthetic_text_data_utils import main as synthetic_text_data_utils_main
from utils.synthetic_text_data_utils import get_dataloader as get_synthetic_dataloader



# ===========================
# Loss Functions
# ===========================

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    """KL divergence loss function for text models."""
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T*T)

class KLDiv(nn.Module):
    """KL divergence loss module."""
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

# ===========================
# Text Data Handling Utilities
# ===========================

def save_text_samples(embeddings, attention_mask, labels, output_dir, max_samples=10):
    """Save text sample statistics for visualization/debugging."""
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, 'embedding_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Embeddings shape: {embeddings.shape}\n")
        f.write(f"Embeddings mean: {embeddings.mean():.4f}\n")
        f.write(f"Embeddings std: {embeddings.std():.4f}\n")
        f.write(f"Embeddings min: {embeddings.min():.4f}\n")
        f.write(f"Embeddings max: {embeddings.max():.4f}\n")
        f.write(f"Attention mask shape: {attention_mask.shape}\n")
        f.write(f"Active lengths: {attention_mask.sum(dim=1).tolist()[:max_samples]}\n")
        f.write(f"Labels: {labels.tolist()[:max_samples]}\n")

# ===========================
# Synthetic Text Data Management
# ===========================

class SyntheticTextDataset(Dataset):
    """
    In-memory dataset for synthetic text embeddings and attention masks.
    Similar to InMemorySyntheticDataset but for text data.
    """
    
    def __init__(self, embeddings=None, attention_masks=None, labels=None, kd_max_length=256):
        self.embeddings = embeddings if embeddings is not None else []
        self.attention_masks = attention_masks if attention_masks is not None else []
        self.labels = labels if labels is not None else []
        self.kd_max_length = kd_max_length
        
        # Efficient label-to-indices mapping for fast sampling
        self.label_to_indices = {}
        self._build_label_index()
    
    def _build_label_index(self):
        """Build the label-to-indices mapping from current labels."""
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label = int(label) if isinstance(label, torch.Tensor) else label
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def add_samples(self, new_embeddings, new_attention_masks, new_labels):
        """Add new embeddings, attention masks, and labels efficiently."""
        if isinstance(new_embeddings, torch.Tensor):
            new_embeddings = new_embeddings.detach().cpu()
            for i in range(new_embeddings.shape[0]):
                self.embeddings.append(new_embeddings[i])
        else:
            self.embeddings.extend(new_embeddings)
        
        if isinstance(new_attention_masks, torch.Tensor):
            new_attention_masks = new_attention_masks.detach().cpu()
            for i in range(new_attention_masks.shape[0]):
                self.attention_masks.append(new_attention_masks[i])
        else:
            self.attention_masks.extend(new_attention_masks)
        
        # Add labels and update index efficiently
        assert new_labels is not None, "Labels must be provided"
        start_idx = len(self.labels)
        
        if isinstance(new_labels, torch.Tensor):
            new_labels = new_labels.detach().cpu()
            for i in range(new_labels.shape[0]):
                label = int(new_labels[i])
                self.labels.append(label)
                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(start_idx + i)
        else:
            for i, label in enumerate(new_labels):
                label = int(label) if isinstance(label, torch.Tensor) else label
                self.labels.append(label)
                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(start_idx + i)
    
    def clear(self):
        """Clear all embeddings, attention masks, and labels."""
        self.embeddings = []
        self.attention_masks = []
        self.labels = []
        self.label_to_indices = {}
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        attention_mask = self.attention_masks[idx] if idx < len(self.attention_masks) else torch.ones(self.kd_max_length, dtype=torch.bool)
        label = self.labels[idx] if idx < len(self.labels) else -1
        
        # Ensure embedding is tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Truncate/pad embedding to kd_max_length
        if embedding.size(0) > self.kd_max_length:
            embedding = embedding[:self.kd_max_length, :]
        elif embedding.size(0) < self.kd_max_length:
            padding_size = self.kd_max_length - embedding.size(0)
            embedding = F.pad(embedding, (0, 0, 0, padding_size), value=0.0)
        
        # Ensure attention_mask is tensor
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        
        # Truncate/pad attention mask to kd_max_length
        if attention_mask.size(0) > self.kd_max_length:
            attention_mask = attention_mask[:self.kd_max_length]
        elif attention_mask.size(0) < self.kd_max_length:
            padding_size = self.kd_max_length - attention_mask.size(0)
            attention_mask = F.pad(attention_mask, (0, padding_size), value=False)
        
        # Convert label to tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
            
        return embedding, attention_mask, label
    
    def __len__(self):
        return len(self.embeddings)
    
    def save_to_file(self, filepath):
        """Save dataset to torch file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'embeddings': self.embeddings,
            'attention_masks': self.attention_masks,
            'labels': self.labels,
            'length': len(self.embeddings)
        }, filepath)
    
    @classmethod
    def load_from_file(cls, filepath, kd_max_length=256):
        """Load dataset from torch file."""
        assert os.path.exists(filepath), f"File {filepath} does not exist"
        
        data = torch.load(filepath, map_location='cpu')
        embeddings = data.get('embeddings', [])
        attention_masks = data.get('attention_masks', [])
        labels = data.get('labels', [])
        return cls(embeddings=embeddings, attention_masks=attention_masks, labels=labels, kd_max_length=kd_max_length)


def sample_text_batch_indices(dataset, label_distribution, batch_size):
    """
    Efficiently sample batch indices according to label distribution for text data.
    Similar to sample_batch_indices but for text datasets.
    """
    if len(dataset) == 0:
        return []
    
    # Handle both dict and tensor inputs
    if isinstance(label_distribution, dict):
        labels = list(label_distribution.keys())
        probs = torch.tensor([label_distribution[l] for l in labels], dtype=torch.float)
    else:
        labels = list(range(len(label_distribution)))
        probs = label_distribution.clone().detach() if isinstance(label_distribution, torch.Tensor) else torch.tensor(label_distribution, dtype=torch.float)
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Sample label indices using multinomial
    label_indices = torch.multinomial(probs, batch_size, replacement=True)
    
    # Count occurrences of each label
    label_counts = Counter(label_indices.tolist())
    
    # Collect sample indices for each label
    batch_indices = []
    for label_idx, count in label_counts.items():
        label = labels[label_idx]
        
        # Get available indices for this label
        available_indices = dataset.label_to_indices.get(label, [])
        
        if available_indices:
            if len(available_indices) >= count:
                selected = torch.randperm(len(available_indices))[:count]
                batch_indices.extend([available_indices[i] for i in selected])
            else:
                selected = torch.randint(0, len(available_indices), (count,))
                batch_indices.extend([available_indices[i] for i in selected])
        else:
            # Fallback: random indices if label not found
            fallback_indices = torch.randint(0, len(dataset), (count,))
            batch_indices.extend(fallback_indices.tolist())
    
    # Shuffle the final batch
    if batch_indices:
        shuffled_order = torch.randperm(len(batch_indices))
        batch_indices = [batch_indices[i] for i in shuffled_order]
    
    return batch_indices[:batch_size]


def sample_multiple_text_batches_optimized(dataset, distributions, batch_size, num_batches_per_dist):
    """
    Efficiently sample multiple batches for multiple distributions at once for text data.
    Similar to sample_multiple_batches_optimized but for text datasets.
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
    
    # Pre-sample all label indices at once
    all_batch_results = []
    
    # Process each distribution
    for dist_idx, (probs, labels) in enumerate(zip(all_probs, labels_list)):
        # Sample all batches for this distribution at once
        total_samples_for_dist = num_batches_per_dist * batch_size
        
        all_label_indices = torch.multinomial(probs, total_samples_for_dist, replacement=True)
        
        # Count all labels at once
        all_label_counts = Counter(all_label_indices.tolist())
        
        # Pre-generate all sample indices for this distribution
        all_sample_indices = []
        for label_idx, count in all_label_counts.items():
            label = labels[label_idx]
            available_indices = dataset.label_to_indices.get(label, [])
            
            if available_indices:
                if len(available_indices) >= count:
                    selected = torch.randperm(len(available_indices))[:count]
                    selected_indices = [available_indices[i] for i in selected]
                else:
                    selected = torch.randint(0, len(available_indices), (count,))
                    selected_indices = [available_indices[i] for i in selected]
            else:
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
                embeddings = []
                attention_masks = []
                labels_batch = []
                for idx in batch_indices:
                    emb, mask, label = dataset[idx]
                    embeddings.append(emb)
                    attention_masks.append(mask)
                    labels_batch.append(label)
                
                if embeddings:
                    batch_data = (
                        torch.stack(embeddings), 
                        torch.stack(attention_masks),
                        torch.tensor(labels_batch, dtype=torch.long)
                    )
                    all_batch_results.append(batch_data)
                else:
                    all_batch_results.append((None, None, None))
            else:
                all_batch_results.append((None, None, None))
    
    return all_batch_results


class MemoryEfficientTextPool(object):
    """
    Memory-efficient synthetic text data pool.
    Similar to SimplifiedMemoryImagePool but for text data.
    """
    
    def __init__(self, exp_dir, max_samples=None, vis_freq=10, kd_max_length=256):
        self.exp_dir = os.path.abspath(exp_dir)
        self.max_samples = max_samples
        self.vis_freq = vis_freq
        self.kd_max_length = kd_max_length
        self.iteration = 0
        
        # Create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Single efficient dataset with built-in label indexing
        self.dataset = SyntheticTextDataset(kd_max_length=kd_max_length)
        
        # Dataset file path
        self.dataset_path = os.path.join(self.exp_dir, 'synthetic_text_dataset.pth')
    
    def add(self, embeddings, attention_masks, current_round, targets=None):
        """Add embeddings, attention masks, and labels to the pool."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        if isinstance(attention_masks, torch.Tensor):
            attention_masks = attention_masks.detach().cpu()
        
        # Add to dataset efficiently
        self.dataset.add_samples(embeddings, attention_masks, targets)
        
        # Apply max_samples limit if specified
        if self.max_samples is not None and len(self.dataset) > self.max_samples:
            excess = len(self.dataset) - self.max_samples
            self.dataset.embeddings = self.dataset.embeddings[excess:]
            self.dataset.attention_masks = self.dataset.attention_masks[excess:]
            self.dataset.labels = self.dataset.labels[excess:]
            # Rebuild index after truncation
            self.dataset._build_label_index()
        
        # Save visualization periodically
        if self.iteration % self.vis_freq == 0:
            vis_dir = os.path.join(
                self.exp_dir, 'dfkd_text_visualizations', f'round_{current_round}'
            )
            cpu_embeddings = embeddings.detach().cpu() if isinstance(embeddings, torch.Tensor) else embeddings
            cpu_masks = attention_masks.detach().cpu() if isinstance(attention_masks, torch.Tensor) else attention_masks
            cpu_targets = targets.detach().cpu() if isinstance(targets, torch.Tensor) else targets
            save_text_samples(cpu_embeddings, cpu_masks, cpu_targets, vis_dir, max_samples=16)
            
            # Save dataset to file (overwrite)
            self.dataset.save_to_file(self.dataset_path)
        
        self.iteration += 1
    
    def sample_batch(self, label_distribution, batch_size):
        """
        Sample a batch according to the given label distribution.
        Returns (embeddings, attention_masks, labels) tensors.
        """
        indices = sample_text_batch_indices(self.dataset, label_distribution, batch_size)
        
        if not indices:
            return None, None, None
        
        # Load actual data
        embeddings = []
        attention_masks = []
        labels = []
        for idx in indices:
            emb, mask, label = self.dataset[idx]
            embeddings.append(emb)
            attention_masks.append(mask)
            labels.append(label)
        
        if embeddings:
            return (
                torch.stack(embeddings), 
                torch.stack(attention_masks),
                torch.tensor(labels, dtype=torch.long)
            )
        else:
            return None, None, None
    
    def get_dataset(self):
        """Get the dataset for backward compatibility."""
        return self.dataset
    
    def load_initial_dataset(self, dataset_path):
        """Load initial dataset from file."""
        assert os.path.exists(dataset_path), f"File {dataset_path} does not exist"
        initial_dataset = SyntheticTextDataset.load_from_file(dataset_path, kd_max_length=self.kd_max_length)
        self.dataset.add_samples(
            initial_dataset.embeddings, 
            initial_dataset.attention_masks,
            initial_dataset.labels
        )
        print(f"Loaded {len(initial_dataset)} initial text samples from {dataset_path}")


# ===========================
# Text Synthesis Utilities
# ===========================

class TextSynthesisMomentumManager:
    """
    External manager for synthesis momentum state for text models.
    Keeps all synthesis-related state separate from the model.
    """
    def __init__(self):
        self.momentum_state = {}  # Maps StatTrackerText id to momentum statistics
    
    def get_momentum(self, stat_tracker):
        """Get momentum state for a StatTrackerText module."""
        tracker_id = id(stat_tracker)
        return self.momentum_state.get(tracker_id, None)
    
    def set_momentum(self, stat_tracker, mean_mmt, var_mmt):
        """Set momentum state for a StatTrackerText module."""
        tracker_id = id(stat_tracker)
        self.momentum_state[tracker_id] = (mean_mmt.detach().clone(), var_mmt.detach().clone())
    
    def update_momentum(self, stat_tracker, mean, var, mmt_rate=0.9):
        assert False, "This function shouldn't be used"
        """Update momentum state for a StatTrackerText module."""
        tracker_id = id(stat_tracker)
        
        if tracker_id not in self.momentum_state:
            # Initialize momentum (detached to avoid gradient conflicts)
            self.momentum_state[tracker_id] = (mean.detach().clone(), var.detach().clone())
        else:
            mean_mmt, var_mmt = self.momentum_state[tracker_id]
            # Detach inputs to avoid gradient graph conflicts
            new_mean_mmt = mmt_rate * mean_mmt + (1 - mmt_rate) * mean.detach()
            new_var_mmt = mmt_rate * var_mmt + (1 - mmt_rate) * var.detach()
            self.momentum_state[tracker_id] = (new_mean_mmt, new_var_mmt)
    
    def clear(self):
        """Clear all momentum state."""
        self.momentum_state.clear()


def compute_text_feature_loss(model, feature_layers):
    """
    Compute feature loss between current batch features and running statistics using tensor operations.
    
    Args:
        model: T5 model with StatTrackerText modules
        feature_layers: List of (mean, var) tuples, where mean and var are tensors of shape [batch_size, hidden_dim]
    
    Returns:
        Total feature loss averaged across layers and batch
    """
    # Get number of stat tracker layers (excluding layer 0)
    num_trackers = len([name for name in model.stat_trackers.keys() 
                       if name.startswith('stat_track_layer_')])
    
    # Verify dimensions
    if len(feature_layers) != num_trackers:
        raise ValueError(f"Expected {num_trackers} feature layers but got {len(feature_layers)}")
    
    total_loss = torch.tensor(0.0, device=feature_layers[0][0].device)
    
    # Process each layer
    for layer_idx_, (current_mean, current_var) in enumerate(feature_layers):
        if layer_idx_ in [1, 4]:
            layer_idx = layer_idx_ + 1
            tracker = model.stat_trackers[f'stat_track_layer_{layer_idx}']
            running_mean, running_var = tracker.get_running_stats()
            
            # Current stats are [batch_size, hidden_dim]
            # Running stats are [hidden_dim]
            # Compute L2 distances efficiently using broadcasting
            mean_loss = (current_mean - running_mean).norm(2, dim=1)  # [batch_size]
            
            # Sum losses for this layer and average over batch
            # layer_loss = (mean_loss + var_loss).mean()
            layer_loss = mean_loss.mean()
            total_loss += layer_loss
        
    # Average over all layers (excluding layer 0)
    total_loss = total_loss / num_trackers
    
    return total_loss



# ===========================
# Meta-Learning Utilities (Same as image-based)
# ===========================

def reptile_grad(src, tar):
    """Apply REPTILE gradient update."""
    # Support for both nn.Module and torch.Tensor
    if isinstance(src, nn.Module) and isinstance(tar, nn.Module):
        src_params = list(src.parameters())
        tar_params = list(tar.parameters())
    elif torch.is_tensor(src) and torch.is_tensor(tar):
        src_params = [src]
        tar_params = [tar]
    else:
        raise TypeError("reptile_grad expects both src and tar to be either nn.Module or torch.Tensor")
    
    for p, tar_p in zip(src_params, tar_params):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.data.add_(p.data - tar_p.data)


def fomaml_grad(src, tar):
    """Apply First-Order MAML gradient update."""
    # Support for both nn.Module and torch.Tensor
    if isinstance(src, nn.Module) and isinstance(tar, nn.Module):
        src_params = list(src.parameters())
        tar_params = list(tar.parameters())
    elif torch.is_tensor(src) and torch.is_tensor(tar):
        src_params = [src]
        tar_params = [tar]
    else:
        raise TypeError("fomaml_grad expects both src and tar to be either nn.Module or torch.Tensor")
    
    for p, tar_p in zip(src_params, tar_params):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.data.add_(tar_p.grad.data)


# ===========================
# Attention Mask Generation
# ===========================

def generate_attention_masks(batch_size, prompt_len, max_len, max_text_len=128, min_text_len=64, device='cuda'):
    """
    Generate random attention masks for text synthesis.
    
    Args:
        batch_size: Number of samples in batch
        prompt_len: Length of prompt vectors
        max_len: Maximum sequence length
        max_text_len: Maximum active text length
        min_text_len: Minimum active text length
        device: Device to create masks on
    
    Returns:
        Attention masks [batch_size, max_len]
    """
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for b in range(batch_size):
        active_len = torch.randint(min_text_len, max_text_len + 1, (1,)).item()
        attention_mask[b, 0:active_len+prompt_len] = True
    return attention_mask


# ===========================
# FastTextMetaSynthesizer
# ===========================

class FastTextMetaSynthesizer:
    """
    Fast Meta-Learning based synthetic embedding generation for text classification.
    Adapted for FL context with teacher model distribution handling.
    Uses trainable prompt vectors instead of a generator network.
    """
    def __init__(self, config):
        """
        Initialize the text synthesizer.
        
        Args:
            config: Configuration object with DFKD settings
            tokenizer: Tokenizer for text processing (optional)
        """

        tokenizer=None
        # Get dataset specifications and DFKD settings
        dataset_name = config['dataset']
        assert dataset_name.lower() == 'news', f"FastTextMetaSynthesizer only supports 'news' dataset, got {dataset_name}"
        
        dfkd_config = config['dfkd_settings']
        dfkd_config.update(config['news_dataset_settings'])
        dfkd_config['max_samples'] = dfkd_config['max_images']
        
        # Dataset and model properties
        from utils.data_utils import get_transform_manager
        transform_manager = get_transform_manager(dataset_name)
        self.num_classes = transform_manager.get_specs()['num_classes']
        
        # Text-specific parameters
        self.kd_max_length = dfkd_config['kd_max_length']
        self.embedding_dim = dfkd_config['embedding_dim']  # T5-small hidden size
        self.prompt_vec_len = dfkd_config['kd_prompt_vec_len']
        # Generator parameters for text
        self.generator_min_active = dfkd_config['kd_generator_min_active']
        self.generator_max_active = dfkd_config['kd_generator_max_active']
        
        # Synthesis parameters
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
        self.class_embedding_weight = dfkd_config['class_embedding_weight'] # not used

        # Training parameters
        self.bn_mmt = dfkd_config['bn_mmt']
        self.ismaml = dfkd_config['is_maml']
        self.ep_start = dfkd_config['warmup_rounds']
        self.reset_l0 = dfkd_config['reset_l0']

        # Experiment settings
        self.exp_dir = config['out_dir']
        
        # Setup data management
        self.data_pool = MemoryEfficientTextPool(
            exp_dir=self.exp_dir,
            max_samples=dfkd_config['max_samples'],
            vis_freq=dfkd_config['vis_freq'],
            kd_max_length=self.kd_max_length
        )
        
        # Tokenizer (optional, for compatibility)
        self.tokenizer = tokenizer
        
        # Training state
        self.ep = 0
        
        # External momentum manager for synthesis
        self.momentum_manager = TextSynthesisMomentumManager()
        
        # Loss tracking windows
        self.loss_window_size = 20
        self.bn_loss_window = []
        self.oh_loss_window = []
        self.adv_loss_window = []
        self.kl_uniform_loss_window = []
        self.diversity_loss_window = []
        self.kd_loss_window = []
        
        # Setup device
        self.device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        
        # Trainable prompt vectors (meta-learned across rounds)
        self.prompt_vecs = torch.randn(
            self.prompt_vec_len,
            self.embedding_dim,
            device=self.device,
            requires_grad=True
        )
        
        # Meta optimizer for prompt vectors
        self.meta_optimizer = torch.optim.Adam(
            [self.prompt_vecs],
            self.lr_g * self.iterations,
            betas=[0.5, 0.999]
        )

        self.synthetic_dataset = self._get_synthetic_dataset()
        self.synthetic_dataloaders = [get_synthetic_dataloader(None, self.synthetic_dataset, 64, i) for i in range(self.num_classes)]

    
    def _get_synthetic_dataset(self, path = "../data/synthetic_news/synthetic_news.jsonl"):
        """
        Get the synthetic dataset.
        """
        if not os.path.exists(path):
            logging.info(f"Synthetic dataset not found at {path}, running synthetic_text_data_utils.py with default configuration")
            synthetic_text_data_utils_main()

        return SyntheticNewsDataset(
            path=path,
            tokenize_on_load=True)

    def _compute_teacher_weights(self, teacher_list):
        """
        Compute normalized weights w_k(j) for each teacher k and label j.
        Same as image-based DFKD.
        """
        num_teachers = len(teacher_list)
        
        # Extract teacher distributions
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
        sum_q = torch.clamp(sum_q, min=1e-8)
        weights = q_values / sum_q  # [num_teachers, num_classes]
        
        return weights, q_values
    
    def _stabilize_loss(self, loss, loss_window, loss_weight):
        """
        Stabilize loss by tracking window and zeroing out outliers.
        Same as image-based DFKD.
        """
        if loss_weight <= 0:
            return loss
            
        loss_value = loss.item()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return loss * 0.0
        
        if len(loss_window) >= 10:
            window_mean = sum(loss_window) / len(loss_window)
            threshold = 3 * window_mean
            
            if loss_value > threshold and window_mean > 0:
                return loss * 0.0
        
        loss_window.append(loss_value)
        
        if len(loss_window) > self.loss_window_size:
            loss_window.pop(0)
            
        return loss

    def _get_trainable_text_embeddings(self, sorted_targets):
        targets = sorted_targets
        unique_targets, counts = torch.unique_consecutive(targets, return_counts=True)

        all_input_ids, all_attention_masks, all_labels = [], [], []

        for cls, num in zip(unique_targets.tolist(), counts.tolist()):
            loader = self.synthetic_dataloaders[cls]
            it = iter(loader)
            n = 0
            input_ids_list, attention_list, labels_list = [], [], []
            while n < num:
                batch = next(it)
                if isinstance(batch, dict):
                    ids = batch['input_ids']
                    mask = batch['attention_mask']
                    label = batch.get('labels', torch.full((ids.size(0),), cls))

                take = min(len(ids), num - n)
                input_ids_list.append(ids[:take])
                attention_list.append(mask[:take])
                labels_list.append(label[:take])
                n += take
            all_input_ids.append(torch.cat(input_ids_list, dim=0))
            all_attention_masks.append(torch.cat(attention_list, dim=0))
            all_labels.append(torch.cat(labels_list, dim=0))

        batch_input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        batch_attention_mask = torch.cat(all_attention_masks, dim=0).to(self.device)
        batch_labels = torch.cat(all_labels, dim=0).to(self.device)

        with torch.no_grad():
            text_embeddings = self.embedder(batch_input_ids)
        return text_embeddings
    
    def synthesize(self, teacher_list, student, current_round, targets=None):
        """
        Synthesize text embeddings using the given teacher list.
        
        Args:
            teacher_list: List of (teacher, distribution) tuples
            student: Student model to train
            current_round: Current FL round
            targets: Optional target labels
            
        Returns:
            Dict containing synthesized results and synthesis info
        """
        start = time.time()
        
        self.ep += 1
        student.eval()

        if not hasattr(self, 'embedder'):
            self.embedder = copy.deepcopy(student.encoder.t5_encoder.shared)
            self.embedder = self.embedder.to(self.device)
        
        # Set all teachers to eval mode
        for teacher, _ in teacher_list:
            teacher.eval()
        
        num_teachers = len(teacher_list)
        best_cost = 1e9
        
        # Track synthesis losses
        synthesis_losses = []
        loss_components = {'bn': [], 'oh': [], 'adv': [], 'kl_uniform': [], 'diversity': [], 'total': []}
        
        best_embeddings = None
        best_attention_mask = None
        
        # Sample targets if not provided
        if targets is None:
            distributions_on_device = [distribution.to(self.device) for _, distribution in teacher_list]
            combined_distribution = torch.stack(distributions_on_device).mean(dim=0)
            targets = torch.multinomial(combined_distribution, self.synthesis_batch_size, replacement=True)
        else:
            targets = targets.to(self.device)
        
        # Sort targets for efficient batching (optional, but can help)
        targets, _ = torch.sort(targets)
        
        # Compute teacher weights for this batch
        teacher_weights, q_values = self._compute_teacher_weights(teacher_list)


        # --- Trainable Text Embeddings ---
        text_embeddings = self._get_trainable_text_embeddings(targets)
        text_embeddings = text_embeddings.detach().clone().requires_grad_()
        
        # Generate attention masks
        attention_mask = generate_attention_masks(
            batch_size=self.synthesis_batch_size,
            prompt_len=self.prompt_vec_len,
            max_len=self.kd_max_length,
            max_text_len=self.generator_max_active,
            min_text_len=self.generator_min_active,
            device=self.device
        )
        
        for it in range(self.iterations):
            # Clone prompt vectors for this iteration
            prompt_vecs_clone = self.prompt_vecs.clone().detach().requires_grad_()
            
            # Build full embedding with prompt + text + padding
            core_embed = torch.cat([
                prompt_vecs_clone.unsqueeze(0).expand(self.synthesis_batch_size, -1, -1),
                text_embeddings
            ], dim=1)
            
            pad_len = max(0, self.kd_max_length - core_embed.shape[1])
            if pad_len > 0:
                pad = torch.zeros(
                    (self.synthesis_batch_size, pad_len, self.embedding_dim),
                    device=self.device
                )
                embeddings = torch.cat([core_embed, pad], dim=1)
            else:
                embeddings = core_embed
            
            embeddings = embeddings[:, :self.kd_max_length, :]
            attention_mask_current = attention_mask[:, :self.kd_max_length]
            
            # Setup optimizer for this iteration
            optimizer = torch.optim.Adam([
                {'params': [prompt_vecs_clone]},
                {'params': [text_embeddings], 'lr': self.lr_z}
            ], lr=self.lr_g, betas=[0.5, 0.999])
            
            #############################################
            # Weighted Multi-Teacher Loss Computation
            #############################################
            
            # Get all teacher outputs and features
            teacher_outputs = []
            teacher_features = []
            
            for teacher, _ in teacher_list:
                result = teacher(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask_current,
                    return_features=True
                )
                t_out, t_feat = result
                
                # Extract logits from ModelOutput
                if hasattr(t_out, 'logits'):
                    teacher_logits = t_out.logits
                else:
                    teacher_logits = t_out
                
                teacher_outputs.append(teacher_logits)
                teacher_features.append(t_feat)
            
            # Stack teacher outputs: [num_teachers, batch_size, num_classes]
            teacher_outputs = torch.stack(teacher_outputs, dim=0)
            
            # Get weights for all samples: [batch_size, num_teachers]
            sample_weights = teacher_weights[:, targets].T
            sample_q_values = q_values[:, targets].T
            
            # Feature loss (BN loss)
            loss_bn = torch.zeros(1, device=self.device, requires_grad=True)
            if self.bn > 0:
                total_loss_bn = torch.tensor(0.0, device=self.device, requires_grad=True)
                for teacher_idx, (teacher, _) in enumerate(teacher_list):
                    if teacher_features[teacher_idx]:
                        feature_loss = compute_text_feature_loss(teacher, teacher_features[teacher_idx])
                        teacher_weight_sum = sample_weights[:, teacher_idx].sum()
                        total_loss_bn = total_loss_bn + teacher_weight_sum * feature_loss
                loss_bn = total_loss_bn / self.synthesis_batch_size
                loss_bn = self._stabilize_loss(loss_bn, self.bn_loss_window, self.bn)
            
            # Cross entropy loss (OH loss)
            ce_losses = torch.zeros(len(teacher_list), self.synthesis_batch_size, device=self.device)
            for teacher_idx in range(len(teacher_list)):
                ce_losses[teacher_idx] = F.cross_entropy(
                    teacher_outputs[teacher_idx], targets, reduction='none'
                )
            
            weighted_ce_losses = sample_weights * ce_losses.T
            loss_oh = weighted_ce_losses.sum() / self.synthesis_batch_size
            loss_oh = self._stabilize_loss(loss_oh, self.oh_loss_window, self.oh)
            
            # KL-from-Uniform loss (optional)
            loss_kl_uniform = torch.tensor(0.0, device=self.device, requires_grad=True)
            if self.kl_uniform > 0:
                uniform_target = torch.ones(self.num_classes, device=self.device) / self.num_classes
                uniform_target = uniform_target.unsqueeze(0).expand(self.synthesis_batch_size, -1)
                
                kl_losses = torch.zeros(len(teacher_list), self.synthesis_batch_size, device=self.device)
                for teacher_idx in range(len(teacher_list)):
                    teacher_probs = F.softmax(teacher_outputs[teacher_idx], dim=1)
                    kl_losses[teacher_idx] = F.kl_div(
                        teacher_probs.log(), uniform_target, reduction='none'
                    ).sum(dim=1)
                
                weighted_kl_losses = ((1-sample_q_values)**5) * kl_losses.T
                loss_kl_uniform = weighted_kl_losses.sum() / self.synthesis_batch_size
                loss_kl_uniform = self._stabilize_loss(loss_kl_uniform, self.kl_uniform_loss_window, self.kl_uniform)
            
            # Diversity loss (optional)
            loss_diversity = torch.tensor(0.0, device=self.device, requires_grad=True)
            # Note: Diversity loss for text features can be complex, skipping for now
            # Can be added similar to image-based DFKD if needed
            
            # Adversarial loss
            loss_adv = torch.tensor(0.0, device=self.device, requires_grad=True)
            if self.adv > 0 and (self.ep >= self.ep_start):
                student_result = student(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask_current
                )
                
                if hasattr(student_result, 'logits'):
                    student_output = student_result.logits
                else:
                    student_output = student_result
                
                adv_losses = torch.zeros(len(teacher_list), self.synthesis_batch_size, device=self.device)
                for teacher_idx in range(len(teacher_list)):
                    teacher_output = teacher_outputs[teacher_idx]
                    
                    s_pred = student_output.max(1)[1]
                    t_pred = teacher_output.max(1)[1]
                    mask = (s_pred == t_pred).float()
                    
                    kl_loss = kldiv(student_output, teacher_output, reduction='none').sum(dim=1)
                    adv_losses[teacher_idx] = -(kl_loss * mask)
                
                weighted_adv_losses = sample_weights * adv_losses.T
                loss_adv = weighted_adv_losses.sum() / self.synthesis_batch_size
                loss_adv = self._stabilize_loss(loss_adv, self.adv_loss_window, self.adv)
            
            # Combine losses
            loss = (self.bn * loss_bn +
                   self.oh * loss_oh +
                   self.adv * loss_adv +
                   self.kl_uniform * loss_kl_uniform +
                   self.diversity_loss_weight * loss_diversity)
            
            # Track best embeddings
            with torch.no_grad():
                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_embeddings = embeddings.data.clone()
                    best_attention_mask = attention_mask_current.clone()
            
            # Track losses
            synthesis_losses.append(loss.item())
            loss_components['bn'].append(loss_bn.item())
            loss_components['oh'].append(loss_oh.item())
            loss_components['adv'].append(loss_adv.item())
            loss_components['kl_uniform'].append(loss_kl_uniform.item())
            loss_components['diversity'].append(loss_diversity.item())
            loss_components['total'].append(loss.item())
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            
            if self.ismaml:
                if it == 0:
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.prompt_vecs, prompt_vecs_clone)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()
                else:
                    optimizer.step()
            else:
                optimizer.step()
        
        # # Update external momentum manager
        # if self.bn_mmt != 0 and best_embeddings is not None:
        #     for teacher, _ in teacher_list:
        #         with torch.no_grad():
        #             _, best_features = teacher(
        #                 inputs_embeds=best_embeddings,
        #                 attention_mask=best_attention_mask,
        #                 return_features=True
        #             )
        #         if best_features:
        #             update_text_stat_tracker_momentum(teacher, best_features, self.momentum_manager, mmt_rate=self.bn_mmt)
        
        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.prompt_vecs, prompt_vecs_clone)
            self.meta_optimizer.step()
        
        student.train()
        
        # Use best embeddings if available
        if best_embeddings is None:
            best_embeddings = embeddings.data
            best_attention_mask = attention_mask_current
        
        # Add to memory-efficient pool
        self.data_pool.add(best_embeddings, best_attention_mask, current_round, targets)
        
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
        
        return {"synthetic": best_embeddings, "attention_mask": best_attention_mask}, synthesis_info
    
    def sample(self, distribution=None):
        """
        Sample data from the synthesized dataset.
        
        Args:
            distribution: If provided, sample according to this distribution.
        
        Returns:
            Tuple of (embeddings, attention_masks, labels)
        """
        if distribution is None:
            distribution = torch.ones(self.num_classes) / self.num_classes
        return self.data_pool.sample_batch(distribution, self.sample_batch_size)


# ===========================
# DFKD Operations for Text
# ===========================

def perform_text_dfkd_with_buffer(current_round, kd_buffer, synthesizer, config, global_model_params, clients, teacher_models=None, student_model=None, wandb_run=None, synthesize_this_round=True):
    """
    Perform Data-Free Knowledge Distillation using buffered teacher models for text.
    Similar to image-based DFKD but adapted for text models.
    
    Args:
        current_round: Current federated learning round
        kd_buffer: Buffer containing (model_params, client_id) tuples
        synthesizer: FastTextMetaSynthesizer instance
        config: Configuration dictionary
        global_model_params: Current global model parameters
        clients: List of client objects
        teacher_models: Optional list of pre-created teacher models
        student_model: Optional pre-created student model
        wandb_run: Optional wandb run object for logging metrics
        synthesize_this_round: Whether to perform synthesis this round.
                               If False, only KD training with existing synthetic data is performed.
                               Default is True to preserve backward compatibility.
        
    Returns:
        Updated student model parameters or update
    """
    assert kd_buffer is not None and len(kd_buffer) > 0, f"KD buffer must have at least 1 model but got {len(kd_buffer)} models"
    
    # Extract teacher model parameters from buffer
    teacher_models_params = [model_params for model_params, _ in kd_buffer]
    
    # Get KD settings from config
    kd_settings = config['dfkd_settings']
    batch_size = kd_settings['kd_batch_size']
    lr = kd_settings['lr']
    T = kd_settings['T']
    warmup_rounds = kd_settings['warmup_rounds']
    kd_nb_of_iters = kd_settings['kd_nb_of_iters']
    
    device = config['train_device']
    
    # Get client class probabilities
    client_map = {client.client_id: client for client in clients}
    teacher_client_class_probs = []
    for _, client_id in kd_buffer:
        client = client_map[client_id]
        client_probs = client.class_proportions
        teacher_client_class_probs.append(client_probs)
    
    # Setup teacher models
    teacher_models_to_use = []
    for i, teacher_params in enumerate(teacher_models_params):
        assert teacher_models is not None, "Teacher models must be provided for DFKD"
        assert len(teacher_models) == len(teacher_models_params), f"Number of teacher models must match parameters"
        
        teacher_model = teacher_models[i]
        reset_model_state(teacher_model, reset_norm_stats=teacher_model.has_running_stats)
        teacher_model = load_model_params(teacher_model, teacher_params, device)
        teacher_model.eval()
        
        # Turn off gradients for teacher models
        if next(teacher_model.parameters()).requires_grad:
            for p in teacher_model.parameters():
                p.requires_grad = False
        teacher_models_to_use.append(teacher_model)
    
    # Setup student model
    assert student_model is not None, "Student model must be provided for DFKD"
    reset_model_state(student_model, reset_norm_stats=student_model.has_running_stats)
    student_model = load_model_params(student_model, global_model_params, device)
    
    # For synthesis, eval mode and turn off gradients
    student_model.eval()
    if next(student_model.parameters()).requires_grad:
        for p in student_model.parameters():
            p.requires_grad = False
    
    # Setup KD criterion and optimizer
    criterion = KLDiv(T=T, reduction='sum')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Create teacher list with distributions
    teacher_models_and_distributions = [
        (teacher_model, torch.tensor(distribution, dtype=torch.float32))
        for teacher_model, distribution in zip(teacher_models_to_use, teacher_client_class_probs)
    ]
    
    # Conditionally perform synthesis based on synthesize_this_round
    if synthesize_this_round:
        # Synthesis phase
        vis_results, synthesis_info = synthesizer.synthesize(teacher_models_and_distributions, student_model, current_round, targets=None)
        
        # Log synthesis metrics
        syn_components = synthesis_info['loss_components']
        best_cost = synthesis_info['best_cost']
        syn_time = synthesis_info['synthesis_time']
        
        logging.info(f"Round {current_round}: DFKD synthesis - Best Cost: {best_cost:.4f}, BN: {syn_components['bn'][-1]:.4f}, OH: {syn_components['oh'][-1]:.4f}, ADV: {syn_components['adv'][-1]:.4f}, KL: {syn_components['kl_uniform'][-1]:.4f}, DIV: {syn_components['diversity'][-1]:.4f}, Time: {syn_time:.2f}s")
        
        # Log to wandb if enabled
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
    
    # KD training phase (after warmup)
    if current_round > warmup_rounds:
        # For KD training, train mode and turn on gradients
        student_model.train()
        if not next(student_model.parameters()).requires_grad:
            for p in student_model.parameters():
                p.requires_grad = True
        
        # Knowledge Distillation Training
        total_loss = 0
        total_samples = 0
        step_losses = []
        
        start_time = time.time()
        
        # Convert teacher distributions to tensors
        teacher_distributions = [torch.tensor(probs, dtype=torch.float32) for probs in teacher_client_class_probs]
        
        # Pre-sample all batches
        all_batches = sample_multiple_text_batches_optimized(
            dataset=synthesizer.data_pool.dataset,
            distributions=teacher_distributions,
            batch_size=batch_size,
            num_batches_per_dist=kd_nb_of_iters
        )
        
        # Organize batches by iteration
        organized_batches = []
        num_teachers = len(teacher_models_to_use)
        
        for kd_iter in range(kd_nb_of_iters):
            iter_batches = []
            for teacher_idx in range(num_teachers):
                batch_idx = teacher_idx * kd_nb_of_iters + kd_iter
                if batch_idx < len(all_batches):
                    iter_batches.append(all_batches[batch_idx])
                else:
                    iter_batches.append((None, None, None))
            organized_batches.append(iter_batches)
        
        # KD training loop
        for kd_iter in range(kd_nb_of_iters):
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            iteration_batch_size = 0
            
            # Get pre-sampled batches for this iteration
            iter_batches = organized_batches[kd_iter]
            
            # For each teacher and its batch
            for teacher_idx, (teacher_model, (embeddings, attention_masks, labels)) in enumerate(zip(teacher_models_to_use, iter_batches)):
                if embeddings is None or embeddings.size(0) == 0:
                    continue
                
                # Move to device
                if embeddings.device != device:
                    embeddings = embeddings.to(device, non_blocking=True)
                if attention_masks.device != device:
                    attention_masks = attention_masks.to(device, non_blocking=True)
                if labels.device != device:
                    labels = labels.to(device, non_blocking=True)
                
                # Student forward pass
                s_result = student_model(
                    inputs_embeds=embeddings.detach(),
                    attention_mask=attention_masks
                )
                s_out = s_result.logits if hasattr(s_result, 'logits') else s_result
                
                # Teacher forward pass
                with torch.no_grad():
                    t_result = teacher_model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_masks
                    )
                    t_out = t_result.logits if hasattr(t_result, 'logits') else t_result
                
                # Loss computation
                teacher_loss = criterion(s_out, t_out.detach())
                
                # Add to accumulated loss
                loss = loss + teacher_loss
                iteration_batch_size += embeddings.size(0)
            
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
        
        # Calculate training metrics
        if len(step_losses) > 0:
            avg_loss = total_loss / len(step_losses)
            min_loss = min(step_losses)
            max_loss = max(step_losses)
            training_time = time.time() - start_time
            
            logging.info(f"Round {current_round}: DFKD KD training - Avg Loss: {avg_loss:.4f}, Min Loss: {min_loss:.4f}, Max Loss: {max_loss:.4f}, Steps: {len(step_losses)}, Time: {training_time:.2f}s")
            
            # Log to wandb if enabled
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

