import numpy as np
import sys
import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from transformers import T5EncoderModel, T5Config

def create_model(config):
    """
    Create a model for federated learning.
    
    Args:
        config (dict): Configuration for the model
        
    Returns:
        object: Model object
    """
    model_name = config.get("model_name", None)
    if model_name is None:
        model_name = {'cifar10': 'resnet18', 'cifar100': 'resnet18', 'femnist': 'convnet', 'news': 't5-small'}[config["dataset"]]
    num_classes = config['nb_of_classes']
    
    if model_name.lower() == 'resnet18':
        pretrained = True if config["pretrained_model_path"] is not None else False
        model = ResNet18(num_classes, pretrained=pretrained, pretrained_path=config["pretrained_model_path"])
        logging.info(f"Created ResNet18 model with {num_classes} classes")
        model.has_running_stats = True  # StatTracker modules have running_mean/running_var
    elif model_name.lower() == 'convnet':
        model = ConvNet(num_classes, in_channels=1)
        logging.info(f"Created ConvNet model with {num_classes} classes")
        model.has_running_stats = True  # StatTracker modules have running_mean/running_var
    elif model_name.lower() == 't5-small':
        # Text classification model for news dataset
        model = T5ForSequenceClassificationCustom(
            model_name_or_path='t5-small',
            num_labels=num_classes,
            freeze_embeddings=True
        )
        logging.info(f"Created T5-small model with {num_classes} classes")
        model.has_running_stats = True  # StatTrackerText modules have running_mean/running_var
    else:
        raise ValueError(f"Model {model_name} not supported. Available models: resnet18, convnet, t5-small")
    
    return model


#########################
# ResNet18 Model
#########################

import torch
import torch.nn as nn
import torch.nn.functional as F

class StatTracker(nn.Module):
    def __init__(self, num_channels, momentum=0.1):
        """
        Simple statistics tracker that mimics BatchNorm running statistics.
        Purely for model functionality, no synthesis-related state.
        
        Args:
            num_channels: C, number of feature channels
            momentum: for updating running_mean/var (like BatchNorm)
        """
        super().__init__()
        # Mimic BN.running_mean / running_var
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var',  torch.ones(num_channels))
        self.momentum = momentum

    def forward(self, x, return_features=False):
        """
        Forward pass that optionally returns current batch statistics.
        
        Args:
            x: input tensor [B, C, H, W]
            return_features: if True, return (output, mean, var)
        
        Returns:
            output tensor, or (output, mean, var) if return_features=True
        """
        if return_features:
            # For synthesis: compute stats WITH gradients preserved
            mean = x.mean(dim=[0,2,3])                    # → shape [C] (with gradients)
            var  = x.var(dim=[0,2,3], unbiased=False)     # → shape [C] (with gradients)
            
            # Update running stats (detach ONLY for buffer updates, not return values)
            if self.training:
                with torch.no_grad():
                    self.running_mean.mul_(1-self.momentum).add_(self.momentum * mean.detach())
                    self.running_var .mul_(1-self.momentum).add_(self.momentum * var.detach())
            
            # Return with gradients intact for synthesis loss computation
            return x, mean, var
        else:
            # For normal training: no gradient tracking for stats computation
            if self.training:
                with torch.no_grad():
                    # Compute stats only for running buffer updates, no gradients
                    mean = x.mean(dim=[0,2,3])
                    var  = x.var(dim=[0,2,3], unbiased=False)
                    self.running_mean.mul_(1-self.momentum).add_(self.momentum * mean)
                    self.running_var .mul_(1-self.momentum).add_(self.momentum * var)
            
            return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.stat_track1 = StatTracker(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stat_track2 = StatTracker(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                ('stat_track', StatTracker(self.expansion*planes))
            ]))

    def forward(self, x, return_features=False):
        if return_features:
            out, mean1, var1 = self.stat_track1(self.conv1(x), return_features=True)
            out = F.relu(out)
            out, mean2, var2 = self.stat_track2(self.conv2(out), return_features=True)
            
            # Check if shortcut has StatTracker (length > 1 means conv + StatTracker)
            if len(self.shortcut) > 1:
                shortcut_out, mean_sc, var_sc = self.shortcut.stat_track(self.shortcut.conv(x), return_features=True)
                features = [(mean1, var1), (mean2, var2), (mean_sc, var_sc)]
            else:
                shortcut_out = self.shortcut(x)
                features = [(mean1, var1), (mean2, var2)]
            
            out = F.relu(out + shortcut_out)
            return out, features
        else:
            out = self.stat_track1(self.conv1(x))
            out = F.relu(out)
            out = self.stat_track2(self.conv2(out))
            shortcut_out = self.shortcut(x)
            out = F.relu(out + shortcut_out)
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-10 images are 32x32, so we use a smaller initial stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stat_track1 = StatTracker(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        if return_features:
            out, mean1, var1 = self.stat_track1(self.conv1(x), return_features=True)
            out = F.relu(out)
            
            # Collect features from all layers
            all_features = [(mean1, var1)]
            
            # Process each layer and collect features
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    out, block_features = block(out, return_features=True)
                    all_features.extend(block_features)
            
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out, all_features
        else:
            out = self.stat_track1(self.conv1(x))
            out = F.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def get_features(self, x):
        """
        Return the feature representation before the final linear layer.
        Useful for knowledge distillation.
        """
        out = self.stat_track1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        return features

    def load_pretrained_weights(self, pretrained_path, strict=False):
        """
        Load pretrained weights from a checkpoint, handling different number of classes.
        
        Args:
            pretrained_path: Path to the pretrained model checkpoint
            strict: If True, requires exact match of all parameters
        
        Returns:
            bool: True if weights loaded successfully
        """
        try:
            # Load checkpoint
            if os.path.isfile(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                # Extract state dict (handle different checkpoint formats)
                if 'model_state_dict' in checkpoint:
                    pretrained_state_dict = checkpoint['model_state_dict']
                else:
                    pretrained_state_dict = checkpoint
                
                # Get current model state dict
                current_state_dict = self.state_dict()
                
                # Filter out linear layer if number of classes is different
                filtered_state_dict = {}
                for key, value in pretrained_state_dict.items():
                    if key.startswith('linear.'):
                        # Check if linear layer dimensions match
                        if key in current_state_dict and current_state_dict[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"Skipping {key} due to dimension mismatch: "
                                  f"pretrained {value.shape} vs current {current_state_dict[key].shape}")
                    else:
                        filtered_state_dict[key] = value
                
                # Load the filtered state dict
                self.load_state_dict(filtered_state_dict, strict=strict)
                
                print(f"Successfully loaded pretrained weights from {pretrained_path}")
                print(f"Loaded {len(filtered_state_dict)} out of {len(pretrained_state_dict)} parameters")
                
                return True
                
            else:
                print(f"No checkpoint found at {pretrained_path}")
                return False
                
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            return False


# Function to create a ResNet-18 model
def ResNet18(num_classes=10, pretrained=False, pretrained_path=None):
    """
    Create a ResNet-18 model with pretrained weights from Tiny ImageNet.
    
    Args: 
        num_classes: Number of classes for the final layer
        pretrained: If True, loads pretrained weights from Tiny ImageNet
        pretrained_path: Path to pretrained model checkpoint. If None, looks for default paths
    
    Returns:
        ResNet18 model with pretrained weights (except final layer if num_classes != 200)
    """

    # Create model
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

    if not pretrained:
        return model

    
    # If no pretrained path specified, look for default paths
    if pretrained_path is None:
        print("No pretrained path specified, looking for default paths")
        # Try to find pretrained weights in the pretrained_resnet18 folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(script_dir, '..', 'pretrained_resnet18', 'checkpoints')
        
        # Look for the most recent experiment directory
        if os.path.exists(checkpoints_dir):
            experiment_dirs = [d for d in os.listdir(checkpoints_dir) 
                             if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith('resnet18_tinyin_')]
            if experiment_dirs:
                # Get the most recent experiment directory
                latest_experiment = sorted(experiment_dirs)[-1]
                pretrained_dir = os.path.join(checkpoints_dir, latest_experiment)
                
                pretrained_path = os.path.join(pretrained_dir, 'best_model.pth')
    
    # Load pretrained weights
    if pretrained_path and os.path.exists(pretrained_path):
        success = model.load_pretrained_weights(pretrained_path, strict=False)
        if success:
            print(f"Model initialized with pretrained weights for {num_classes} classes")
        else:
            print(f"Failed to load pretrained weights, using random initialization")
    else:
        print(f"Pretrained weights not found at {pretrained_path}, using random initialization")
    
    return model

#########################
# ConvNet Model (FEMNIST)
#########################

# The model is adapted from the FedDyn paper (https://arxiv.org/abs/2111.04263)
class ConvNet(nn.Module):
    def __init__(self, num_classes=62, in_channels=1):
        super(ConvNet,self).__init__()
        self.n_cls = num_classes
        
        # Convolutional layers with StatTrackers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.stat_track1 = StatTracker(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.stat_track2 = StatTracker(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.stat_track3 = StatTracker(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, self.n_cls)

    def forward(self, x, return_features=False):

        if return_features:
            features = []

        # First conv block
        x = self.conv1(x) # 28x28 -> 26x26
        if return_features:
            x, mean1, var1 = self.stat_track1(x, return_features=True)
            features.append((mean1, var1))            
        else:
            x = self.stat_track1(x)
        x = F.relu(x)


        x = self.conv2(x) # 26x26 -> 24x24
        if return_features:
            x, mean2, var2 = self.stat_track2(x, return_features=True)
            features.append((mean2, var2))
        else:
            x = self.stat_track2(x)
        x = F.relu(x)
        x = self.pool(x) # 24x24 -> 12x12
        
        x = self.conv3(x) # 12x12 -> 10x10
        if return_features:
            x, mean3, var3 = self.stat_track3(x, return_features=True)
            features.append((mean3, var3))
        else:
            x = self.stat_track3(x)
        x = F.relu(x)
        x = self.pool(x) # 10x10 -> 5x5
        
        # Flatten
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if return_features:
            return x, features
        else:
            return x
        return x



#########################
# Utility Functions
#########################

def reset_stat_tracker_stats(model):
    """
    Reset only the StatTracker/StatTrackerText running statistics in a model.
    Useful for synthesis or when you need to clear only StatTracker state.
    
    Args:
        model: PyTorch model containing StatTracker or StatTrackerText modules
    """
    for module in model.modules():
        if isinstance(module, StatTracker):
            # Reset running statistics to initial values (for image models)
            module.running_mean.zero_()  # Reset to zeros
            module.running_var.fill_(1.0)  # Reset to ones
        elif isinstance(module, StatTrackerText):
            # Reset running statistics to initial values (for text models)
            module.running_mean.zero_()  # Reset to zeros
            module.running_var.fill_(1.0)  # Reset to ones
    return model

def reset_stat_tracker_stats_in_state_dict(state_dict):
    """
    Reset StatTracker/StatTrackerText running statistics in a state dictionary.
    Sets running_mean to zeros and running_var to ones for all StatTracker modules.
    Works for both image models (StatTracker) and text models (StatTrackerText).
    
    Args:
        state_dict (dict): Model state dictionary to modify
        
    Returns:
        dict: Modified state dictionary with reset StatTracker/StatTrackerText statistics
    """
    for key, value in state_dict.items():
        if 'stat_track' in key:
            if key.endswith('.running_mean'):
                state_dict[key] = torch.zeros_like(value)
            elif key.endswith('.running_var'):
                state_dict[key] = torch.ones_like(value)    
    return state_dict

def reset_model_state(model, reset_norm_stats=False):
    """
    Lightweight reset of model state to ensure deterministic behavior.
    Only clears the essential state that causes non-determinism.
    
    Args:
        model: PyTorch model to reset
        reset_norm_stats: Whether to reset normalization layer running statistics
    
    Normalization Layer State Behavior:
    - BatchNorm1d/2d/3d: HAS running_mean, running_var (needs reset)
    - SyncBatchNorm: HAS running_mean, running_var (needs reset) 
    - InstanceNorm1d/2d/3d: HAS running stats IF track_running_stats=True (needs reset)
    - GroupNorm: NO running stats (stateless, no reset needed)
    - LayerNorm: NO running stats (stateless, no reset needed)
    - LocalResponseNorm: NO running stats (stateless, no reset needed)
    - StatTracker: HAS running_mean, running_var (needs reset, for image models)
    - StatTrackerText: HAS running_mean, running_var (needs reset, for text models)
    """
    # Clear gradients (essential and cheap)
    model.zero_grad()
    
    # Clear the specific problematic state: forward hooks and features
    # This is the main source of non-determinism in ResNet models
    if hasattr(model, 'hooks') and hasattr(model, 'features'):
        # Only clear features dict, don't remove hooks (expensive)
        model.features = {}
    


    # Reset normalization layer running statistics if requested
    if reset_norm_stats:
        for module in model.modules():
            # BatchNorm variants (most common)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()
            
            # InstanceNorm (only if tracking running stats)
            elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if hasattr(module, 'track_running_stats') and module.track_running_stats:
                    if hasattr(module, 'reset_running_stats'):
                        module.reset_running_stats()
            
            # StatTracker modules (custom implementation for image models)
            elif isinstance(module, StatTracker):
                # Reset running statistics to initial values
                module.running_mean.zero_()  # Reset to zeros
                module.running_var.fill_(1.0)  # Reset to ones
            
            # StatTrackerText modules (custom implementation for text models)
            elif isinstance(module, StatTrackerText):
                # Reset running statistics to initial values
                module.running_mean.zero_()  # Reset to zeros
                module.running_var.fill_(1.0)  # Reset to ones
            
            # Note: GroupNorm, LayerNorm, LocalResponseNorm are stateless - no reset needed
    
    # Set to train mode (cheap and ensures consistent starting state)
    model.train()

def get_model_params(model, target_device=None):
    """
    Extract model parameters into a state dictionary.
    
    This includes both learnable parameters and registered buffers (e.g., BatchNorm's 
    running_mean/running_var, StatTracker's running_mean/running_var).
    
    Args:
        model (nn.Module): PyTorch model
        target_device (str, optional): Target device for parameters. 
            If None, keeps original device of each parameter.
    
    Returns:
        dict: State dictionary containing model parameters and buffers
    """
    if target_device is None:
        return {
            name: param.clone().detach() 
            for name, param in model.state_dict().items()
        }
    else:
        return {
            name: param.clone().detach().to(target_device)
            for name, param in model.state_dict().items()
        }
def load_model_params(model, params_dict, target_device=None):
    """
    Load parameters from a state dictionary into a model.
    
    This loads both learnable parameters and registered buffers (e.g., BatchNorm's 
    running_mean/running_var, StatTracker's running_mean/running_var).
    
    Args:
        model (nn.Module): PyTorch model to load parameters into
        params_dict (dict): State dictionary containing parameters and buffers
        target_device (str, optional): Target device for parameters.
            If None, keeps original device of each parameter.
    
    Returns:
        nn.Module: Model with loaded parameters and buffers on target_device
    """
    with torch.no_grad():
        for name in model.state_dict().keys():
            if name in params_dict:
                model.state_dict()[name].copy_(params_dict[name].to(model.state_dict()[name].device))
    
    if target_device is not None:
        model = model.to(target_device)
    
    return model

def get_model_diff(params_new, params_old, scale=1.0):
    """
    Calculate the scaled difference between two model parameter dictionaries.
    
    This computes differences for both learnable parameters and registered buffers
    (e.g., BatchNorm's running_mean/running_var, StatTracker's running_mean/running_var).
    
    Args:
        params_new (dict): New model parameters and buffers
        params_old (dict): Old model parameters and buffers
        scale (float): Scale factor to apply to the difference
    
    Returns:
        dict: Dictionary containing scaled parameter and buffer differences
    """
    diff_dict = {}
    with torch.no_grad():
        for name in params_new.keys():
            if name in params_old:
                diff_dict[name] = scale * (params_new[name] - params_old[name])
    
    return diff_dict

def add_update_to_params(params, update_dicts, scale=1.0, interpolate=False):
    """
    Add scaled parameter updates to a parameter dictionary.
    
    This applies updates to both learnable parameters and registered buffers
    (e.g., BatchNorm's running_mean/running_var, StatTracker's running_mean/running_var).
    
    Args:
        params (dict): Parameter and buffer dictionary to update
        update_dicts (Union[dict, List[dict]]): Dictionary or list of dictionaries containing parameter and buffer updates
        scale (float): Scale factor to apply to the updates
        interpolate (bool): If True, interpolate the updates instead of adding them
    Returns:
        dict: Updated parameter and buffer dictionary
    """
    if not isinstance(update_dicts, list):
        update_dicts = [update_dicts]
        
    updated_params = {}
    with torch.no_grad():
        for name in params.keys():
            if interpolate:
                updated_params[name] = params[name].clone()*(1-abs(scale))
                for update_dict in update_dicts:
                    if name in update_dict:
                        updated_params[name] += abs(scale) * update_dict[name] / len(update_dicts)
            else:    
                updated_params[name] = params[name].clone()
                for update_dict in update_dicts:
                    if name in update_dict:
                        updated_params[name] += scale * update_dict[name] / len(update_dicts)
    
    return updated_params



def average_updates(update_dicts):
    """
    Average a list of update (delta) dictionaries.

    Args:
        update_dicts: list of parameter delta dicts
    Returns:
        dict: averaged delta
    """
    avg = {}
    n = len(update_dicts)
    with torch.no_grad():
        for name in update_dicts[0].keys():
            avg[name] = sum(d[name] for d in update_dicts) / n
    return avg


def weighted_average_updates(update_dicts, weights):
    """
    Compute a weighted average of parameter delta dictionaries.

    Used by FedAC to form the weighted pseudo-gradient from buffered client updates.

    Args:
        update_dicts: list of parameter delta dicts (one per buffered client)
        weights: list of non-negative floats summing to 1.0

    Returns:
        dict: weighted-average delta
    """
    assert len(update_dicts) == len(weights), "update_dicts and weights must have the same length"
    weighted_avg = {}
    with torch.no_grad():
        for name in update_dicts[0].keys():
            weighted_avg[name] = sum(w * d[name] for w, d in zip(weights, update_dicts))
    return weighted_avg


def compute_cosine_similarity_param_dicts(dict1, dict2, eps=1e-10):
    """
    Compute cosine similarity between two parameter dictionaries, treated as flat vectors.

    Used by FedAC to score each client's update direction vs. the global model change:
        r_i = cos(x^{t-1} - x^tau, g_i^tau)

    Args:
        dict1: first parameter dict (e.g. global model change = x^{t-1} - x^tau)
        dict2: second parameter dict (e.g. client pseudo-gradient g_i^tau)
        eps: small constant to avoid division by zero when either vector is near-zero

    Returns:
        float: cosine similarity in [-1, 1], or 0.0 if either vector is near-zero
    """
    with torch.no_grad():
        flat1 = torch.cat([v.flatten().float() for v in dict1.values()])
        flat2 = torch.cat([v.flatten().float() for v in dict2.values()])
        norm1 = torch.norm(flat1).item()
        norm2 = torch.norm(flat2).item()
        if norm1 < eps or norm2 < eps:
            return 0.0
        cos_sim = torch.dot(flat1, flat2).item() / (norm1 * norm2)
    return float(cos_sim)


class ServerAMSGrad:
    """
    AMSGrad-style server optimizer for FADAS.
    Treats averaged client deltas as pseudo-gradients and applies
    bias-corrected AMSGrad updates to the global model parameters.
    """

    def __init__(self, param_names, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # step counter for bias correction

        # Per-parameter state, keyed by parameter name string
        # Lazy-initialized on first step to get correct shapes/devices
        self.m = {name: None for name in param_names}      # first moment
        self.v = {name: None for name in param_names}      # second moment
        self.v_hat = {name: None for name in param_names}   # max second moment

    def step(self, current_params, delta_avg, lr, lr_lambda=1.0):
        """
        Perform one AMSGrad update step.

        Args:
            current_params: dict of current model parameters
            delta_avg: dict of averaged client update deltas (pseudo-gradient)
            lr: base server learning rate (eta)
            lr_lambda: delay-adaptive scaling factor (1.0 or 1/max_staleness)

        Returns:
            dict: updated model parameters
        """
        self.t += 1
        updated_params = {}

        with torch.no_grad():
            for name in current_params.keys():
                if name not in delta_avg:
                    updated_params[name] = current_params[name].clone()
                    continue

                g = delta_avg[name]  # pseudo-gradient

                # Lazy init moments on first step
                if self.m[name] is None:
                    self.m[name] = torch.zeros_like(g)
                    self.v[name] = torch.zeros_like(g)
                    self.v_hat[name] = torch.zeros_like(g)

                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g

                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g

                # Max of second moments (AMSGrad)
                self.v_hat[name] = torch.max(self.v_hat[name], self.v[name])

                # Bias corrections
                bc1 = 1 - self.beta1 ** self.t
                bc2 = 1 - self.beta2 ** self.t

                # Compute step
                step = (lr * lr_lambda / bc1) * self.m[name] / (torch.sqrt(self.v_hat[name] / bc2) + self.eps)

                updated_params[name] = current_params[name] + step

        return updated_params

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'm': self.m,
            'v': self.v,
            'v_hat': self.v_hat
        }

    def load_state_dict(self, state):
        """Load optimizer state from checkpoint."""
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.t = state['t']
        self.m = state['m']
        self.v = state['v']
        self.v_hat = state['v_hat']


class ServerFedAC:
    """
    Nadam (prospective momentum) server optimizer for FedAC.

    Implements the adaptive server update described in:
        "Efficient Asynchronous Federated Learning with Prospective Momentum
         Aggregation and Fine-Grained Correction" (FedAC, AAAI 2024).

    We implement only the server-side component (Basic-WTG-APWM ablation):
      - Client weight evaluation via cosine similarity (temporal gradient score)
      - Adaptive server update with prospective weighted momentum (Nadam)

    The client-side fine-grained gradient correction is dropped for fairness
    (our clients use the same local Adam training as all other baselines).

    Update equations (no bias correction, as in the FedAC paper):
        m^t = beta1 * m^{t-1} + (1 - beta1) * g_t
        v^t = beta2 * v^{t-1} + (1 - beta2) * g_t^2
        m_hat^t = beta1 * m^t + (1 - beta1) * g_t   [prospective / Nadam step]
        x^t = x^{t-1} + eta_g * m_hat^t / (sqrt(v^t) + eps)

    where g_t is the cosine-similarity-weighted average of client pseudo-gradients.

    Recommended hyperparameters from the FedAC paper: beta1=0.6, beta2=0.9, eps=1e-8.
    """

    def __init__(self, param_names, beta1=0.6, beta2=0.9, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # aggregation step counter

        # Per-parameter moment estimates; lazily initialized on first step.
        self.m = {name: None for name in param_names}   # first moment
        self.v = {name: None for name in param_names}   # second moment (uncentered)

    def step(self, current_params, g_t, lr):
        """
        Perform one Nadam (prospective momentum) update step.

        Args:
            current_params: dict of current global model parameters
            g_t: dict of weighted-average client pseudo-gradients (g_t)
            lr: server learning rate (eta_g)

        Returns:
            dict: updated model parameters
        """
        self.t += 1
        updated_params = {}

        with torch.no_grad():
            for name in current_params.keys():
                if name not in g_t:
                    updated_params[name] = current_params[name].clone()
                    continue

                g = g_t[name]

                # Lazy initialization
                if self.m[name] is None:
                    self.m[name] = torch.zeros_like(g)
                    self.v[name] = torch.zeros_like(g)

                # Update first moment (standard momentum)
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g

                # Update second moment (uncentered variance)
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g

                # Prospective (Nadam) momentum: look one step ahead
                m_hat = self.beta1 * self.m[name] + (1 - self.beta1) * g

                # Nadam update (no bias correction, as in the FedAC paper)
                step = lr * m_hat / (torch.sqrt(self.v[name]) + self.eps)
                updated_params[name] = current_params[name] + step

        return updated_params

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'm': self.m,
            'v': self.v,
        }

    def load_state_dict(self, state):
        """Load optimizer state from checkpoint."""
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.t = state['t']
        self.m = state['m']
        self.v = state['v']


def combine_kd_with_original(original_update, kd_update, beta):
    """
    Combine the original client update with the KD update using beta weighting linearly.
    
    Formula: combined_update = (1 - beta) * original_update + beta * kd_update
    
    Args:
        original_update: The original update from the client
        kd_update: The KD-enhanced update 
        beta (float): Beta value for combining updates (0.0 = only original, 1.0 = only KD)
        
    Returns:
        dict: Combined update
    """
    if beta == 0.0:
        return original_update
    elif beta == 1.0:
        return kd_update
    
    # Combine updates using beta weighting
    combined_update = {}
    for name in original_update.keys():
        if name in kd_update:
            combined_update[name] = (1 - beta) * original_update[name] + beta * kd_update[name]
        else:
            combined_update[name] = original_update[name]
    
    return combined_update

def downweight_update_based_on_staleness(original_update, beta):
    """
    Downweight the original update based on staleness.
    
    Args:
        original_update: The original update from the client
        beta: The beta value for downweighting
    """
    if beta == 0.0:
        return original_update
    
    # Combine updates using beta weighting
    downweighted_update = {}
    for name in original_update.keys():
        downweighted_update[name] = (1 - beta) * original_update[name]
    
    return downweighted_update

############################################################
# News Task Model and Utilities
############################################################

# now keep only track of the first token index
class StatTrackerText(nn.Module):
    """
    Statistics tracker for text/transformer models that mimics BatchNorm running statistics.
    Tracks mean and variance of transformer layer outputs for synthesis.
    Similar to StatTracker but adapted for text with attention masking.
    """
    def __init__(self, num_features, momentum=0.1):
        """
        Args:
            num_features: Hidden dimension size (e.g., 512 for T5-small)
            momentum: For updating running_mean/var (like BatchNorm)
        """
        super().__init__()
        # Mimic BN.running_mean / running_var
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = momentum
        self.num_features = num_features
    
    def forward(self, x, attention_mask=None, return_features=False):
        """
        Forward pass that optionally returns current batch statistics.
        
        Args:
            x: input tensor [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len] where True means active token
            return_features: if True, return (output, mean, var)
        
        Returns:
            output tensor, or (output, mean, var) if return_features=True
        """
        if return_features:

            mean = x[:, 0, :]  # [B, hidden_dim]
            var = x[:, 0, :]    # [B, hidden_dim]
            
            
            # Update running stats (detach and average across batch for buffer updates)
            if self.training:
                with torch.no_grad():
                    # batch_mean = mean.mean(dim=0)  # [hidden_dim] - average across batch
                    # batch_var = var.mean(dim=0)    # [hidden_dim] - average across batch
                    batch_mean = x[:, 0, :].mean(dim=0)  # [hidden_dim] - average across batch
                    batch_var = x[:, 0, :].var(dim=0)    # [hidden_dim] - average across batch
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            
            # Return with gradients intact for synthesis loss computation
            return x, mean, var
        else:
            # For normal training: no gradient tracking for stats computation
            if self.training:
                with torch.no_grad():
                    mean = x[:, 0, :].mean(dim=0)  # [hidden_dim]
                    var = x[:, 0, :].var(dim=0)   # [hidden_dim]
                    
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
            
            return x
    
    def get_running_stats(self):
        """Get current running statistics."""
        return self.running_mean.clone(), self.running_var.clone()
    
    def reset_running_stats(self):
        """Reset running statistics."""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)

class FrozenEmbedding(nn.Module):
    """
    A wrapper around the T5 embedding layer that makes it completely non-trainable.
    This prevents accidental gradient computation even if requires_grad is set to True later.
    """
    def __init__(self, original_embedding):
        super().__init__()
        # Store the original embedding weights as a buffer (non-trainable)
        self.register_buffer('weight', original_embedding.weight.data.clone())
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.padding_idx = original_embedding.padding_idx
        
    def forward(self, input_ids):
        """Forward pass using frozen embeddings with no gradient computation."""
        with torch.no_grad():
            # Use F.embedding to avoid any gradient tracking
            return torch.nn.functional.embedding(
                input_ids, self.weight, self.padding_idx, 
                None, 2.0, False, False
            )
    
    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}, frozen=True'


class SafeT5EncoderWrapper(nn.Module):
    """
    Wrapper around T5EncoderModel that ensures embedding layer is completely frozen.
    """
    def __init__(self, t5_encoder):
        super().__init__()
        self.t5_encoder = t5_encoder
        
        # Replace the shared embedding layer with frozen version
        if hasattr(self.t5_encoder, 'shared'):
            original_embedding = self.t5_encoder.shared
            self.frozen_embedding = FrozenEmbedding(original_embedding)
            
            # Replace the original embedding with a dummy non-trainable version
            # This maintains compatibility while preventing training
            dummy_embedding = nn.Embedding.from_pretrained(
                original_embedding.weight.data, freeze=True
            )
            dummy_embedding.requires_grad_(False)
            self.t5_encoder.shared = dummy_embedding
        else:
            raise ValueError("T5 encoder does not have a shared embedding layer, check it!")
            # self.frozen_embedding = None
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """
        Forward pass that uses frozen embeddings for input_ids or passes through inputs_embeds.
        """
        if input_ids is not None and self.frozen_embedding is not None:
            # Use our frozen embedding layer
            inputs_embeds = self.frozen_embedding(input_ids)
            # Clear input_ids to prevent T5 from using its own embedding
            input_ids = None
        
        # Forward through the T5 encoder
        return self.t5_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped T5 encoder."""
        if name in ['t5_encoder', 'frozen_embedding']:
            return super().__getattr__(name)
        return getattr(self.t5_encoder, name)


class T5ForSequenceClassificationCustom(nn.Module):
    """
    Custom T5 model for sequence classification that supports inputs_embeds.
    Uses T5EncoderModel + classification head.
    """
    
    def __init__(self, model_name_or_path="t5-small", num_labels=20, freeze_embeddings=True):
        super().__init__()
        
        # Get config for dimensions
        config = T5Config.from_pretrained(model_name_or_path)
        self.hidden_size = config.d_model
        self.num_layers = config.num_layers
        self.num_labels = num_labels
        self.config = config
        
        # Load T5 encoder (supports inputs_embeds)
        t5_encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        
        # Wrap with safe encoder if freezing embeddings
        if freeze_embeddings:
            self.encoder = SafeT5EncoderWrapper(t5_encoder)
            self._embeddings_frozen = True
        else:
            self.encoder = t5_encoder
            self._embeddings_frozen = False
        
        # Add StatTrackerText modules for each transformer layer
        # Use "stat_track" prefix for state_dict compatibility
        self.stat_trackers = nn.ModuleDict()
        for i in range(self.num_layers):
            self.stat_trackers[f'stat_track_layer_{i+1}'] = StatTrackerText(self.hidden_size, momentum=0.1)
        
        # Classification head (similar to T5ForSequenceClassification)
        self.classification_head = T5ClassificationHead(self.hidden_size, num_labels)
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, return_features=False):
        """
        Forward pass supporting both input_ids and inputs_embeds.
        """
        # Handle input_ids case: prepend token 3200 to beginning of every sample
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            # Create tensor with token 3200 for each sample in batch
            prefix_token = torch.full((batch_size, 1), 3200, dtype=input_ids.dtype, device=device)
            
            # Prepend token 3200 to input_ids
            input_ids = torch.cat([prefix_token, input_ids], dim=1)
        
        # Handle inputs_embeds case: prepend embedding of token 3200 to beginning of every sample
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            
            # Get embedding for token 3200
            token_3200_embedding = self.encoder.encoder.embed_tokens.weight[3200].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
            token_3200_embedding = token_3200_embedding.expand(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
            token_3200_embedding = token_3200_embedding.to(device)
            
            # Prepend token 3200 embedding to inputs_embeds
            inputs_embeds = torch.cat([token_3200_embedding, inputs_embeds], dim=1)
            
        # Prepend True to attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            prefix_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Use T5 encoder with output_hidden_states to get all layer outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True  # Get all layer outputs
        )
        
        # Get sequence output and all hidden states
        sequence_output = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        all_hidden_states = encoder_outputs.hidden_states    # Tuple of 7 tensors: embeddings + 6 layers
        # print(f"nb of layers: {self.num_layers}")
        # print(self.stat_trackers.keys())
        # print(f"len, shape of all_hidden_states: {len(all_hidden_states)}, {all_hidden_states[0].shape}")
        # Pass through StatTrackerText for each transformer layer (skip embeddings at index 0)
        features = []
        for i in range(1, len(all_hidden_states)):  # Skip layer 0 (embeddings), use layers 1-6
            layer_output = all_hidden_states[i]  # (batch_size, seq_len, hidden_size)
            tracker_name = f'stat_track_layer_{i}'  # Map layer 1, etc.
            
            if tracker_name not in self.stat_trackers:
                raise ValueError(f"Stat tracker {tracker_name} not found, check it!")

            stat_tracker = self.stat_trackers[tracker_name]
            
            if return_features:
                _, mean, var = stat_tracker(layer_output, attention_mask, return_features=True)
                features.append((mean, var))
            else:
                # Still update running stats even if not returning features
                stat_tracker(layer_output, attention_mask, return_features=False)
        
        # Apply classification head
        logits = self.classification_head(sequence_output, attention_mask)
        
        # Prepare output
        output = type('ModelOutput', (), {})()
        output.logits = logits
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            output.loss = loss_fct(logits, labels)
        
        # Return features if requested (for synthesis)
        if return_features:
            return output, features
        
        return output
    
    @property
    def transformer(self):
        """Provide access to transformer for compatibility."""
        return self.encoder
    
    def save_pretrained(self, save_directory):
        """Save model."""
        self.encoder.save_pretrained(save_directory)
        # Save classification head separately
        torch.save(self.classification_head.state_dict(), 
                  f"{save_directory}/classification_head.bin")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels=20, freeze_embeddings=True):
        """Load pretrained model."""
        model = cls(model_name_or_path, num_labels, freeze_embeddings=freeze_embeddings)
        
        # Try to load classification head if it exists
        try:
            classification_head_path = f"{model_name_or_path}/classification_head.bin"
            model.classification_head.load_state_dict(
                torch.load(classification_head_path, map_location='cpu')
            )
        except:
            pass  # Use randomly initialized classification head
        
        return model
    
    def freeze_embeddings(self):
        """
        Freeze embeddings if not already frozen.
        This method is safe to call multiple times.
        """
        if not self._embeddings_frozen:
            # Wrap the current encoder with SafeT5EncoderWrapper
            self.encoder = SafeT5EncoderWrapper(self.encoder)
            self._embeddings_frozen = True
        
    def are_embeddings_frozen(self):
        """Check if embeddings are frozen."""
        return self._embeddings_frozen
    
    def get_embedding_info(self):
        """Get information about the embedding layer."""
        if self._embeddings_frozen and hasattr(self.encoder, 'frozen_embedding'):
            emb = self.encoder.frozen_embedding
            return {
                'frozen': True,
                'shape': emb.weight.shape,
                'num_embeddings': emb.num_embeddings,
                'embedding_dim': emb.embedding_dim,
                'is_buffer': True  # Stored as buffer, not parameter
            }
        else:
            # Try to get info from regular T5 encoder
            try:
                shared = self.encoder.shared if hasattr(self.encoder, 'shared') else None
                if shared is not None:
                    return {
                        'frozen': False,
                        'shape': shared.weight.shape,
                        'num_embeddings': shared.num_embeddings,
                        'embedding_dim': shared.embedding_dim,
                        'requires_grad': shared.weight.requires_grad
                    }
            except:
                pass
            return {'frozen': False, 'info': 'Unable to access embedding layer'}



# [CLS] pooling
class T5ClassificationHead(nn.Module):
    """Classification head for T5 sequence classification."""
    
    def __init__(self, hidden_size, num_labels, dropout_rate=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply classification head to hidden states.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - optional
        
        Returns:
            logits: (batch_size, num_labels)
        """

        # [CLS] pooling
        pooled_hidden_states = hidden_states[:, 0, :]  # (batch_size, hidden_size)
        
        # Apply classification head
        pooled_hidden_states = self.dropout(pooled_hidden_states)
        pooled_hidden_states = self.dense(pooled_hidden_states)
        pooled_hidden_states = torch.tanh(pooled_hidden_states)
        pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.out_proj(pooled_hidden_states)
        
        return logits