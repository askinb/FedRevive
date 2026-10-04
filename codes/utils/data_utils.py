import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import random
from torch.nn import functional as F

# Add kornia imports for differentiable transforms
import kornia.augmentation as K
import torch.nn as nn

class DataTransformManager:
    """
    Centralized data transformation manager that provides consistent transforms 
    across all parts of the federated learning system.
    
    Supports three transform formats:
    1. Standard torchvision transforms (for PIL/raw images)
    2. Tensor operations (for already-tensor data)
    3. Kornia transforms (for differentiable operations in DFKD)
    """
    
    # Centralized dataset specifications
    DATASET_SPECS = {
        'cifar10': {
            'num_classes': 10,
            'channels': 3,
            'img_size': 32,
            'mean': [0.4914, 0.4822, 0.4465],  # Standardized CIFAR10 values
            'std': [0.2023, 0.1994, 0.2010]
        },
        'cifar100': {
            'num_classes': 100,
            'channels': 3,
            'img_size': 32,
            'mean': [0.507, 0.487, 0.441],     # Standardized CIFAR100 values
            'std': [0.268, 0.257, 0.276]
        },
        'femnist': {
                    'num_classes': 62,
                    'channels': 1,
                    'img_size': 28,
                    'mean': [0.0],  # No normalization - keep in [0,1] range
                    'std': [1.0]    # No normalization - keep in [0,1] range
        },
        'news': {
            'num_classes': 20,  # 20NewsGroup dataset
            # No image-related specs for text classification task
        }
    }
    
    def __init__(self, dataset_name: str):
        """
        Initialize the transform manager for a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset ('cifar10', 'cifar100', etc.)
        """
        self.dataset_name = dataset_name.lower()
        
        if self.dataset_name not in self.DATASET_SPECS:
            raise ValueError(f"Dataset {self.dataset_name} not supported. "
                           f"Available datasets: {list(self.DATASET_SPECS.keys())}")
        
        self.specs = self.DATASET_SPECS[self.dataset_name]
        self.num_classes = self.specs['num_classes']
        
        # For text datasets (like 'news'), we don't need image transformations
        if self.dataset_name == 'news':
            # Text datasets don't have image-specific attributes
            self.mean = None
            self.std = None
            self.img_size = None
            # No need to initialize transform pipelines for text
        else:
            # Image datasets - initialize transforms
            self.mean = self.specs['mean']
            self.std = self.specs['std']
            self.img_size = self.specs['img_size']
            
            # Initialize different transform formats
            self._init_torchvision_transforms()
            self._init_tensor_transforms()
            self._init_kornia_transforms()
    
    def _init_torchvision_transforms(self):
        """Initialize standard torchvision transforms for PIL/raw images."""
        # Base transforms for all datasets
        base_train_transforms = [
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if self.dataset_name == 'femnist':
            base_train_transforms = []
        
        # Dataset-specific augmentations
        if self.dataset_name == 'cifar100':
            base_train_transforms.append(transforms.RandomRotation(15))
        
        # Complete training transform pipeline
        self.torchvision_train = transforms.Compose(
            base_train_transforms + [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
        )
        
        # Test/validation transforms (no augmentation)
        self.torchvision_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Transform without normalization (for special cases)
        self.torchvision_train_no_norm = transforms.Compose(
            base_train_transforms + [transforms.ToTensor()]
        )
        
        self.torchvision_test_no_norm = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def _init_tensor_transforms(self):
        """Initialize tensor-based transforms for already-tensor data."""
        # Custom tensor-based transforms
        class TensorRandomCrop(nn.Module):
            def __init__(self, size, padding=4):
                super().__init__()
                self.size = size if isinstance(size, (list, tuple)) else [size, size]
                self.padding = padding
                
            def forward(self, x):
                if self.padding > 0:
                    x = F.pad(x, [self.padding] * 4, mode='reflect')
                _, _, h, w = x.shape
                th, tw = self.size
                if h == th and w == tw:
                    return x
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                return x[:, :, i:i+th, j:j+tw]
        
        class TensorRandomHorizontalFlip(nn.Module):
            def __init__(self, p=0.5):
                super().__init__()
                self.p = p
                
            def forward(self, x):
                if torch.rand(1).item() < self.p:
                    return torch.flip(x, [-1])
                return x
        
        class TensorRandomRotation(nn.Module):
            def __init__(self, degrees):
                super().__init__()
                self.degrees = degrees
                
            def forward(self, x):
                B = x.shape[0]
                # sample in degrees → radians
                angles = (torch.rand(B, device=x.device) * 2 * self.degrees - self.degrees)
                angles = angles * math.pi / 180.0

                # build [B×2×3] rotation matrices with no translation
                theta = torch.zeros(B, 2, 3, device=x.device)
                cos = torch.cos(angles)
                sin = torch.sin(angles)
                theta[:, 0, 0] = cos
                theta[:, 0, 1] = -sin
                theta[:, 1, 0] = sin
                theta[:, 1, 1] = cos
                # theta[:, :, 2] stays zero

                # sample & apply
                grid = F.affine_grid(theta, x.size(), align_corners=True)
                return F.grid_sample(x, grid, align_corners=True)
        
        class TensorNormalize(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
                self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
                
            def forward(self, x):
                return (x - self.mean) / self.std
        
        # Build tensor transform pipelines
        train_tensor_ops = [
            TensorRandomCrop(self.img_size, padding=4),
            TensorRandomHorizontalFlip(),
        ]

        if self.dataset_name == 'femnist':
            train_tensor_ops = []
        
        if self.dataset_name == 'cifar100':
            train_tensor_ops.append(TensorRandomRotation(15))
        
        self.tensor_train = nn.Sequential(
            *train_tensor_ops,
            TensorNormalize(self.mean, self.std)
        )
        
        self.tensor_test = TensorNormalize(self.mean, self.std)
        
        # Without normalization versions
        self.tensor_train_no_norm = nn.Sequential(*train_tensor_ops)
        self.tensor_test_no_norm = nn.Identity()
    
    def _init_kornia_transforms(self):
        """Initialize kornia transforms for differentiable operations."""
        # Kornia augmentation pipeline
        train_kornia_ops = [
            K.RandomCrop(size=(self.img_size, self.img_size), padding=4, padding_mode='reflect'),
            K.RandomHorizontalFlip(p=0.5),
        ]
        
        if self.dataset_name == 'femnist':
            train_kornia_ops = []

        if self.dataset_name == 'cifar100':
            train_kornia_ops.append(K.RandomRotation(degrees=15))
        
        # Add normalization
        train_kornia_ops.append(
            K.Normalize(mean=torch.tensor(self.mean), std=torch.tensor(self.std))
        )
        
        self.kornia_train = K.AugmentationSequential(
            *train_kornia_ops,
            data_keys=["input"]  # Apply to input only
        )
        
        self.kornia_test = K.AugmentationSequential(
            K.Normalize(mean=torch.tensor(self.mean), std=torch.tensor(self.std)),
            data_keys=["input"]
        )
        
        # Without normalization versions
        self.kornia_train_no_norm = K.AugmentationSequential(
            *train_kornia_ops[:-1],  # Exclude normalization
            data_keys=["input"]
        )
        
        self.kornia_test_no_norm = K.AugmentationSequential(
            data_keys=["input"]  # Empty transform
        )
    
    def get_transforms(self, mode='train', format='torchvision', normalize=True):
        """
        Get transforms for the specified mode and format.
        
        Args:
            mode (str): 'train' or 'test'/'val'
            format (str): 'torchvision', 'tensor', or 'kornia'
            normalize (bool): Whether to include normalization
            
        Returns:
            Transform object for the specified configuration
        """
        if format == 'torchvision':
            if mode == 'train':
                return self.torchvision_train if normalize else self.torchvision_train_no_norm
            else:
                return self.torchvision_test if normalize else self.torchvision_test_no_norm
        elif format == 'tensor':
            if mode == 'train':
                return self.tensor_train if normalize else self.tensor_train_no_norm
            else:
                return self.tensor_test if normalize else self.tensor_test_no_norm
        elif format == 'kornia':
            if mode == 'train':
                return self.kornia_train if normalize else self.kornia_train_no_norm
            else:
                return self.kornia_test if normalize else self.kornia_test_no_norm
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'torchvision', 'tensor', or 'kornia'")
    
    def get_normalizer(self):
        """
        Get a reversible normalizer for the dataset.
        
        Returns:
            Normalizer object that can normalize and denormalize
        """
        class Normalizer:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std

            def __call__(self, x, reverse=False):
                if reverse:
                    _mean = [-m / s for m, s in zip(self.mean, self.std)]
                    _std = [1/s for s in self.std]
                else:
                    _mean = self.mean
                    _std = self.std
                
                _mean = torch.as_tensor(_mean, dtype=x.dtype, device=x.device)
                _std = torch.as_tensor(_std, dtype=x.dtype, device=x.device)
                x = (x - _mean[None, :, None, None]) / (_std[None, :, None, None])
                return x
        
        return Normalizer(self.mean, self.std)
    
    def get_specs(self):
        """Get dataset specifications."""
        return self.specs.copy()

# Global transform managers - initialize lazily
_transform_managers = {}

def get_transform_manager(dataset_name: str) -> DataTransformManager:
    """
    Get or create a transform manager for the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        DataTransformManager instance for the dataset
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _transform_managers:
        _transform_managers[dataset_name] = DataTransformManager(dataset_name)
    return _transform_managers[dataset_name]

class ExternalKDDataset(Dataset):
    """
    Wrapper class for external KD datasets loaded from .pth files.
    
    Handles datasets that are saved as dictionaries with 'images', 'labels', and 'length' keys.
    """
    
    def __init__(self, dataset_path, transform=None):
        """
        Initialize the external KD dataset.
        
        Args:
            dataset_path (str): Path to the .pth file containing the dataset
            transform (callable, optional): Transform to apply to the images
        """
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Load the dataset
        try:
            self.dataset_dict = torch.load(dataset_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load external KD dataset from {dataset_path}: {e}")
        
        # Validate the dataset structure
        if not isinstance(self.dataset_dict, dict):
            raise ValueError(f"External KD dataset must be a dictionary, got {type(self.dataset_dict)}")
        
        required_keys = ['images', 'labels']
        for key in required_keys:
            if key not in self.dataset_dict:
                raise ValueError(f"External KD dataset missing required key: {key}")
        
        self.images = self.dataset_dict['images']
        self.labels = self.dataset_dict['labels']
        
        # Get length from the dataset or calculate it
        if 'length' in self.dataset_dict:
            self.length = self.dataset_dict['length']
        else:
            self.length = len(self.images)
        
        # Validate data consistency
        if len(self.images) != len(self.labels):
            raise ValueError(f"Number of images ({len(self.images)}) doesn't match number of labels ({len(self.labels)})")
        
        if self.length != len(self.images):
            raise ValueError(f"Dataset length ({self.length}) doesn't match actual data length ({len(self.images)})")
        
        print(f"Loaded external KD dataset with {self.length} samples from {dataset_path}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Get item at index.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        
        # Handle label conversion
        if isinstance(label, torch.Tensor):
            if label.dim() > 0:
                label = label.item()
            else:
                label = int(label)
        elif not isinstance(label, int):
            label = int(label)
        
        # Apply transform if provided
        # Note: For external datasets that already contain tensors, we need to handle transforms carefully
        if self.transform:
            # If image is already a tensor, we need to handle the transform pipeline differently
            if isinstance(image, torch.Tensor):
                # Apply transforms that work with tensors
                image = self._apply_tensor_transforms(image)
            else:
                # Apply normal transforms for PIL/numpy images
                image = self.transform(image)
        
        return image, label

    def _apply_tensor_transforms(self, image):
        """
        Apply transforms to tensor images efficiently using centralized transforms.
        
        Args:
            image (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Transformed image tensor
        """
        if self.transform is None:
            return image
        
        # New efficient approach: use tensor transforms directly if available
        if hasattr(self.transform, 'transforms'):
            # For tensor inputs, apply tensor-compatible transforms
            for transform in self.transform.transforms:
                if isinstance(transform, transforms.ToTensor):
                    # Skip ToTensor since image is already a tensor
                    continue
                elif isinstance(transform, transforms.Normalize):
                    # Apply normalization directly to tensor
                    image = transform(image)
                elif isinstance(transform, (transforms.RandomCrop, transforms.CenterCrop)):
                    # Apply crop transforms (these work with tensors)
                    image = transform(image)
                elif isinstance(transform, transforms.RandomHorizontalFlip):
                    # Apply horizontal flip (works with tensors)
                    image = transform(image)
                elif isinstance(transform, transforms.RandomRotation):
                    # Apply rotation (works with tensors)
                    image = transform(image)
                # Add more tensor-compatible transforms as needed
        else:
            # Fallback: apply transform directly (assuming it handles tensors)
            image = self.transform(image)
        
        return image

class FEMNISTDataset(Dataset):
    """
    Custom Dataset class for FEMNIST that loads all data from selected client partitions.
    Follows the same pattern as CIFAR datasets - centralized dataset with index-based splitting.
    """
    
    def __init__(self, client_partition_ids, transform=None):
        """
        Initialize FEMNIST dataset with selected client partition IDs.
        Loads all data from these partitions into memory for efficient access.
        
        Args:
            client_partition_ids (list): List of partition IDs for the selected clients
            transform: Transform to apply to images
        """
        # FEMNIST imports
        try:
            from flwr_datasets import FederatedDataset
            from flwr_datasets.partitioner import NaturalIdPartitioner
            FEMNIST_AVAILABLE = True
        except ImportError:
            print("Warning: flwr_datasets not available. FEMNIST dataset cannot work.")
            FEMNIST_AVAILABLE = False
        if not FEMNIST_AVAILABLE:
            raise ImportError("FEMNIST dataset requires flwr_datasets. Please install it by running 'pip install flwr-datasets[vision]'.")
        
        self.transform = transform
        
        # Create federated dataset
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
        )
        
        # Load all data from selected client partitions
        self.images = []
        self.labels = []
        self.client_indices = [[] for _ in range(len(client_partition_ids))]  # List of lists: client_indices[i] = [indices for client i]
        
        print(f"Loading FEMNIST data from {len(client_partition_ids)} client partitions...")
        
        current_global_idx = 0
        for client_id, partition_id in enumerate(client_partition_ids):
            partition = fds.load_partition(partition_id=partition_id)
            partition_length = len(partition)
            print(f"  Client {client_id} (Partition {partition_id}): {partition_length} samples")
            
            client_start_idx = current_global_idx
            
            # Load all samples from this partition
            for local_idx in range(partition_length):
                sample = partition[local_idx]
                self.images.append(sample['image'])  # PIL Image
                self.labels.append(sample['character'])  # Class label
                self.client_indices[client_id].append(current_global_idx)  # Add index to this client's list
                current_global_idx += 1
            
            client_end_idx = current_global_idx
            print(f"  Client {client_id} (Partition {partition_id}): {partition_length} samples, indices {client_start_idx}-{client_end_idx-1}")
        
        self.total_length = len(self.images)
        self.client_partition_ids = client_partition_ids
        print(f"Total samples in FEMNIST dataset: {self.total_length}")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Get item at global index.
        
        Args:
            idx (int): Global index
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        image = self.images[idx]  # PIL Image
        label = self.labels[idx]  # Class label (integer)
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_client_indices(self, client_id):
        """
        Get all indices belonging to a specific client.
        
        Args:
            client_id (int): Client ID
            
        Returns:
            list: List of global indices for this client
        """
        if client_id >= len(self.client_indices):
            return []
        return self.client_indices[client_id]
    
    def update_client_indices_after_splits(self, remaining_train_indices):
        """
        Update client indices to only include those that remain in the training set.
        This should be called after train/val/test/kd splits are created.
        
        Args:
            remaining_train_indices (list): List of indices that remain in training set
        """
        remaining_indices_set = set(remaining_train_indices)
        
        # Create mapping: old_index -> new_index for remaining indices
        old_to_new_index = {}
        for new_idx, old_idx in enumerate(remaining_train_indices):
            old_to_new_index[old_idx] = new_idx
        
        # Update each client's indices list
        for client_id in range(len(self.client_indices)):
            updated_client_indices = []
            
            # For each index this client had, check if it's still in training
            for old_idx in self.client_indices[client_id]:
                if old_idx in remaining_indices_set:
                    # Update to new index position
                    new_idx = old_to_new_index[old_idx]
                    updated_client_indices.append(new_idx)
            
            # Replace client's indices with updated list
            self.client_indices[client_id] = updated_client_indices
        
        # Update total length to match remaining training data
        self.total_length = len(remaining_train_indices)
        
        # Print summary
        client_counts = [len(indices) for indices in self.client_indices]
        print(f"Updated FEMNISTDataset client indices: {len(remaining_train_indices)} remaining samples")
        print(f"  Client sample distribution: {client_counts}")
    
    def get_client_sample_counts(self):
        """
        Get the number of samples for each client.
        
        Returns:
            dict: {client_id: sample_count}
        """
        return {client_id: len(indices) for client_id, indices in enumerate(self.client_indices)}

def select_femnist_client_partitions(num_clients, max_partitions=3597):
    """
    Select partition IDs for FEMNIST clients. If we need more clients than partitions,
    repeat some partitions randomly.
    
    Args:
        num_clients (int): Number of clients needed
        max_partitions (int): Maximum number of available partitions
    
    Returns:
        list: List of partition IDs, one per client
    """
    all_partition_ids = list(range(max_partitions))
    
    if num_clients <= max_partitions:
        # Randomly select unique partitions
        selected_partitions = random.sample(all_partition_ids, num_clients)
    else:
        # Need to repeat some partitions
        selected_partitions = all_partition_ids.copy()  # Use all partitions first
        remaining_clients = num_clients - max_partitions
        
        # Randomly select additional partitions (with replacement)
        additional_partitions = random.choices(all_partition_ids, k=remaining_clients)
        selected_partitions.extend(additional_partitions)
        
        # Shuffle to mix repeated partitions
        random.shuffle(selected_partitions)
    
    print(f"Selected {len(selected_partitions)} partition IDs for {num_clients} clients")
    return selected_partitions

class NewsGroupDataset(Dataset):
    """
    Custom Dataset class for 20NewsGroup text classification.
    Stores text data and labels for federated learning with tokenization.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize 20NewsGroup dataset.
        
        Args:
            texts (list): List of text strings
            labels (list): List of integer labels
            tokenizer: Tokenizer (e.g., from transformers library)
            max_length (int): Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_length = len(self.texts)
        
        # Validate data consistency
        if len(self.texts) != len(self.labels):
            raise ValueError(f"Number of texts ({len(self.texts)}) doesn't match number of labels ({len(self.labels)})")
        
        print(f"Loaded NewsGroup dataset with {self.total_length} samples")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Get item at index.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: Dictionary with tokenized inputs and label
        """
        if idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_length}")
        
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_number_of_classes(dataset_name):
    """Get number of classes for a dataset using the centralized transform manager."""
    transform_manager = get_transform_manager(dataset_name)
    return transform_manager.get_specs()['num_classes']

def load_external_kd_dataset(external_kd_path, dataset_name, is_training=True):
    """
    Load external KD dataset from a .pth file using centralized transforms.
    
    Args:
        external_kd_path (str): Path to the external KD dataset file
        dataset_name (str): Name of the base dataset (for determining transforms)
        is_training (bool): Whether this dataset is used for training (adds augmentations)
        
    Returns:
        ExternalKDDataset: Loaded external KD dataset with appropriate transforms
    """
    if not os.path.exists(external_kd_path):
        raise FileNotFoundError(f"External KD dataset not found at: {external_kd_path}")
    
    # Get centralized transforms
    transform_manager = get_transform_manager(dataset_name)
    mode = 'train' if is_training else 'test'
    transform = transform_manager.get_transforms(mode=mode, format='torchvision', normalize=True)
    
    # Load the external dataset
    external_dataset = ExternalKDDataset(external_kd_path, transform=transform)
    
    return external_dataset

def load_femnist(config=None, **kwargs):
    """
    Load FEMNIST dataset and split according to configuration using centralized transforms.
    Follows the same pattern as CIFAR datasets.
    
    Args:
        config: Configuration object with dataset parameters
        **kwargs: Additional arguments
        
    Returns:
        dict: Dictionary containing FEMNIST datasets
    """
    # FEMNIST imports
    try:
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import NaturalIdPartitioner
        FEMNIST_AVAILABLE = True
    except ImportError:
        print("Warning: flwr_datasets not available. FEMNIST dataset cannot work.")
        FEMNIST_AVAILABLE = False
    if not FEMNIST_AVAILABLE:
        raise ImportError("FEMNIST dataset requires flwr_datasets. Please install it by running 'pip install flwr-datasets[vision]'.")
    
    print("Loading FEMNIST dataset...")
    
    # Get centralized transforms
    transform_manager = get_transform_manager('femnist')
    transform_train = transform_manager.get_transforms(mode='train', format='torchvision', normalize=True)
    transform_test = transform_manager.get_transforms(mode='test', format='torchvision', normalize=True)
    
    # Get configuration parameters
    num_clients = config["num_clients"] 
    val_split = config["val_split"]
    test_split = config["test_split"]
    kd_split = config["kd_split"]
    
    # Select client partitions
    client_partition_ids = select_femnist_client_partitions(num_clients)
    
    # Create the full training dataset with training transforms (loads all client data)
    full_train_dataset = FEMNISTDataset(client_partition_ids, transform=transform_train)
    
    # Get all available indices (all samples from selected clients)
    all_available_indices = list(range(len(full_train_dataset)))
    
    # Calculate split sizes
    total_size = len(all_available_indices)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    kd_size = int(kd_split * total_size)
    train_size = total_size - val_size - test_size - kd_size
    
    print(f"FEMNIST dataset split sizes:")
    print(f"  Total samples from {num_clients} clients: {total_size}")
    print(f"  Validation: {val_size}")
    print(f"  Test: {test_size}")
    print(f"  KD: {kd_size}")
    print(f"  Train (remaining): {train_size}")
    
    # First, randomly select indices for val/test/kd (these will be EXCLUDED from training)
    shuffled_indices = all_available_indices.copy()
    random.shuffle(shuffled_indices)
    
    # Allocate indices for val/test/kd first
    val_indices = shuffled_indices[:val_size]
    test_indices = shuffled_indices[val_size:val_size + test_size]
    kd_indices = shuffled_indices[val_size + test_size:val_size + test_size + kd_size]
    
    # Remaining indices go to training (NO OVERLAP with val/test/kd)
    train_indices = shuffled_indices[val_size + test_size + kd_size:]
    
    print(f"  Actual train size: {len(train_indices)} (no overlap with val/test/kd)")
    
    # Verify no overlap
    all_split_indices = set(val_indices + test_indices + kd_indices + train_indices)
    assert len(all_split_indices) == total_size, "Index allocation error: missing indices"
    
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    kd_set = set(kd_indices)
    
    assert len(train_set & val_set) == 0, "Train and validation indices overlap!"
    assert len(train_set & test_set) == 0, "Train and test indices overlap!"
    assert len(train_set & kd_set) == 0, "Train and KD indices overlap!"
    assert len(val_set & test_set) == 0, "Validation and test indices overlap!"
    assert len(val_set & kd_set) == 0, "Validation and KD indices overlap!"
    assert len(test_set & kd_set) == 0, "Test and KD indices overlap!"
    
    print(f"  ✅ Verified: No overlap between train/val/test/kd splits")
    
    # Create KD dataset BEFORE updating client indices (uses original full dataset)
    kd_full_dataset = FEMNISTDataset(client_partition_ids, transform=transform_train)
    kd_dataset = Subset(kd_full_dataset, kd_indices)
    
    # CRITICAL: Update client indices in the training dataset to exclude val/test/kd samples
    # This ensures split_data_for_clients() gets correct indices
    full_train_dataset.update_client_indices_after_splits(train_indices)
    
    # Create datasets with appropriate transforms
    train_dataset = Subset(full_train_dataset, list(range(len(train_indices))))  # Use sequential indices now
    
    # Create validation and test datasets with test transforms
    full_test_dataset = FEMNISTDataset(client_partition_ids, transform=transform_test)
    val_dataset = Subset(full_test_dataset, val_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'kd_dataset': kd_dataset,
        'num_classes': transform_manager.get_specs()['num_classes'],
        'client_partition_ids': client_partition_ids,  # Store for client data splitting
        'full_train_dataset': full_train_dataset  # Store for client splitting
    }

def load_news(config=None, **kwargs):
    """
    Load 20NewsGroup dataset and split according to configuration.
    
    Args:
        config: Configuration object with dataset parameters
        **kwargs: Additional arguments (e.g., tokenizer, max_length)
        
    Returns:
        dict: Dictionary containing 20NewsGroup datasets
    """
    # Import sklearn for 20NewsGroup
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        raise ImportError("20NewsGroup dataset requires scikit-learn. Please install it by running 'pip install scikit-learn'.")
    
    print("Loading 20NewsGroup dataset...")
    
    # Get tokenizer and max_length from kwargs
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
    max_length = kwargs.get('max_length', 256)
    
    # Load training data
    train_data = fetch_20newsgroups(
        subset='train', 
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Load test data
    test_data_raw = fetch_20newsgroups(
        subset='test', 
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    print(f"20NewsGroup loaded: {len(train_data.data)} train samples, {len(test_data_raw.data)} test samples")
    
    # Get split ratios from config or use defaults
    val_split = config["val_split"] if config and "val_split" in config else 0.15
    test_split = config["test_split"] if config and "test_split" in config else 0.15
    kd_split = config["kd_split"] if config and "kd_split" in config else 0.15
    
    # Calculate sizes
    total_train_size = len(train_data.data)
    val_size = int(val_split * total_train_size)
    kd_size = int(kd_split * total_train_size)
    train_size = total_train_size - val_size - kd_size
    
    print(f"20NewsGroup dataset split sizes:")
    print(f"  Total training samples: {total_train_size}")
    print(f"  Validation: {val_size}")
    print(f"  KD: {kd_size}")
    print(f"  Train (remaining): {train_size}")
    
    # Create indices for splitting
    all_indices = list(range(total_train_size))
    random.shuffle(all_indices)
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    kd_indices = all_indices[train_size + val_size:]
    
    # Create training dataset
    train_texts = [train_data.data[i] for i in train_indices]
    train_labels = [train_data.target[i] for i in train_indices]
    train_dataset = NewsGroupDataset(train_texts, train_labels, tokenizer, max_length)
    
    # Create validation dataset
    val_texts = [train_data.data[i] for i in val_indices]
    val_labels = [train_data.target[i] for i in val_indices]
    val_dataset = NewsGroupDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create KD dataset
    kd_texts = [train_data.data[i] for i in kd_indices]
    kd_labels = [train_data.target[i] for i in kd_indices]
    kd_dataset = NewsGroupDataset(kd_texts, kd_labels, tokenizer, max_length)
    
    # Create test dataset from the separate test split
    test_dataset = NewsGroupDataset(
        test_data_raw.data, 
        test_data_raw.target.tolist(), 
        tokenizer, 
        max_length
    )
    
    # Get number of classes
    transform_manager = get_transform_manager('news')
    num_classes = transform_manager.get_specs()['num_classes']
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'kd_dataset': kd_dataset,
        'num_classes': num_classes
    }

def load_data(dataset_name, **kwargs):
    """
    Load dataset for federated learning.
    
    Args:
        dataset_name (str): Name of the dataset to load
        **kwargs: Additional arguments specific to the dataset
        
    Returns:
        dict: Dictionary containing data, such as:
            - train_dataset: Training dataset
            - val_dataset: Validation dataset
            - test_dataset: Test dataset
            - kd_dataset: Knowledge distillation dataset
            - num_classes: Number of classes
    """
    if dataset_name.lower() == 'cifar10':
        return load_cifar10(**kwargs)
    elif dataset_name.lower() == 'cifar100':
        return load_cifar100(**kwargs)
    elif dataset_name.lower() == 'femnist':
        return load_femnist(**kwargs)
    elif dataset_name.lower() == 'news':
        return load_news(**kwargs)
    else:
        print(f"Dataset {dataset_name} not implemented yet. Using dummy data.")
        # Return dummy data
        return {
            'train_dataset': np.zeros((100, 28, 28, 1)),
            'val_dataset': np.zeros((20, 28, 28, 1)),
            'test_dataset': np.zeros((20, 28, 28, 1)),
            'kd_dataset': np.zeros((20, 28, 28, 1)),
            'num_classes': 10
        }

def get_loss_fn(dataset_name):
    if dataset_name.lower() in ['cifar10', 'cifar100', 'femnist', 'news']:
        return F.cross_entropy
    else:
        return F.cross_entropy

def load_cifar10(config=None, **kwargs):
    """
    Load CIFAR10 dataset and split according to configuration using centralized transforms.
    
    Args:
        config: Configuration object with dataset parameters
        **kwargs: Additional arguments
        
    Returns:
        dict: Dictionary containing CIFAR10 datasets
    """
    data_dir = os.path.join('./../data', 'cifar10')
    os.makedirs(data_dir, exist_ok=True)
    
    # Get centralized transforms
    transform_manager = get_transform_manager('cifar10')
    transform_train = transform_manager.get_transforms(mode='train', format='torchvision', normalize=True)
    transform_test = transform_manager.get_transforms(mode='test', format='torchvision', normalize=True)
    
    # Load the full training dataset with training transforms
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    
    # Load the test dataset with test transforms
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    
    # Get split ratios from config or use defaults
    val_split = config["val_split"] if config and "val_split" in config else 0.15
    test_split = config["test_split"] if config and "test_split" in config else 0.15
    kd_split = config["kd_split"] if config and "kd_split" in config else 0.15
    
    # Calculate sizes
    total_train_size = len(full_train_dataset)
    val_size = int(val_split * total_train_size)
    kd_size = int(kd_split * total_train_size)
    train_size = total_train_size - val_size - kd_size
    
    # Split the training dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, kd_dataset = random_split(
        full_train_dataset, [train_size, val_size, kd_size], generator=generator)
    
    # Create a validation dataset with test transforms
    val_test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_test)
    val_dataset = Subset(val_test_dataset, val_dataset.indices)
    
    # Handle KD dataset - check if external KD dataset is provided
    external_kd_path = config.get("external_kd_dataset") if config else None
    
    if external_kd_path:
        # Use external KD dataset with training transforms (for KD training)
        kd_dataset = load_external_kd_dataset(external_kd_path, "cifar10", is_training=True)
        print(f"Using external KD dataset from: {external_kd_path}")
    else:
        # Create a KD dataset with training transforms (for KD training)
        kd_train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=transform_train)
        kd_dataset = Subset(kd_train_dataset, kd_dataset.indices)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'kd_dataset': kd_dataset,
        'num_classes': transform_manager.get_specs()['num_classes']
    }

def load_cifar100(config=None, **kwargs):
    """
    Load CIFAR100 dataset and split according to configuration using centralized transforms.
    
    Args:
        config: Configuration object with dataset parameters
        **kwargs: Additional arguments
        
    Returns:
        dict: Dictionary containing CIFAR100 datasets
    """
    data_dir = os.path.join('./../data', 'cifar100')
    os.makedirs(data_dir, exist_ok=True)
    
    # Get centralized transforms
    transform_manager = get_transform_manager('cifar100')
    transform_train = transform_manager.get_transforms(mode='train', format='torchvision', normalize=True)
    transform_test = transform_manager.get_transforms(mode='test', format='torchvision', normalize=True)
    
    # Load the full training dataset with training transforms
    full_train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    
    # Load the test dataset with test transforms
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    
    # Get split ratios from config or use defaults
    val_split = config["val_split"] if config and "val_split" in config else 0.15
    test_split = config["test_split"] if config and "test_split" in config else 0.15
    kd_split = config["kd_split"] if config and "kd_split" in config else 0.15
    
    # Calculate sizes
    total_train_size = len(full_train_dataset)
    val_size = int(val_split * total_train_size)
    kd_size = int(kd_split * total_train_size)
    train_size = total_train_size - val_size - kd_size
    
    # Split the training dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, kd_dataset = random_split(
        full_train_dataset, [train_size, val_size, kd_size], generator=generator)
    
    # Create a validation dataset with test transforms
    val_test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=transform_test)
    val_dataset = Subset(val_test_dataset, val_dataset.indices)
    
    # Handle KD dataset - check if external KD dataset is provided
    external_kd_path = config.get("external_kd_dataset") if config else None
    
    if external_kd_path:
        # Use external KD dataset with training transforms (for KD training)
        kd_dataset = load_external_kd_dataset(external_kd_path, "cifar100", is_training=True)
        print(f"Using external KD dataset from: {external_kd_path}")
    else:
        # Create a KD dataset with training transforms (for KD training)
        kd_train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=transform_train)
        kd_dataset = Subset(kd_train_dataset, kd_dataset.indices)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'kd_dataset': kd_dataset,
        'num_classes': transform_manager.get_specs()['num_classes']
    }

def split_data_for_clients(dataset, num_clients, distribution='iid', **kwargs):
    """
    Split the data among clients according to the specified distribution.
    
    Args:
        data (dict): Dictionary containing the dataset
        num_clients (int): Number of clients
        distribution (str): Distribution type ('iid', 'non_iid_dirichlet', etc.)
        **kwargs: Additional arguments specific to the distribution
        
    Returns:
        list: List of data dictionaries, one for each client
    """
    client_data = []
    train_dataset = dataset
    
    # Get client data size
    client_data_size = kwargs.get('client_data_size', 1000)

    # Check if this is a FEMNIST dataset (could be wrapped in Subset)
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if isinstance(base_dataset, FEMNISTDataset):
        print(f"Splitting FEMNIST data among {num_clients} clients using the pre-selected partition IDs.")
        print(f"FEMNIST dataset is naturally non-IID.")
        return split_femnist_for_clients(base_dataset, num_clients)
    
    if distribution.lower() == 'iid':
        client_data = split_iid(train_dataset, num_clients, client_data_size)
    elif distribution.lower() == 'non_iid_dirichlet':
        assert 'dirichlet_alpha' in kwargs, "dirichlet_alpha > 0 is required for non_iid_dirichlet distribution"
        dirichlet_alpha = kwargs['dirichlet_alpha'] 
        client_data = split_dirichlet(train_dataset, num_clients, client_data_size, dirichlet_alpha)
    else:
        print(f"Distribution {distribution} not implemented yet. Using IID distribution.")
        client_data = split_iid(train_dataset, num_clients, client_data_size)
    
    return client_data

def split_femnist_for_clients(femnist_data, num_clients):
    """
    Split FEMNIST data among clients using the pre-loaded dataset.
    Each client gets data from their assigned partition (naturally non-IID).
    Much simpler now with list-of-lists client indices structure.
    
    Args:
        femnist_data (dict): Dictionary returned by load_femnist() OR the dataset itself
        num_clients (int): Number of clients
        
    Returns:
        list: List of datasets, one for each client
    """
    # Handle both cases: femnist_data dict or direct dataset
    if isinstance(femnist_data, dict):
        full_train_dataset = femnist_data['full_train_dataset']
        client_partition_ids = femnist_data['client_partition_ids']
    else:
        # Direct dataset case (when called from split_data_for_clients)
        full_train_dataset = femnist_data
        client_partition_ids = full_train_dataset.client_partition_ids
    
    # Get client sample counts for verification
    client_sample_counts = full_train_dataset.get_client_sample_counts()
    print(f"Client sample distribution after splits: {client_sample_counts}")
    
    # Create individual client datasets - much simpler now!
    client_datasets = []
    
    for client_id in range(num_clients):
        # Get client indices directly from the list-of-lists structure
        client_indices = full_train_dataset.get_client_indices(client_id)
        
        # Create subset for this client
        client_dataset = Subset(full_train_dataset, client_indices)
        client_datasets.append(client_dataset)
        
        partition_id = client_partition_ids[client_id] if client_id < len(client_partition_ids) else "Unknown"
        print(f"Client {client_id}: Partition {partition_id}, {len(client_dataset)} training samples")
    
    # Verification
    total_client_samples = sum(len(dataset) for dataset in client_datasets)
    print(f"Total client samples: {total_client_samples}, Dataset size: {len(full_train_dataset)}")
    
    assert total_client_samples == len(full_train_dataset), \
        f"Mismatch: {total_client_samples} client samples != {len(full_train_dataset)} dataset size"
    
    return client_datasets

def split_iid(dataset, num_clients, client_data_size):
    """
    Split dataset IID (Independent and Identically Distributed) among clients.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        client_data_size: Number of samples per client
        
    Returns:
        List of datasets, one for each client
    """
    # Use indices that correspond to positions in the dataset
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    # Create a list to store each client's indices
    client_indices = []
    
    # Assign indices to clients
    index_position = 0
    for i in range(num_clients):
        # Check if we need to wrap around the dataset
        if index_position + client_data_size > len(all_indices):
            # Shuffle and restart from beginning
            random.shuffle(all_indices)
            index_position = 0
        
        # Assign indices to this client
        client_indices.append(all_indices[index_position:index_position + client_data_size])
        index_position += client_data_size
    
    # Convert indices to Subset objects with the dataset
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets

def split_dirichlet(dataset, num_clients, client_data_size, alpha):
    """
    Split dataset using Dirichlet distribution among clients.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        client_data_size: Number of samples per client
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of datasets, one for each client
    """
    # Check if this is a text dataset (NewsGroupDataset) that returns dicts
    is_text_dataset = isinstance(dataset, NewsGroupDataset)
    
    # Create a temporary dataloader with larger batch size for efficiency
    batch_size = 256  # Process data in larger batches for better performance
    temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get all indices and labels
    all_indices = []
    all_labels = []
    
    for batch_idx, batch_data in enumerate(temp_loader):
        # Handle different dataset return types
        if is_text_dataset:
            # Text datasets return dict with 'labels' key
            label_batch = batch_data['labels']
        else:
            # Image datasets return tuple (data, label)
            _, label_batch = batch_data
        
        # Get the actual batch size (last batch might be smaller)
        actual_batch_size = label_batch.size(0)
        
        # Process each item in the batch
        for i in range(actual_batch_size):
            # Calculate the global index
            idx = batch_idx * batch_size + i
            label = label_batch[i].item()
            all_indices.append(idx)
            all_labels.append(label)
    
    # Group indices by class
    indices_by_class = {}
    for idx, label in zip(all_indices, all_labels):
        if label not in indices_by_class:
            indices_by_class[label] = []
        indices_by_class[label].append(idx)
    
    # Shuffle indices within each class
    for label in indices_by_class:
        random.shuffle(indices_by_class[label])
    
    # Keep track of which indices have been used
    used_indices = {label: 0 for label in indices_by_class}
    
    # Create a list to store each client's indices
    client_indices = [[] for _ in range(num_clients)]
    
    # Number of classes
    num_classes = len(indices_by_class)
    
    # For each client, sample from Dirichlet to determine class distribution
    for i in range(num_clients):
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        
        # Calculate number of samples per class for this client
        samples_per_class = np.round(proportions * client_data_size).astype(int)
        samples_per_class = {label: count for label, count in zip(indices_by_class.keys(), samples_per_class)}
        
        # Adjust to ensure total is client_data_size
        total = sum(samples_per_class.values())
        if total < client_data_size:
            # Add remaining samples to a random class
            diff = client_data_size - total
            random_class = random.choice(list(samples_per_class.keys()))
            samples_per_class[random_class] += diff
        elif total > client_data_size:
            # Remove excess samples from classes with most samples
            diff = total - client_data_size
            while diff > 0:
                max_class = max(samples_per_class, key=samples_per_class.get)
                samples_per_class[max_class] -= 1
                diff -= 1
        
        # Assign indices to this client
        for label, count in samples_per_class.items():
            if count <= 0:
                continue
                
            # Get the indices for this class
            class_indices = indices_by_class[label]
            
            # Check if we need to wrap around for this class
            if used_indices[label] + count > len(class_indices):
                # Shuffle and restart from beginning for this class
                random.shuffle(class_indices)
                used_indices[label] = 0
            
            # Assign indices to this client
            client_indices[i].extend(class_indices[used_indices[label]:used_indices[label] + count])
            used_indices[label] += count
    
    # Convert indices to Subset objects
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets


