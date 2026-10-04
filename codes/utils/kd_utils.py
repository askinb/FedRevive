import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .model_utils import create_model, load_model_params, get_model_params, get_model_diff, reset_model_state
from .data_utils import NewsGroupDataset
from collections import defaultdict
import random


def is_text_dataset(dataset):
    """
    Check if a dataset is a text dataset (NewsGroupDataset).
    Handles datasets that might be wrapped in Subset.
    
    Args:
        dataset: PyTorch dataset to check
        
    Returns:
        bool: True if text dataset, False otherwise
    """
    from torch.utils.data import Subset
    # Unwrap Subset if needed
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    return isinstance(base_dataset, NewsGroupDataset)


def extract_batch_data(batch, is_text_dataset=False):
    """
    Extract inputs and labels from a batch, handling both image and text datasets.
    
    Args:
        batch: Batch data from dataloader
        is_text_dataset (bool): Whether this is from a text dataset
        
    Returns:
        tuple: (inputs, labels) where inputs could be tensor or dict
    """
    if is_text_dataset:
        # Text datasets return dict with 'input_ids', 'attention_mask', 'labels'
        # We return the full batch dict minus labels, and labels separately
        labels = batch['labels']
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        return inputs, labels
    else:
        # Image datasets return tuple (data, labels)
        return batch


def forward_model(model, inputs, is_text_model=False):
    """
    Forward pass through model, handling both image and text models.
    
    Args:
        model: PyTorch model
        inputs: Input data (tensor for image models, dict for text models)
        is_text_model (bool): Whether this is a text model
        
    Returns:
        torch.Tensor: Model output (logits)
    """
    if is_text_model:
        # Text models expect input_ids and attention_mask as keyword arguments
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # Text models return object with .logits attribute
        if hasattr(outputs, 'logits'):
            return outputs.logits
        else:
            return outputs
    else:
        # Image models expect tensor input and return tensor output (logits)
        return model(inputs)


def move_inputs_to_device(inputs, device, is_text_dataset=False):
    """
    Move inputs to the specified device, handling both image and text inputs.
    
    Args:
        inputs: Input data (tensor for images, dict for text)
        device: Target device
        is_text_dataset (bool): Whether this is from a text dataset
        
    Returns:
        Inputs moved to device
    """
    if is_text_dataset:
        # Move dict elements to device
        return {key: value.to(device) for key, value in inputs.items()}
    else:
        # Move tensor to device
        return inputs.to(device)


class LabelAwareBatchSampler(torch.utils.data.Sampler):
    """
    Custom Batch Sampler that enforces a specific label distribution for each batch.
    Each client can have its own sampler with its own label_probs distribution.
    """
    def __init__(self, labels, batch_size, label_probs, num_batches=None):
        """
        Args:
            labels: List of labels from the dataset
            batch_size: Size of each batch
            label_probs: Dictionary mapping label -> probability (e.g., {0: 0.3, 1: 0.2, ...})
            num_batches: Number of batches to generate (if None, will be len(labels) // batch_size)
        """
        self.labels = labels
        self.batch_size = batch_size
        self.label_probs = label_probs
        self.label_to_indices = self._build_label_index()
        self.num_batches = num_batches or max(1, len(labels) // batch_size)
        
        # Convert label_probs to lists for random.choices
        self.label_keys = list(self.label_probs.keys())
        self.label_weights = list(self.label_probs.values())

    def _build_label_index(self):
        """Build a mapping from label to list of indices that have that label."""
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_to_indices[int(label)].append(idx)
        return label_to_indices

    def __iter__(self):
        """Generate batches with the specified label distribution."""
        for _ in range(self.num_batches):
            batch = []
            attempts = 0
            max_attempts = self.batch_size * 10  # Prevent infinite loops
            
            while len(batch) < self.batch_size and attempts < max_attempts:
                # Choose a label according to the distribution
                chosen_label = random.choices(
                    population=self.label_keys,
                    weights=self.label_weights,
                    k=1
                )[0]
                
                # Get candidates for this label
                candidates = self.label_to_indices.get(chosen_label, [])
                if candidates:
                    # Add a random sample from this label
                    batch.append(random.choice(candidates))
                
                attempts += 1
            
            # If we couldn't fill the batch due to label constraints, 
            # fill remaining spots with random samples
            while len(batch) < self.batch_size:
                # Sample from any available label
                available_labels = [label for label in self.label_keys 
                                  if self.label_to_indices.get(label, [])]
                if available_labels:
                    random_label = random.choice(available_labels)
                    candidates = self.label_to_indices[random_label]
                    batch.append(random.choice(candidates))
                else:
                    # Fallback: random index
                    batch.append(random.randint(0, len(self.labels) - 1))
            
            yield batch

    def __len__(self):
        return self.num_batches


def extract_labels_from_dataset(dataset):
    """
    Extract all labels from a dataset.
    Handles both image datasets (tuple output) and text datasets (dict output).
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        List of labels
    """
    labels = []
    # Check if this is a text dataset
    is_text = is_text_dataset(dataset)
    
    # Use a temporary dataloader to extract labels efficiently
    temp_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    
    for batch in temp_loader:
        if is_text:
            # Text datasets return dict with 'labels' key
            label_batch = batch['labels']
        else:
            # Image datasets return tuple (data, labels)
            _, label_batch = batch
        
        # Handle both single labels and batch of labels
        if isinstance(label_batch, torch.Tensor):
            labels.extend(label_batch.tolist())
        else:
            labels.extend(label_batch)
    
    return labels


def create_client_kd_samplers_and_dataloaders(kd_dataset, clients, config):
    """
    Create LabelAwareBatchSampler and corresponding DataLoaders for each client
    based on their data distribution.
    
    Args:
        kd_dataset: The KD dataset to sample from
        clients: List of client objects with samples_per_class attribute
        config: Configuration object
        
    Returns:
        Tuple of (client_samplers, client_dataloaders) where both are dictionaries
        mapping client_id -> sampler/dataloader
    """
    # Extract labels from KD dataset
    kd_labels = extract_labels_from_dataset(kd_dataset)
    
    batch_size = config['simple_kd_settings']['batch_size']
    num_classes = config['nb_of_classes']
    
    # Calculate number of batches - we want enough batches to avoid running out
    # during multiple iterations of KD training
    num_batches_per_sampler = max(100, len(kd_labels) // batch_size)
    
    client_samplers = {}
    client_dataloaders = {}
    
    for client in clients:
        client_id = client.client_id

        # Retrieve client's label probabilities
        label_probs = {label: prob for label, prob in enumerate(client.class_proportions)}
        
        # Create sampler for this client
        sampler = LabelAwareBatchSampler(
            labels=kd_labels,
            batch_size=batch_size,
            label_probs=label_probs,
            num_batches=num_batches_per_sampler
        )
        
        # Create dataloader using this sampler
        dataloader = torch.utils.data.DataLoader(
            kd_dataset,
            batch_sampler=sampler,
            num_workers=0  # Keep it simple for now
        )
        
        client_samplers[client_id] = sampler
        client_dataloaders[client_id] = dataloader
    
    return client_samplers, client_dataloaders


def multi_model_kd_distillation_loss_distribution_match_batch_optimized(
    student_model, teacher_models_list, client_dataloaders, temperature, 
    teacher_client_ids, is_text_model=False
):
    """
    Optimized version using pre-created client-specific dataloaders.
    Each teacher model gets its own batch sampled according to its client's data distribution.
    With the new buffer approach, all client IDs are provided in teacher_client_ids.
    Supports both image and text models.
    
    Args:
        student_model: The student model
        teacher_models_list: List of teacher models
        client_dataloaders: Dictionary mapping client_id -> dataloader
        temperature: Temperature parameter for softening
        teacher_client_ids: List of client IDs corresponding to teacher models (all clients)
        is_text_model (bool): Whether this is a text model (default: False)
        
    Returns:
        Average KL divergence loss across all teacher models
    """
    device = next(student_model.parameters()).device
    
    assert len(teacher_client_ids) == len(teacher_models_list), f"Number of teacher client IDs ({len(teacher_client_ids)}) must match number of teacher models ({len(teacher_models_list)}). Check the config or the code!"
    
    # Forward pass for student model on each batch and collect losses
    losses = []
    
    for i, (teacher_model, client_id) in enumerate(zip(teacher_models_list, teacher_client_ids)):
        if client_id in client_dataloaders:
            # Get dataloader for this client
            dataloader = client_dataloaders[client_id]
            # Create or get iterator for this dataloader
            try:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            except (StopIteration, RuntimeError):
                # If dataloader is exhausted or has issues, create a new iterator
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Extract inputs and labels (handles both image and text datasets)
            batch_inputs, batch_labels = extract_batch_data(batch, is_text_dataset=is_text_model)
            
            # Move to device
            batch_inputs = move_inputs_to_device(batch_inputs, device, is_text_dataset=is_text_model)
            batch_labels = batch_labels.to(device)
            
            # Student forward pass (handles both image and text models)
            student_outputs = forward_model(student_model, batch_inputs, is_text_model=is_text_model)
            
            # Teacher forward pass (no gradient needed)
            with torch.no_grad():
                teacher_outputs = forward_model(teacher_model, batch_inputs, is_text_model=is_text_model)
            
            # Calculate KL divergence loss for this teacher-student pair
            soft_student = F.log_softmax(student_outputs / temperature, dim=1)
            soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
            loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)
            
            losses.append(loss)
    
    # Simple average of all losses (no weighting as specified)
    avg_loss = torch.stack(losses).mean()
    
    return avg_loss


def simple_kd_distillation_loss(student_logits, teacher_logits, temperature):
    """
    Compute the simple knowledge distillation loss using KL divergence.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        temperature: Temperature parameter for softening
        
    Returns:
        KL divergence loss
    """
    # Simple knowledge distillation loss using KL divergence
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)
    
    return soft_loss


def weighted_class_kd_distillation_loss(student_logits, teacher_logits, temperature, class_probs):
    """
    Compute weighted KL divergence loss where each class term is weighted by the client's class distribution.
    
    Args:
        student_logits: Logits from the student model [batch_size, num_classes]
        teacher_logits: Logits from the teacher model [batch_size, num_classes]
        temperature: Temperature parameter for softening
        class_probs: Class probabilities from client's data distribution [num_classes]
        
    Returns:
        Weighted KL divergence loss
    """
    # Apply temperature and get probabilities
    soft_student = F.log_softmax(student_logits / temperature, dim=1)  # [batch_size, num_classes]
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)      # [batch_size, num_classes]
    
    # Convert class_probs to tensor if it's not already
    if not isinstance(class_probs, torch.Tensor):
        class_probs = torch.tensor(class_probs, dtype=soft_student.dtype, device=soft_student.device)
    
    # Ensure class_probs is on the same device and has the right shape
    class_probs = class_probs.to(soft_student.device)
    if class_probs.dim() == 1:
        class_probs = class_probs.unsqueeze(0)  # [1, num_classes]
    
    # Calculate pointwise KL divergence terms: P(i) * log(P(i)/Q(i)) = P(i) * (log(P(i)) - log(Q(i)))
    # In our case: soft_teacher * (soft_student - log(soft_teacher))
    # But since soft_student is already log_softmax and soft_teacher is softmax:
    # KL terms = soft_teacher * (soft_student - soft_teacher.log())
    kl_terms = soft_teacher * (soft_student - soft_teacher.log())  # [batch_size, num_classes]
    
    # Weight each class term by the client's class distribution
    weighted_kl_terms = kl_terms * class_probs  # Broadcasting: [batch_size, num_classes] * [1, num_classes]
    
    # Sum over classes and average over batch
    weighted_loss = -weighted_kl_terms.sum(dim=1).mean() * (temperature**2)
    
    return weighted_loss


def calculate_class_weights(kd_buffer, clients, num_classes):
    """
    Calculate weights for each class based on client data distributions.
    With the new buffer approach, all client IDs including current are in the kd_buffer.
    
    Args:
        kd_buffer: List of tuples (model_parameters, client_id) in most-recent-first order
        clients: List of client objects
        num_classes: Number of classes in the dataset
        
    Returns:
        dict: Class-based weights where keys are class indices and values are weight arrays
    """
    assert kd_buffer is not None and len(kd_buffer) > 0, f"KD buffer must have at least 1 model but got {len(kd_buffer)} models. Check the config or the code!"
    
    # Get client IDs from KD buffer (in the new approach, all clients including current are in buffer)
    client_ids = [client_id for _, client_id in kd_buffer]
    
    # Create mapping from client_id to client object
    client_map = {client.client_id: client for client in clients}
    
    # Calculate weights for each class
    class_weights = {}
    for class_idx in range(num_classes):
        # Get class ratios for each client
        class_ratios = []
        for client_id in client_ids:
            if client_id in client_map:
                client = client_map[client_id]
                total_samples = sum(client.samples_per_class)
                if total_samples > 0:
                    ratio = client.samples_per_class[class_idx] / total_samples
                else:
                    ratio = 0.0
            else:
                ratio = 0.0
            class_ratios.append(ratio)
        
        # Normalize ratios to get weights
        total_ratio = sum(class_ratios)
        if total_ratio > 0:
            weights = [ratio / total_ratio for ratio in class_ratios]
        else:
            # If all ratios are zero, use uniform weights
            weights = [1.0 / len(client_ids) for _ in client_ids]
        
        class_weights[class_idx] = weights
    
    # Final safety check: ensure all weights have the expected length
    expected_length = len(client_ids)
    for class_idx in class_weights:
        assert len(class_weights[class_idx]) == expected_length
    
    return class_weights


def weighted_avg_logits(teacher_logits_list, weights):
    """
    Compute weighted average of teacher logits.
    
    Args:
        teacher_logits_list: List of logits from teacher models
        weights: List of weights for each teacher
        
    Returns:
        Weighted average logits
    """
    weighted_logits = []
    for i, logits in enumerate(teacher_logits_list):
        weighted_logits.append(weights[i] * logits)
    return torch.stack(weighted_logits).sum(dim=0)


def weighted_avg_losses(losses, weights):
    """
    Compute weighted average of losses.
    
    Args:
        losses: List of loss tensors
        weights: List of weights for each loss
        
    Returns:
        Weighted average loss
    """
    weighted_losses = []
    for i, loss in enumerate(losses):
        weighted_losses.append(weights[i] * loss)
    return torch.stack(weighted_losses).sum()


def multi_model_kd_distillation_loss_avg_logits(student_logits, teacher_logits_list, temperature, class_weights=None, labels=None):
    """
    Compute KD loss by first averaging teacher logits, then computing KL divergence.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits_list: List of logits from teacher models
        temperature: Temperature parameter for softening
        class_weights: Dictionary of class-based weights (optional)
        labels: Ground truth labels for weighting (optional)
        
    Returns:
        KL divergence loss
    """
    if class_weights is not None:
        assert labels is not None, f"Labels must be provided for weighted averaging. Check the config or the code!"
        # Use weighted averaging based on class distributions
        batch_size = student_logits.size(0)
        avg_teacher_logits = torch.zeros_like(student_logits)
        
        for i in range(batch_size):
            label = labels[i].item()
            weights = class_weights[label] # .get(label, [1.0 / len(teacher_logits_list)] * len(teacher_logits_list))
            
            sample_logits = [logits[i:i+1] for logits in teacher_logits_list]
            avg_teacher_logits[i:i+1] = weighted_avg_logits(sample_logits, weights)
    else:
        # Simple average of teacher logits
        avg_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
    
    # Compute KD loss with averaged logits
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(avg_teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)
    
    return soft_loss


def multi_model_kd_distillation_loss_avg_loss(student_logits, teacher_logits_list, temperature, class_weights=None, labels=None, config=None, teacher_client_class_probs=None):
    """
    Compute KD loss by computing individual KL divergences, then averaging the losses.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits_list: List of logits from teacher models
        temperature: Temperature parameter for softening
        class_weights: Dictionary of class-based weights (optional)
        labels: Ground truth labels for weighting (optional)
        config: Configuration object (optional, for weighted_class_loss flag)
        teacher_client_class_probs: Class probabilities for each teacher model (optional)
        
    Returns:
        Average KL divergence loss
    """
    # Check if weighted class loss is enabled
    use_weighted_class_loss = config['simple_kd_settings']['weighted_class_loss']
    
    if class_weights is not None:
        assert labels is not None, f"Labels must be provided for weighted averaging. Check the config or the code!"
        # Use weighted averaging based on class distributions
        batch_size = student_logits.size(0)
        total_loss = 0.0
        
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        
        for i in range(batch_size):
            label = labels[i].item()
            weights = class_weights[label]
            
            # Compute individual losses for this sample
            sample_losses = []
            
            # Handle weighted class loss case
            if use_weighted_class_loss:
                assert teacher_client_class_probs is not None, f"Teacher client class probabilities must be provided for weighted class loss. Check the config or the code!"
                assert len(teacher_logits_list) == len(teacher_client_class_probs), f"Number of teacher logits ({len(teacher_logits_list)}) must match number of teacher client class probabilities ({len(teacher_client_class_probs)}). Check the config or the code!"
                
                # Process all teachers at once for weighted class loss
                for j, teacher_logits in enumerate(teacher_logits_list):
                    loss = weighted_class_kd_distillation_loss(
                        soft_student[i:i+1], 
                        teacher_logits[i:i+1], 
                        temperature, 
                        teacher_client_class_probs[j]
                    )
                    sample_losses.append(loss)
            else:
                # Process all teachers at once for regular KL divergence
                soft_teachers = [F.softmax(logits[i:i+1] / temperature, dim=1) for logits in teacher_logits_list]
                losses = [F.kl_div(soft_student[i:i+1], soft_teacher, reduction='batchmean') * (temperature**2) 
                         for soft_teacher in soft_teachers]
                sample_losses.extend(losses)
            
            # Weighted average of losses for this sample
            sample_weighted_loss = weighted_avg_losses(sample_losses, weights)
            total_loss += sample_weighted_loss
        
        return total_loss / batch_size
    else:
        # Simple averaging of losses
        losses = []
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        
        # Compute individual losses
        for j, teacher_logits in enumerate(teacher_logits_list):
            if use_weighted_class_loss and teacher_client_class_probs is not None and j < len(teacher_client_class_probs):
                # Use weighted class loss for this teacher
                loss = weighted_class_kd_distillation_loss(
                    student_logits, 
                    teacher_logits, 
                    temperature, 
                    teacher_client_class_probs[j]
                )
            else:
                # Use regular KL divergence
                soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
                loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature**2)
            losses.append(loss)
        
        # Average the losses
        avg_loss = torch.stack(losses).mean()
        
        return avg_loss


def perform_kd_with_buffer(kd_buffer, kd_dataloader, config, global_model_params, clients=None, teacher_models=None, student_model=None, client_kd_dataloaders=None):
    """
    Perform knowledge distillation using all models in the KD buffer as teachers.
    Supports both image and text models.
    
    Simple and clean approach:
    - All models in kd_buffer are teachers (most-recent-first order)
    - Student is based on the global model parameters
    - Current client is always at kd_buffer[0]
    - Returns the KD-enhanced model parameters or update
    
    Args:
        kd_buffer (list): List of (model_parameters, client_id) tuples in most-recent-first order
        kd_dataloader: DataLoader for knowledge distillation
        config: Configuration object containing KD settings
        global_model_params: Current global model parameters to use as student base
        clients: List of client objects (optional, for weighted averaging)
        teacher_models: Pre-created list of teacher models (optional, for performance)
        student_model: Pre-created student model (optional, for performance)
        client_kd_dataloaders: Dictionary mapping client_id -> dataloader (optional, for distribution_match_batch)
        
    Returns:
        dict: KD update (difference between distilled model and global model)
    """
    assert kd_buffer is not None and len(kd_buffer) > 0, f"KD buffer must have at least 1 model but got {len(kd_buffer)} models. Check the config or the code!"
    
    # Detect if this is a text model by checking the dataset
    is_text_model = is_text_dataset(kd_dataloader.dataset)
    
    # Extract teacher model parameters from buffer
    teacher_models_params = [model_params for model_params, _ in kd_buffer]
    # Get KD settings from config
    kd_settings = config['simple_kd_settings']
    learning_rate = kd_settings['learning_rate']
    nb_of_iters = kd_settings['nb_of_iters']
    temperature = kd_settings['temperature']
    kd_version = kd_settings['kd_version'] # avg_logits, avg_loss, distribution_match_batch
    weighted_avg = kd_settings['weighted_avg']
    weighted_class_loss = kd_settings['weighted_class_loss']
    
    device = config['train_device']
    
    # Calculate class weights if weighted averaging is enabled
    class_weights = None
    if weighted_avg:
        assert clients is not None, f"Clients must be provided for weighted averaging. Check the config or the code!"
        class_weights = calculate_class_weights(kd_buffer, clients, config['nb_of_classes'])
    
    # Extract client class probabilities for weighted class loss (only needed for avg_loss)
    teacher_client_class_probs = None
    if weighted_class_loss:
        assert kd_version == "avg_loss", f"Weighted class loss is only supported for avg_loss. Check the config or the code!"
        assert clients is not None, f"Clients must be provided for weighted class loss. Check the config or the code!"
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
        if teacher_models is None:
            teacher_model = create_model(config=config)
        else:
            assert len(teacher_models) == len(teacher_models_params), f"Number of teacher models ({len(teacher_models)}) must match number of teacher model parameters ({len(teacher_models_params)}). Check the config or the code!"
            teacher_model = teacher_models[i]
            reset_model_state(teacher_model, reset_norm_stats=teacher_model.has_running_stats)  # Comprehensive reset
        teacher_model = load_model_params(teacher_model, teacher_params, device)
        teacher_model.eval()
        teacher_models_to_use.append(teacher_model)

    # Use provided student model or create new one
    if student_model is None:
        student_model = create_model(config=config)
    else:
        reset_model_state(student_model, reset_norm_stats=student_model.has_running_stats)
    student_model = load_model_params(student_model, global_model_params, device)
    student_model.train()
    
    # Setup optimizer for student model
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    
    # Create iterator for the KD dataloader (only needed for non-distribution_match_batch versions)
    if kd_version != "distribution_match_batch":
        kd_iter = iter(kd_dataloader)
    
    # Perform multi-model knowledge distillation for specified iterations
    for iteration in range(nb_of_iters):
        optimizer.zero_grad()
        
        if kd_version == "distribution_match_batch":
            # For distribution_match_batch, we use different batches for each teacher
            # Extract client IDs directly from kd_buffer (much simpler!)
            teacher_client_ids = [client_id for _, client_id in kd_buffer]
            assert len(teacher_client_ids) == len(teacher_models_to_use), f"Number of teacher client IDs ({len(teacher_client_ids)}) must match number of teacher models ({len(teacher_models_to_use)}). Check the config or the code!"
            assert client_kd_dataloaders is not None, f"Client KD dataloaders must be provided for distribution_match_batch. Check the config or the code!"
            # Use optimized version with pre-created dataloaders (pass is_text_model flag)
            loss = multi_model_kd_distillation_loss_distribution_match_batch_optimized(
                student_model, 
                teacher_models_to_use, 
                client_kd_dataloaders, 
                temperature, 
                teacher_client_ids,
                is_text_model=is_text_model
            )
        else:
            # For avg_logits and avg_loss, use the existing approach with same batch
            # Get a batch from KD dataloader
            try:
                batch = next(kd_iter)
            except StopIteration:
                # If we run out of data, create a new iterator
                kd_iter = iter(kd_dataloader)
                batch = next(kd_iter)
            
            # Extract inputs and labels (handles both image and text datasets)
            inputs, labels = extract_batch_data(batch, is_text_dataset=is_text_model)
            
            # Move to device
            inputs = move_inputs_to_device(inputs, device, is_text_dataset=is_text_model)
            labels = labels.to(device)
            
            # Forward pass for student model (handles both image and text models)
            student_outputs = forward_model(student_model, inputs, is_text_model=is_text_model)
            
            # Forward pass for all teacher models (no gradient calculation needed)
            teacher_outputs_list = []
            with torch.no_grad():
                for teacher_model in teacher_models_to_use:
                    teacher_outputs = forward_model(teacher_model, inputs, is_text_model=is_text_model)
                    teacher_outputs_list.append(teacher_outputs)
            
            # Calculate multi-model knowledge distillation loss
            if kd_version == "avg_logits":
                loss = multi_model_kd_distillation_loss_avg_logits(
                    student_outputs, teacher_outputs_list, temperature, class_weights, labels
                )
            elif kd_version == "avg_loss":
                loss = multi_model_kd_distillation_loss_avg_loss(
                    student_outputs, teacher_outputs_list, temperature, class_weights, labels, config, teacher_client_class_probs
                )
            else:
                raise ValueError(f"Unknown kd_version: {kd_version}")
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
    
    # Get the distilled model parameters
    distilled_model_params = get_model_params(student_model, config["model_device"])
    
    # If server_lr is negative, return the distilled model parameters
    # Else, return the difference between the distilled and current model
    if config['server_lr'] < 0:
        return distilled_model_params
    
    # Calculate the KD update (difference between distilled and global model)
    kd_update = get_model_diff(distilled_model_params, global_model_params)
    
    return kd_update



def sample_batch_by_distribution(kd_dataloader, client_class_probs, batch_size, num_classes):
    """
    Sample a batch from the KD dataloader according to a specific class distribution.
    
    Args:
        kd_dataloader: The KD dataloader to sample from
        client_class_probs: List/array of class probabilities for the client [prob_class_0, prob_class_1, ...]
        batch_size: Desired batch size
        num_classes: Number of classes in the dataset
        
    Returns:
        Tuple of (inputs, labels) sampled according to the distribution
    """
    # Convert to tensor if needed
    if not isinstance(client_class_probs, torch.Tensor):
        client_class_probs = torch.tensor(client_class_probs, dtype=torch.float32)
    
    # Normalize to ensure sum = 1
    client_class_probs = client_class_probs / client_class_probs.sum()
    
    # Calculate target number of samples per class
    target_samples_per_class = (client_class_probs * batch_size).round().int()
    
    # Adjust to ensure exact batch size
    total_samples = target_samples_per_class.sum().item()
    if total_samples != batch_size:
        # Adjust the class with highest probability
        max_class = torch.argmax(client_class_probs)
        target_samples_per_class[max_class] += batch_size - total_samples
    
    # Collect samples from the dataloader
    collected_inputs = []
    collected_labels = []
    samples_per_class = torch.zeros(num_classes, dtype=torch.int)
    
    # Create iterator for KD dataloader
    kd_iter = iter(kd_dataloader)
    
    while len(collected_inputs) < batch_size:
        try:
            inputs, labels = next(kd_iter)
        except StopIteration:
            # Reset iterator if we run out of data
            kd_iter = iter(kd_dataloader)
            inputs, labels = next(kd_iter)
        
        # Check each sample in the batch
        for i in range(inputs.size(0)):
            if len(collected_inputs) >= batch_size:
                break
                
            label = labels[i].item()
            
            # Check if we need more samples from this class
            if samples_per_class[label] < target_samples_per_class[label]:
                collected_inputs.append(inputs[i])
                collected_labels.append(labels[i])
                samples_per_class[label] += 1
    
    # Stack the collected samples
    batch_inputs = torch.stack(collected_inputs)
    batch_labels = torch.stack(collected_labels)
    
    return batch_inputs, batch_labels