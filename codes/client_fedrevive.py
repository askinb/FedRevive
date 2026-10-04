import time
import numpy as np
from typing import Dict, Any, Union, Optional
import torch
from torch.utils.data import DataLoader
from utils.data_utils import get_loss_fn

class Client:
    """
    Client class for federated learning.
    """
    
    def __init__(self, client_id, data=None, config=None):
        """
        Initialize a client.
        
        Args:
            client_id (int): Unique identifier for the client
            data (dict, optional): Client's training data
            config (dict, optional): Client-specific configuration
        """
        self.client_id = client_id
        self.data = data
        self.config = config
        
        # Three delay generators
        self.train_delay_generator = None
        self.download_delay_generator = None
        self.upload_delay_generator = None
        
        # Track samples per class
        self.samples_per_class = None
        
        # Dataset and dataloader
        self.dataset = None
        self.dataloader = None
        
        # Local model
        self.local_model = None
        self.model_version = 0
        
        # Track events associated with this client
        self.assigned_events = set()
    
    def set_delay_generators(self, train_config, download_config, upload_config):
        """
        Set the delay generators for the client.
        
        Args:
            train_config (dict): Configuration for training delay
            download_config (dict): Configuration for download delay
            upload_config (dict): Configuration for upload delay
        """
        self.delay_selected_means = {}
        self.train_delay_generator = self._create_delay_generator(train_config, type = 'train')
        self.download_delay_generator = self._create_delay_generator(download_config, type = 'download')
        self.upload_delay_generator = self._create_delay_generator(upload_config, type = 'upload')
    
    def _create_delay_generator(self, config, type = 'train'):
        """
        Create a delay generator based on the configuration.
        
        Args:
            config (dict): Configuration for the delay generator
            
        Returns:
            function: A function that generates delays when called
        """
        delay_type = config.get("type", "constant")
        mean_distribution = config.get("mean_distribution", [(1.0, 1.0)])
        std = config.get("std", 0.1)
        
        # Normalize probabilities in mean_distribution
        total_prob = sum(prob for prob, _ in mean_distribution)
        norm_mean_dist = [(prob/total_prob, mean) for prob, mean in mean_distribution]
        
        # Select mean value based on probability distribution
        def select_mean():
            r = np.random.random()
            cumulative_prob = 0
            for prob, mean in norm_mean_dist:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    return mean
            return norm_mean_dist[-1][1]  # Default to last mean if something goes wrong
        
        # Assign the selected mean
        self.delay_selected_means[type] = select_mean()

        if delay_type == "constant":
            return lambda: self.delay_selected_means[type]
        
        elif delay_type == "exponential":
            return lambda: np.random.exponential(self.delay_selected_means[type])
        
        elif delay_type == "uniform":
            # For uniform distribution around mean with standard deviation std
            return lambda: np.random.uniform(max(0, self.delay_selected_means[type] - std), self.delay_selected_means[type] + std)
        
        else:  # Default to constant
            return lambda: self.delay_selected_means[type]
    
    def set_dataset(self, dataset, batch_size):
        """
        Set the client's dataset and create a dataloader.
        
        Args:
            dataset: Dataset for the client
            batch_size (int): Batch size for training
        """
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Count samples per class
        self.count_samples_per_class()
    
    def count_samples_per_class(self):
        """
        Count the number of samples per class in the client's dataset.
        """
        if self.dataset is None:
            return
        
        # Initialize counts
        num_classes = self.config['nb_of_classes']
        self.samples_per_class = [0] * num_classes
        
        # Create a temporary dataloader with larger batch size for efficiency
        batch_size = 128
        temp_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
        # Determine dataset type by checking the first batch
        dataset_name = self.config.get('dataset', '').lower()
        
        # Count samples per class using the dataloader with batches
        if dataset_name in ['news']:
            # Text dataset: returns dictionary with 'labels' key
            for batch in temp_loader:
                label_batch = batch['labels']
                # Count occurrences of each class in this batch
                for label in range(num_classes):
                    self.samples_per_class[label] += (label_batch == label).sum().item()
        else:
            # Image dataset: returns tuple (data, target)
            for _, label_batch in temp_loader:
                # Count occurrences of each class in this batch
                for label in range(num_classes):
                    self.samples_per_class[label] += (label_batch == label).sum().item()
        
        # Set samples per class as proportions of total samples
        total_samples = sum(self.samples_per_class)
        if total_samples > 0:
            self.ground_truth_class_proportions = [count / total_samples for count in self.samples_per_class]
        else:
            self.ground_truth_class_proportions = [1.0 / num_classes] * num_classes
        
        # Initialize class_proportions with ground truth (will be overridden by server if estimation is used)
        # not even correct groundtruth
        self.class_proportions = [1.0 / num_classes] * num_classes # self.ground_truth_class_proportions.copy()
        
        # Flag to track if using estimated distributions
        self.using_estimated_distribution = False
    
    def local_train(self, model, config):
        """
        Train the local model on the client's data.
        """
        # Copy the initial model parame
        # Create optimizer 

        model.train()
        if config['local_optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['local_learning_rate'], momentum=config['local_momentum'])
        elif config['local_optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['local_learning_rate'])

        # Train the model
        loss_fn = get_loss_fn(config['dataset'])
        
        # Image classification tasks (CIFAR10, CIFAR100, FEMNIST)
        if config['dataset'] in ['cifar10', 'cifar100', 'femnist']:
            iter_num = 0
            total_loss = 0
            total_correct = 0
            total_samples = 0

            continue_training_flag = True
            while continue_training_flag:
                for batch_idx, (data, target) in enumerate(self.dataloader):
                    data, target = data.to(config['train_device']), target.to(config['train_device'])
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    total_correct += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += len(data)

                    iter_num += 1
                    if iter_num >= config['local_iters']:
                        # Calculate final metrics
                        avg_loss = total_loss / iter_num
                        accuracy = total_correct / total_samples 
                        continue_training_flag = False
                        break
        
        # Text classification task (20NewsGroup)
        elif config['dataset'] in ['news']:
            iter_num = 0
            total_loss = 0
            total_correct = 0
            total_samples = 0

            continue_training_flag = True
            while continue_training_flag:
                for batch_idx, batch in enumerate(self.dataloader):
                    # Text data comes as a dictionary with input_ids, attention_mask, labels
                    input_ids = batch['input_ids'].to(config['train_device'])
                    attention_mask = batch['attention_mask'].to(config['train_device'])
                    labels = batch['labels'].to(config['train_device'])
                    
                    optimizer.zero_grad()
                    
                    # T5 model forward pass
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                    # Extract logits and compute loss
                    logits = output.logits
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    total_loss += loss.item()
                    pred = logits.argmax(dim=1, keepdim=True)
                    total_correct += pred.eq(labels.view_as(pred)).sum().item()
                    total_samples += len(labels)

                    iter_num += 1
                    if iter_num >= config['local_iters']:
                        # Calculate final metrics
                        avg_loss = total_loss / iter_num
                        accuracy = total_correct / total_samples 
                        continue_training_flag = False
                        break
        
        else:
            raise ValueError(f"Dataset {config['dataset']} not supported in local_train")

        return {"final_local_model": model, "train_loss": avg_loss, "train_accuracy": accuracy}
    
    def evaluate(self, model=None):
        """
        Evaluate a model on the client's local data.
        Placeholder for now, will be implemented later.
        
        Args:
            model (optional): Model to evaluate. If None, evaluate local model.
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"Client {self.client_id} evaluating model")
        
        # Placeholder: This will be implemented later with actual evaluation logic
        
        # Return dummy metrics for now
        return {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0
        }
    
    def add_event(self, event_id, event_type=None):
        """
        Add an event ID to the client's assigned events
        
        Args:
            event_id (int): The ID of the event
            event_type (EventType, optional): The type of the event
        """
        self.assigned_events.add(event_id)
        
        # Track training events specifically
        if event_type and event_type.name == "LOCAL_TRAIN":
            if not hasattr(self, 'train_events'):
                self.train_events = set()
            self.train_events.add(event_id)
    
    def remove_event(self, event_id):
        """Remove an event ID from the client's assigned events"""
        if event_id in self.assigned_events:
            self.assigned_events.remove(event_id)
            
        # Also remove from train_events if present
        if hasattr(self, 'train_events') and event_id in self.train_events:
            self.train_events.remove(event_id)
    
    def has_train_event(self):
        """Check if client has any assigned training events"""
        if not hasattr(self, 'train_events'):
            self.train_events = set()
        return len(self.train_events) > 0
    
    def set_estimated_distribution(self, estimated_proportions):
        """
        Set the client's class proportions to estimated values.
        
        Args:
            estimated_proportions: List of estimated class proportions
        """
        self.class_proportions = estimated_proportions.copy()
        self.using_estimated_distribution = True
        
    def get_distribution_info(self):
        """
        Get information about current distribution being used.
        
        Returns:
            dict: Information about current distribution
        """
        return {
            'ground_truth_proportions': self.ground_truth_class_proportions,
            'current_proportions': self.class_proportions,
            'using_estimated': self.using_estimated_distribution
        } 