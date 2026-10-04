import os
import json
import pickle
import time
import numpy as np
import random
import torch
from typing import Dict, Any, Optional, Tuple, Set
# Import only what's needed to avoid circular imports
# InMemorySyntheticDataset will be imported locally when needed
import logging


def check_if_resumable(experiment_dir: str) -> bool:
    """
    Check if the experiment directory is resumable.
    """
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint_exp_state")
    if not os.path.exists(checkpoint_dir):
        return False
    else:
        # Core required files for all algorithms
        required_files = [
            os.path.join(checkpoint_dir, "server_state.pth"),
            os.path.join(checkpoint_dir, "update_buffer.pth"),
            os.path.join(checkpoint_dir, "event_queue.json"),
            os.path.join(checkpoint_dir, "event_queue_data.pth"),
            os.path.join(checkpoint_dir, "estimation_state.pth")  # Add estimation state
        ]
        
        # Check if all core files exist
        core_files_exist = all(os.path.exists(f) for f in required_files)
        
        if not core_files_exist:
            return False
        
        # Additional check for DFKD files if DFKD is enabled
        # Load config to check if DFKD is enabled
        config_path = os.path.join(experiment_dir, "config.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check if DFKD is enabled
                kd_augmentation = config['kd_augmentation']
                
                if kd_augmentation == 'dfkd':
                    # DFKD is enabled, check for DFKD files
                    dfkd_files = [
                        os.path.join(checkpoint_dir, "dfkd_state.pth"),
                        os.path.join(checkpoint_dir, "dfkd_dataset.pth")
                    ]
                    return all(os.path.exists(f) for f in dfkd_files)
            except Exception:
                # If config loading fails, just check core files
                pass
        
        return True

def setup_wandb_resume(config, experiment_dir: str, try_to_resume: bool):
    """
    Setup WandB with resume capability if the run exists.
    
    Args:
        config: The configuration object
        experiment_dir: Path to the experiment directory
        try_to_resume: Whether to attempt resuming an existing run
        
    Returns:
        tuple: (wandb_run, is_resumed)
    """
    if not config["wandb_flag"]:
        return None, False
    
    try:
        wandb = __import__('wandb')
    except ImportError:
        config["wandb_flag"] = False
        return None, False
    
    # If not trying to resume, start new run
    if not try_to_resume:
        wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            group=config["wandb_group"]
        )
        # Save the run ID for future resume
        if wandb_run and wandb_run.id:
            wandb_id_file = os.path.join(experiment_dir, 'wandb_run_id.txt')
            with open(wandb_id_file, 'w') as f:
                f.write(wandb_run.id)
        wandb.config.update(config.get_dict())
        return wandb_run, False
    
    # Try to resume existing run
    wandb_id_file = os.path.join(experiment_dir, 'wandb_run_id.txt')
    if not os.path.exists(wandb_id_file):
        # No saved run ID, start new run
        wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            group=config["wandb_group"]
        )
        if wandb_run and wandb_run.id:
            with open(wandb_id_file, 'w') as f:
                f.write(wandb_run.id)
        wandb.config.update(config.get_dict())
        return wandb_run, False
    
    # Try to resume using saved run ID
    try:
        with open(wandb_id_file, 'r') as f:
            resume_id = f.read().strip()
        
        wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            group=config["wandb_group"],
            id=resume_id,
            resume="must"
        )
        print(f"Resumed WandB run: {resume_id}")
        return wandb_run, True
        
    except Exception as e:
        print(f"Failed to resume WandB run: {e}")
        print("Starting new WandB run...")
        # Start new run if resume fails
        wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            group=config["wandb_group"]
        )
        if wandb_run and wandb_run.id:
            with open(wandb_id_file, 'w') as f:
                f.write(wandb_run.id)
        wandb.config.update(config.get_dict())
        return wandb_run, False


def should_save_checkpoint(server,) -> bool:
    """
    Check if it's time to save a checkpoint based on wall-clock time.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if checkpoint should be saved
    """
    last_save_time = server.last_checkpoint_time
    current_time = time.time()
    elapsed = current_time - last_save_time
    return elapsed >= server.config["save_exp_state_interval"]*60

def atomic_torch_save(data, path, logger=None):
    """
    Atomically save torch data to prevent corruption.
    
    Args:
        data: Data to save
        path: Target path
        logger: Optional logger for messages
    """
    import tempfile
    import shutil
    
    # Create temp file in same directory to ensure atomic rename works
    temp_path = path + '.tmp'
    
    try:
        # Save to temporary file first
        torch.save(data, temp_path)
        
        # Verify the file can be loaded
        torch.load(temp_path, map_location='cpu')
        
        # Atomically rename temp file to final name
        shutil.move(temp_path, path)
        
        if logger:
            logger.info(f"Atomically saved to {path}")
        
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def save_experiment_state(server) -> None:
    """
    Save the experiment state to a checkpoint file.
    
    Args:
        server: The server object
        
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save server state (model versions, time, and version counter) atomically
    server_state_path = os.path.join(checkpoint_dir, "server_state.pth")
    try:
        server_state = {
            'model_version_parameters': server.model_version_parameters,
            'current_time': server.current_time,
            'global_model_version': server.global_model_version,
            'next_event_id': server.next_event_id,
            'rounds_completed': server.rounds_completed
        }
        
        # Use atomic save for large server state
        atomic_torch_save(server_state, server_state_path, server.logger)
        
    except Exception as e:
        server.logger.error(f"Failed to save server state: {e}")
        # Don't continue with other saves if server state fails
        return

    # Save metrics to JSON file
    metrics_path = os.path.join(checkpoint_dir, "metrics_ckpt.json")
    try:
        metrics_data = {
            'test_metrics': server.test_metrics,
            'train_metrics': server.train_metrics
        }
        
        # Atomic save for JSON too
        temp_metrics_path = metrics_path + '.tmp'
        with open(temp_metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Verify JSON is valid
        with open(temp_metrics_path, 'r') as f:
            json.load(f)
        
        # Atomic rename
        import shutil
        shutil.move(temp_metrics_path, metrics_path)
        server.logger.info(f"Metrics saved to {metrics_path}")
        
    except Exception as e:
        server.logger.error(f"Failed to save metrics: {e}")
        if os.path.exists(temp_metrics_path):
            os.remove(temp_metrics_path)

    # Save other components
    save_event_queue(server, checkpoint_dir)
    save_update_buffer(server, checkpoint_dir)
    save_kd_buffer(server, checkpoint_dir)
    save_dfkd_state(server, checkpoint_dir)
    save_random_states(server, checkpoint_dir)
    save_estimation_state(server, checkpoint_dir)


def save_event_queue(server, checkpoint_dir) -> None:
    """
    Save the event queue to checkpoint files.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to save checkpoint files
    """
    try:
        # Save simple event attributes to JSON
        events_list = []
        event_data_dict = {}
        
        for event in server.event_queue.queue:
            # Extract simple attributes
            event_simple = {
                'event_type': event.event_type.name,
                'start_time': event.start_time,
                'duration': event.duration,
                'finish_time': event.finish_time,
                'client_id': event.client.client_id if event.client else None,
                'event_id': event.event_id
            }
            events_list.append(event_simple)
            
            # Store complex data separately if it exists
            if event.data:
                event_data_dict[event.event_id] = event.data
        
        # Save simple attributes to JSON atomically
        events_json_path = os.path.join(checkpoint_dir, "event_queue.json")
        temp_json_path = events_json_path + '.tmp'
        with open(temp_json_path, 'w') as f:
            json.dump(events_list, f, indent=2)
        
        # Verify and atomically rename
        with open(temp_json_path, 'r') as f:
            json.load(f)
        import shutil
        shutil.move(temp_json_path, events_json_path)
        
        # Save complex event data to torch file atomically
        if event_data_dict:
            event_data_path = os.path.join(checkpoint_dir, "event_queue_data.pth")
            atomic_torch_save(event_data_dict, event_data_path, server.logger)
        
        server.logger.info(f"Event queue saved with {len(events_list)} events")
    except Exception as e:
        server.logger.error(f"Failed to save event queue: {e}")


def save_update_buffer(server, checkpoint_dir) -> None:
    """
    Save the update buffer to a checkpoint file.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to save checkpoint files
    """
    try:
        update_buffer_path = os.path.join(checkpoint_dir, "update_buffer.pth")
        atomic_torch_save(server.update_buffer, update_buffer_path, server.logger)
        server.logger.info(f"Update buffer saved with {len(server.update_buffer)} updates")
    except Exception as e:
        server.logger.error(f"Failed to save update buffer: {e}")

def save_kd_buffer(server, checkpoint_dir) -> None:
    """
    Save the KD buffer to a checkpoint file.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to save checkpoint files
    """
    try:
        kd_buffer_path = os.path.join(checkpoint_dir, "kd_buffer.pth")
        atomic_torch_save(server.kd_buffer, kd_buffer_path, server.logger)
        server.logger.info(f"KD buffer saved with {len(server.kd_buffer)} updates")
    except Exception as e:
        server.logger.error(f"Failed to save KD buffer: {e}")

def save_random_states(server, checkpoint_dir) -> None:
    """
    Save random number generator states for all libraries.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to save checkpoint files
    """
    try:
        random_states = {}
        
        # Save Python's built-in random state
        random_states['python_random'] = random.getstate()
        
        # Save NumPy random state
        random_states['numpy_random'] = np.random.get_state()
        
        # Save PyTorch CPU random state
        random_states['torch_cpu'] = torch.get_rng_state()
        
        # Save PyTorch CUDA random state if available
        if torch.cuda.is_available():
            random_states['torch_cuda'] = torch.cuda.get_rng_state_all()
        
        # Save to file using pickle for complex state objects
        random_states_path = os.path.join(checkpoint_dir, "random_states.pkl")
        with open(random_states_path, 'wb') as f:
            pickle.dump(random_states, f)
        
        server.logger.info(f"Random states saved to {random_states_path}")
    except Exception as e:
        server.logger.error(f"Failed to save random states: {e}")


def load_server_state(server) -> bool:
    """
    Load server state from checkpoint.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    server_state_path = os.path.join(checkpoint_dir, "server_state.pth")
    
    if not os.path.exists(server_state_path):
        server.logger.warning(f"Server state file not found: {server_state_path}")
        return False
    
    try:
        server_state = torch.load(server_state_path, map_location=server.config["model_device"])
        server.logger.info(f"Server state loaded from {server_state_path}")
        
        # Load server state variables
        server.model_version_parameters = server_state['model_version_parameters']
        server.current_time = server_state['current_time']
        server.global_model_version = server_state['global_model_version']
        server.next_event_id = server_state['next_event_id']
        server.rounds_completed = server_state['rounds_completed']
        
        # Load the current version parameters to the global model
        if server.model_version_parameters and server.global_model_version in server.model_version_parameters:
            from utils.model_utils import load_model_params
            server.global_model = load_model_params(
                server.global_model, 
                server.model_version_parameters[server.global_model_version], 
                server.config["model_device"]
            )
            server.logger.info(f"Loaded model version {server.global_model_version} to global model")
        else:
            server.logger.error(f"Model version {server.global_model_version} not found in saved parameters")
            return False
        
        return True
    except Exception as e:
        server.logger.error(f"Failed to load server state: {e}")
        return False

def load_metrics(server) -> bool:
    """
    Load metrics from checkpoint.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    metrics_path = os.path.join(checkpoint_dir, "metrics_ckpt.json")
    
    if not os.path.exists(metrics_path):
        server.logger.warning(f"Metrics checkpoint file not found: {metrics_path}")
        return False
    
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        server.test_metrics = metrics_data['test_metrics']
        server.train_metrics = metrics_data['train_metrics']
        
        server.logger.info(f"Metrics loaded from {metrics_path}")
        return True
    except Exception as e:
        server.logger.error(f"Failed to load metrics: {e}")
        return False

def load_event_queue(server) -> bool:
    """
    Load event queue from checkpoint.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    events_json_path = os.path.join(checkpoint_dir, "event_queue.json")
    event_data_path = os.path.join(checkpoint_dir, "event_queue_data.pth")
    
    if not os.path.exists(events_json_path):
        server.logger.warning(f"Event queue file not found: {events_json_path}")
        return False
    
    try:
        # Load simple event attributes from JSON
        with open(events_json_path, 'r') as f:
            events_list = json.load(f)
        
        # Load complex event data if it exists
        event_data_dict = {}
        if os.path.exists(event_data_path):
            event_data_dict = torch.load(event_data_path, map_location=server.config["model_device"])
        
        # Reconstruct events
        from utils.event_utils import Event, EventType, EventQueue
        
        # Create new event queue
        server.event_queue = EventQueue()
        
        for event_simple in events_list:
            # Find client object
            client = None
            if event_simple['client_id'] is not None:
                client = server.clients[event_simple['client_id']]
            
            # Get event data (might not exist for all events)
            event_data = event_data_dict.get(event_simple['event_id'], None)
            
            # Create event with explicit duration to avoid delay generation
            event = Event(
                event_type=getattr(EventType, event_simple['event_type']),
                start_time=event_simple['start_time'],
                duration=event_simple['duration'],  # Use saved duration
                client=client,
                data=event_data,
                config=server.config,
                event_id=event_simple['event_id']
            )
            
            # Manually set finish_time to avoid recalculation
            event.finish_time = event_simple['finish_time']
            
            # Add event to queue and client tracking
            server.event_queue.add_event(event)
            if client:
                client.add_event(event.event_id, event.event_type)
        
        server.logger.info(f"Event queue loaded with {len(events_list)} events")
        return True
    except Exception as e:
        server.logger.error(f"Failed to load event queue: {e}")
        return False

def load_update_buffer(server) -> bool:
    """
    Load update buffer from checkpoint.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    update_buffer_path = os.path.join(checkpoint_dir, "update_buffer.pth")
    
    if not os.path.exists(update_buffer_path):
        server.logger.warning(f"Update buffer file not found: {update_buffer_path}")
        return False
    
    try:
        server.update_buffer = torch.load(update_buffer_path, map_location=server.config["model_device"])
        server.logger.info(f"Update buffer loaded with {len(server.update_buffer)} updates")
        return True
    except Exception as e:
        server.logger.error(f"Failed to load update buffer: {e}")
        return False

def load_kd_buffer(server) -> bool:
    """
    Load KD buffer from checkpoint.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    kd_buffer_path = os.path.join(checkpoint_dir, "kd_buffer.pth")
    
    if not os.path.exists(kd_buffer_path):
        server.logger.warning(f"KD buffer file not found: {kd_buffer_path}")
        return False
    
    try:
        server.kd_buffer = torch.load(kd_buffer_path, map_location=server.config["model_device"])
        server.logger.info(f"KD buffer loaded with {len(server.kd_buffer)} updates")    
        return True
    except Exception as e:
        server.logger.error(f"Failed to load KD buffer: {e}")
        return False

def load_random_states(server) -> bool:
    """
    Load random number generator states for all libraries.
    
    Args:
        server: The server object
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    checkpoint_dir = os.path.join(server.config["out_dir"], "checkpoint_exp_state")
    random_states_path = os.path.join(checkpoint_dir, "random_states.pkl")
    
    if not os.path.exists(random_states_path):
        server.logger.warning(f"Random states file not found: {random_states_path}")
        return False
    
    try:
        # Load random states
        with open(random_states_path, 'rb') as f:
            random_states = pickle.load(f)
        
        # Restore Python's built-in random state
        if 'python_random' in random_states:
            random.setstate(random_states['python_random'])
        
        # Restore NumPy random state
        if 'numpy_random' in random_states:
            np.random.set_state(random_states['numpy_random'])
        
        # Restore PyTorch CPU random state
        if 'torch_cpu' in random_states:
            torch.set_rng_state(random_states['torch_cpu'])
        
        # Restore PyTorch CUDA random state if available and was saved
        if torch.cuda.is_available() and 'torch_cuda' in random_states:
            torch.cuda.set_rng_state_all(random_states['torch_cuda'])
        
        server.logger.info(f"Random states loaded from {random_states_path}")
        return True
    except Exception as e:
        server.logger.error(f"Failed to load random states: {e}")
        return False


def save_dfkd_state(server, checkpoint_dir) -> None:
    """
    Save DFKD synthesizer state to checkpoint files.
    Handles both image and text datasets.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to save checkpoint files
    """
    # Only save if DFKD is enabled and synthesizer exists
    if (server.config['kd_augmentation'] != 'dfkd'):
        return
    
    assert server.synthesizer is not None, "DFKD synthesizer is not initialized but trying to save DFKD state"

    dfkd_state = {}
    
    # Check if this is a text dataset
    is_text_dataset = server.config['dataset'].lower() == 'news'
    dfkd_state['is_text_dataset'] = is_text_dataset
    
    if is_text_dataset:
        # Text dataset: save prompt vectors instead of generator
        dfkd_state['prompt_vecs'] = server.synthesizer.prompt_vecs.detach().cpu()
        # Note: No generator for text datasets
    else:
        # Image dataset: save generator model state
        dfkd_state['generator_state_dict'] = server.generator.state_dict()
        dfkd_state['generator_params'] = server.generator.params  # (nz, ngf, img_size, nc, num_classes)
    
    # Save synthesizer state (common for both image and text)
    dfkd_state['synthesizer_ep'] = server.synthesizer.ep
    
    # Text datasets don't have prev_z
    if not is_text_dataset:
        dfkd_state['synthesizer_prev_z'] = server.synthesizer.prev_z
    
    # Save momentum manager state
    dfkd_state['momentum_manager_state'] = server.synthesizer.momentum_manager.momentum_state
    
    # Save meta optimizer state
    dfkd_state['meta_optimizer_state_dict'] = server.synthesizer.meta_optimizer.state_dict()
    
    # Save dataset from data pool (this will be saved separately as a .pth file)
    data_pool_path = os.path.join(checkpoint_dir, "dfkd_dataset.pth")
    server.synthesizer.data_pool.dataset.save_to_file(data_pool_path)
    dfkd_state['data_pool_path'] = data_pool_path
    dfkd_state['data_pool_iteration'] = server.synthesizer.data_pool.iteration

    # Save loss tracking windows (common for both image and text)
    dfkd_state['bn_loss_window'] = server.synthesizer.bn_loss_window
    dfkd_state['oh_loss_window'] = server.synthesizer.oh_loss_window
    dfkd_state['adv_loss_window'] = server.synthesizer.adv_loss_window
    dfkd_state['kl_uniform_loss_window'] = server.synthesizer.kl_uniform_loss_window
    dfkd_state['diversity_loss_window'] = server.synthesizer.diversity_loss_window
    dfkd_state['kd_loss_window'] = server.synthesizer.kd_loss_window
    
    # Save main DFKD state atomically
    dfkd_state_path = os.path.join(checkpoint_dir, "dfkd_state.pth")
    atomic_torch_save(dfkd_state, dfkd_state_path, server.logger)
    
    dataset_type = "text" if is_text_dataset else "image"
    server.logger.info(f"DFKD state saved ({dataset_type} dataset) with {len(server.synthesizer.data_pool.dataset)} synthetic samples")




def load_dfkd_state(server, checkpoint_dir) -> bool:
    """
    Load DFKD synthesizer state from checkpoint files.
    Handles both image and text datasets.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to load checkpoint files from
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    # Only load if DFKD is enabled and synthesizer exists
    if (server.config['kd_augmentation'] != 'dfkd'):
        return True  # Return True since this is not an error
    
    # Check if synthesizer is properly initialized
    if not hasattr(server, 'synthesizer') or server.synthesizer is None:
        server.logger.error("DFKD synthesizer not initialized but trying to load DFKD state")
        return False
    
    dfkd_state_path = os.path.join(checkpoint_dir, "dfkd_state.pth")
    
    if not os.path.exists(dfkd_state_path):
        server.logger.warning(f"DFKD state file not found: {dfkd_state_path}")
        return False
    
    # Load DFKD state
    dfkd_state = torch.load(dfkd_state_path, map_location=server.config["model_device"])
    
    # Check if this is a text dataset, default to False if not found
    is_text_dataset = dfkd_state.get('is_text_dataset', False)
    
    if is_text_dataset:
        # Text dataset: restore prompt vectors
        if 'prompt_vecs' in dfkd_state:
            server.synthesizer.prompt_vecs = dfkd_state['prompt_vecs'].to(server.synthesizer.device).requires_grad_(True)
            # Update meta optimizer to track the loaded prompt vectors
            server.synthesizer.meta_optimizer = torch.optim.Adam(
                [server.synthesizer.prompt_vecs],
                server.synthesizer.lr_g * server.synthesizer.iterations,
                betas=[0.5, 0.999]
            )
        else:
            server.logger.warning("Prompt vectors not found in DFKD state for text dataset")
            return False
    else:
        # Image dataset: restore generator model state
        if 'generator_state_dict' in dfkd_state and server.generator is not None:
            server.generator.load_state_dict(dfkd_state['generator_state_dict'])
            # Ensure generator is on the correct device (same as synthesizer device)
            server.generator = server.generator.to(server.synthesizer.device)
        else:
            server.logger.warning("Generator state not found in DFKD state for image dataset")
            return False
    
    # Restore synthesizer state (common for both image and text)
    if 'synthesizer_ep' in dfkd_state:
        server.synthesizer.ep = dfkd_state['synthesizer_ep']
    else:
        server.logger.warning("Synthesizer ep not found in DFKD state")
        return False
    
    # Only restore prev_z for image datasets
    if not is_text_dataset:
        if 'synthesizer_prev_z' in dfkd_state:
            server.synthesizer.prev_z = dfkd_state['synthesizer_prev_z']
        else:
            server.logger.warning("Synthesizer prev_z not found in DFKD state")
            return False
    
    # Restore momentum manager state
    if 'momentum_manager_state' in dfkd_state:
        server.synthesizer.momentum_manager.momentum_state = dfkd_state['momentum_manager_state']
    else:
        server.logger.warning("Momentum manager state not found in DFKD state")
        return False
    
    # Restore meta optimizer state (after creating new optimizer for text datasets)
    if 'meta_optimizer_state_dict' in dfkd_state:
        server.synthesizer.meta_optimizer.load_state_dict(dfkd_state['meta_optimizer_state_dict'])
    else:
        server.logger.warning("Meta optimizer state not found in DFKD state")
        return False
    
    # Load dataset from data pool
    if 'data_pool_path' in dfkd_state and os.path.exists(dfkd_state['data_pool_path']):
        if is_text_dataset:
            # Text dataset: use SyntheticTextDataset
            from utils.text_dfkd_utils import SyntheticTextDataset
            kd_max_length = server.config['dfkd_settings'].get('kd_max_length', 256)
            server.synthesizer.data_pool.dataset = SyntheticTextDataset.load_from_file(
                dfkd_state['data_pool_path'], 
                kd_max_length=kd_max_length
            )
        else:
            # Image dataset: use InMemorySyntheticDataset
            from utils.dfkd_utils import InMemorySyntheticDataset
            dataset_name = server.config['dataset']
            server.synthesizer.data_pool.dataset = InMemorySyntheticDataset.load_from_file(
                dfkd_state['data_pool_path'], 
                dataset_name=dataset_name
            )
        
        server.logger.info(f"DFKD dataset loaded with {len(server.synthesizer.data_pool.dataset)} samples")
    else:
        server.logger.warning("Data pool path not found in DFKD state")
        return False
        
    # Restore iteration counter
    if 'data_pool_iteration' in dfkd_state:
        server.synthesizer.data_pool.iteration = dfkd_state['data_pool_iteration']
    else:
        server.logger.warning("Data pool iteration not found in DFKD state")
        return False
    
    # Restore loss tracking windows (common for both image and text)
    if 'bn_loss_window' in dfkd_state:
        server.synthesizer.bn_loss_window = dfkd_state['bn_loss_window']
    else:
        server.logger.warning("BN loss window not found in DFKD state")
    
    if 'oh_loss_window' in dfkd_state:
        server.synthesizer.oh_loss_window = dfkd_state['oh_loss_window']
    else:
        server.logger.warning("OH loss window not found in DFKD state")
        
    if 'adv_loss_window' in dfkd_state:
        server.synthesizer.adv_loss_window = dfkd_state['adv_loss_window']
    else:
        server.logger.warning("Adversarial loss window not found in DFKD state")
        
    if 'kl_uniform_loss_window' in dfkd_state:
        server.synthesizer.kl_uniform_loss_window = dfkd_state['kl_uniform_loss_window']
    else:
        server.logger.warning("KL uniform loss window not found in DFKD state")
        
    if 'diversity_loss_window' in dfkd_state:
        server.synthesizer.diversity_loss_window = dfkd_state['diversity_loss_window']
    else:
        server.logger.warning("Diversity loss window not found in DFKD state")
        
    if 'kd_loss_window' in dfkd_state:
        server.synthesizer.kd_loss_window = dfkd_state['kd_loss_window']
    else:
        server.logger.warning("KD loss window not found in DFKD state")
    
    dataset_type = "text" if is_text_dataset else "image"
    server.logger.info(f"DFKD state loaded ({dataset_type} dataset) with {len(server.synthesizer.data_pool.dataset)} synthetic samples")
    return True


def save_estimation_state(server, checkpoint_dir) -> None:
    """
    Save distribution estimation state to checkpoint files.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to save checkpoint files
    """
    try:
        estimation_state = {}
        
        # Save estimation parameters (must exist for ablation experiments)
        estimation_state['estimation_rounds'] = server.estimation_rounds
        estimation_state['estimation_temperature'] = server.estimation_temperature
        estimation_state['num_random_inputs'] = server.num_random_inputs
        estimation_state['how_many_dist_est_update_later'] = server.how_many_dist_est_update_later
        
        # Save estimation results and flags (must exist for ablation experiments)
        estimation_state['estimated_client_distributions'] = server.estimated_client_distributions
        estimation_state['ground_truth_client_distributions'] = server.ground_truth_client_distributions
        estimation_state['client_update_counts'] = server.client_update_counts
        estimation_state['estimation_completed'] = server.estimation_completed
        estimation_state['final_estimation_error_reported'] = server.final_estimation_error_reported
        
        # Save client distribution states (must exist for ablation experiments)
        client_distribution_states = {}
        for client in server.clients:
            client_distribution_states[client.client_id] = {
                'ground_truth_class_proportions': client.ground_truth_class_proportions,
                'class_proportions': client.class_proportions,
                'using_estimated_distribution': client.using_estimated_distribution
            }
        estimation_state['client_distribution_states'] = client_distribution_states
        
        # Save to file atomically
        estimation_state_path = os.path.join(checkpoint_dir, "estimation_state.pth")
        atomic_torch_save(estimation_state, estimation_state_path, server.logger)
        
        server.logger.info(f"Estimation state saved with {len(server.estimated_client_distributions)} client distributions")
        
    except Exception as e:
        server.logger.error(f"Failed to save estimation state: {e}")


def load_estimation_state(server, checkpoint_dir) -> bool:
    """
    Load distribution estimation state from checkpoint files.
    
    Args:
        server: The server object
        checkpoint_dir: Directory to load checkpoint files from
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    estimation_state_path = os.path.join(checkpoint_dir, "estimation_state.pth")
    
    if not os.path.exists(estimation_state_path):
        server.logger.warning(f"Estimation state file not found: {estimation_state_path}")
        return False
    
    try:
        estimation_state = torch.load(estimation_state_path, map_location=server.config["model_device"])
        
        # Restore estimation parameters
        server.estimation_rounds = estimation_state['estimation_rounds']
        server.estimation_temperature = estimation_state['estimation_temperature']
        server.num_random_inputs = estimation_state['num_random_inputs']
        server.how_many_dist_est_update_later = estimation_state['how_many_dist_est_update_later']
        
        # Restore estimation results and flags
        server.estimated_client_distributions = estimation_state['estimated_client_distributions']
        server.ground_truth_client_distributions = estimation_state['ground_truth_client_distributions']
        server.client_update_counts = estimation_state['client_update_counts']
        server.estimation_completed = estimation_state['estimation_completed']
        server.final_estimation_error_reported = estimation_state['final_estimation_error_reported']
        
        # Restore client distribution states
        client_distribution_states = estimation_state['client_distribution_states']
        for client in server.clients:
            client_id = client.client_id
            if client_id in client_distribution_states:
                client_state = client_distribution_states[client_id]
                client.ground_truth_class_proportions = client_state['ground_truth_class_proportions']
                client.class_proportions = client_state['class_proportions']
                client.using_estimated_distribution = client_state['using_estimated_distribution']
            else:
                # For ablation experiments, all client states must be present
                raise RuntimeError(f"Client {client_id} distribution state not found in checkpoint - cannot resume experiment")
        
        server.logger.info(f"Estimation state loaded with {len(server.estimated_client_distributions)} client distributions")
        server.logger.info(f"Estimation completed: {server.estimation_completed}")
        server.logger.info(f"Final error reported: {server.final_estimation_error_reported}")
        
        # Validate consistency
        num_clients = len(server.clients)
        if len(server.estimated_client_distributions) != num_clients:
            server.logger.warning(f"Mismatch in client distributions: expected {num_clients}, got {len(server.estimated_client_distributions)}")
        
        if len(server.client_update_counts) != num_clients:
            server.logger.warning(f"Mismatch in client update counts: expected {num_clients}, got {len(server.client_update_counts)}")
        
        return True
        
    except Exception as e:
        server.logger.error(f"Failed to load estimation state: {e}")
        # For ablation experiments, estimation state is critical - don't provide fallbacks
        raise RuntimeError(f"Failed to resume experiment: estimation state could not be loaded: {e}")


    
