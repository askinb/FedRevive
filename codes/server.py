import os
import numpy as np
import time
from typing import Dict, List, Any, Union, Optional
import copy
import json

from utils.logging_utils import *
from utils.data_utils import *
from utils.model_utils import *
from utils.model_utils import ServerAMSGrad, average_updates, ServerFedAC, weighted_average_updates, compute_cosine_similarity_param_dicts
from utils.event_utils import *
from utils.exp_state_utils import *
from utils.kd_utils import *
# Import DFKD components
from utils.dfkd_utils import *
from utils.text_dfkd_utils import perform_text_dfkd_with_buffer
from utils.text_dfkd_utils import FastTextMetaSynthesizer
from config import Config
import random
import torch
from client import Client
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Server:
    """
    Server class for asynchronous federated learning.
    """
    
    def __init__(self, config: Union[str, Dict[str, Any], Config, None] = None):
        """
        Initialize the server.
        
        Args:
            config: Configuration for the experiment. Can be one of:
                - Path to a JSON file (str)
                - Dictionary containing configuration (Dict)
                - Config object
                - None (use default configuration)
        """
        self.config = Config(config)
        # Handle resume from existing experiment
        if bool(self.config["continue_exp_from"]) and check_if_resumable(self.config["continue_exp_from"]):
            self.config.load_from_file(os.path.join(self.config["continue_exp_from"], "config.json"))      
            self.is_resuming = True
            self.resume_from_dir = self.config["continue_exp_from"]
        else:
            self.is_resuming = False

        # Setup logger
        self.logger = setup_logger("Server", self.config["out_dir"], resume=self.is_resuming)
        if self.is_resuming:
            self.logger.info(f"\n"+("-"*40)+f"Resuming experiment from {self.config['continue_exp_from']}"+("-"*40)+"\n")
        else:
            self.logger.info(f"\n"+("-"*40)+"Starting new experiment"+("-"*40)+"\n")

        os.makedirs(self.config["out_dir"], exist_ok=True)

        # Initialize wall-clock time tracking
        self.wall_clock_start_time = time.time()
        self.last_checkpoint_time = self.wall_clock_start_time

        # Initialize wandb
        self.set_wandb()

        self.initialize_metrics()

        if torch.cuda.is_available():
            torch.cuda.set_device(self.config["gpu_id"])
            self.logger.info(f"Using GPU {self.config['gpu_id']} if needed")
        
        self.logger.info("Server initialized")
        self.config.save_to_file(os.path.join(self.config["out_dir"], "config.json"))

    def set_wandb(self):
        if self.config["wandb_flag"]:
            try:
                self.wandb = __import__('wandb')
                # Use the new wandb resume functionality
                self.wandb_run, wandb_resumed = setup_wandb_resume(self.config, self.config["out_dir"], self.is_resuming)
                if wandb_resumed:
                    self.logger.info("Resumed existing WandB run")
                else:
                    self.logger.info("Started new WandB run")
            except ImportError:
                self.logger.warning("WandB is not installed. Flag is set to False.")
                self.config["wandb_flag"] = False
                return

    def load_data(self):
        """
        Load data for the federated learning experiment.
        """
        self.logger.info(f"Loading dataset: {self.config['dataset']}")
        self.datasets_dict = load_data(self.config["dataset"], config=self.config)
        
        # Extract test data for server evaluation
        if 'test_dataset' in self.datasets_dict:
            self.test_data = self.datasets_dict['test_dataset']
            # Create dataloader for the test data
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=self.config['eval_batch_size'],
                shuffle=False
            )

        # Extract KD data for simple knowledge distillation
        if 'kd_dataset' in self.datasets_dict:
            self.kd_data = self.datasets_dict['kd_dataset']
            # Create KD dataloader once for efficiency
            self.kd_dataloader = torch.utils.data.DataLoader(
                self.kd_data, 
                batch_size=self.config['simple_kd_settings']['batch_size'],
                shuffle=True
            )
            self.logger.info(f"Simple KD dataset loaded with {len(self.kd_data)} samples")
        else:
            self.kd_data = None
            self.kd_dataloader = None
            self.logger.warning("No simple KD dataset available")
        
        self.logger.info(f"Dataset loaded successfully")
    
    def create_clients(self):
        """
        Create clients for the federated learning experiment.
        """
        num_clients = self.config["num_clients"]
        self.logger.info(f"Creating {num_clients} clients")
        
        # Create client objects
        self.clients = []
        for i in range(num_clients):
            client = Client(client_id=i, config=self.config)
            
            client.set_delay_generators(
                train_config=self.config['localtrain_delay'],
                download_config=self.config['download_delay'],
                upload_config=self.config['upload_delay']
            )
            
            self.clients.append(client)
        
        self.logger.info(f"Created {num_clients} clients")
    
    def distribute_data_to_clients(self):
        """
        Distribute data among the clients according to the distribution configuration.
        """
        if not self.datasets_dict or 'train_dataset' not in self.datasets_dict:
            self.logger.warning("No training data available to distribute")
            return
        
        train_dataset = self.datasets_dict['train_dataset']
        num_clients = len(self.clients)
        distribution = self.config['data_distribution']
        
        self.logger.info(f"Distributing data to {num_clients} clients using {distribution} distribution")
        
        # Split data for clients
        client_datasets = split_data_for_clients(
            dataset=train_dataset,
            num_clients=num_clients,
            distribution=distribution,
            client_data_size=self.config['client_data_size'],
            dirichlet_alpha=self.config.get('dirichlet_alpha',None)
        )
        
        # Assign data to clients
        local_batch_size = self.config['local_batch_size']
        for i, client in enumerate(self.clients):
            if i < len(client_datasets):
                client.set_dataset(client_datasets[i], batch_size=local_batch_size)
        self.logger.info(f"Data distributed to {len(self.clients)} clients successfully")
    
    def train(self):
        """
        Run the asynchronous federated learning training process.
        """
        self.logger.info(f"{'Resuming' if self.is_resuming else 'Starting'} FL training with algorithm {self.config['algorithm']}")
        
        # Main event loop
        while not self.event_queue.is_empty():
            # Get the next event from the queue
            event = self.event_queue.get_next_event()
            
            # Update current time
            self.current_time = event.finish_time
            
            self.handle_event(event)

            # Check if it's time to save checkpoint
            if self.config["save_exp_state_interval"] > 0 and should_save_checkpoint(self):
                save_experiment_state(self)
                self.last_checkpoint_time = time.time()
                        
            if self.check_stopping_conditions(): 
                # Run a final evaluation
                self.evaluate_global_model()
                if self.config["wandb_flag"]:
                    self.wandb.finish()
                break        
        
        self.logger.info("Training complete")
        
        # Save final metrics and model
        self.save_metrics()
        self.save_model_checkpoint()

        if self.config["wandb_flag"]:
            self.wandb.finish()
        
        return {
            'rounds_completed': self.rounds_completed,
            'total_time': self.current_time,
            'test_accuracy': self.test_metrics['accuracies'][-1] if self.test_metrics['accuracies'] else 0.0,
            'test_loss': self.test_metrics['losses'][-1] if self.test_metrics['losses'] else 0.0
        }

    
    def schedule_initial_events(self):
        """
        Schedule initial download events for all clients.
        """
        if self.config['algorithm'] in ['async_fl', 'sync_fl', 'fedfa']:
            # Randomly select number of active jobs-many clients without replacement
            selected_clients = np.random.choice(self.clients, size=self.config[self.config['algorithm']+'_settings']['nb_of_active_jobs'], replace=False)
            model_version = self.global_model_version
            for client in selected_clients:
                # Create a download event with 0 delay (starts immediately)
                event = self.create_event(
                    event_type=EventType.DOWNLOAD_JOB,
                    start_time=self.current_time,
                    duration=None,
                    client=client,
                    data={"model_version": model_version}
                )
            
    def handle_download_event(self, event):
        """
        Handle a download event.
        Client downloads the global model.
        
        Args:
            event (Event): The download event to handle
        """ 
        # Schedule a local training event
        train_event = self.create_event(
            event_type=EventType.LOCAL_TRAIN,
            start_time=self.current_time,
            duration=None,
            client=event.client,
            data=event.data
        )

    
    def handle_train_event(self, event):
        """
        Handle a local training event.
        Client trains the model on its local data.
        
        Args:
            event (Event): The training event to handle
        """
        client = event.client
        
        # Use the reusable local training model with lightweight reset for deterministic behavior
        reset_model_state(self.local_training_model, reset_norm_stats=self.local_training_model.has_running_stats)
        
        # Load the model parameters for this training event
        local_model = load_model_params(self.local_training_model, self.model_version_parameters[event.data["model_version"]], self.config["train_device"])

        # Client trains the model
        train_result_dict = client.local_train(local_model, config = event.config)
        model_params = get_model_params(train_result_dict["final_local_model"], self.config["model_device"])
        train_loss = train_result_dict["train_loss"]
        train_accuracy = train_result_dict["train_accuracy"]

        # Track training metrics
        self.train_metrics['rounds'].append(self.rounds_completed)
        self.train_metrics['times'].append(self.current_time)
        self.train_metrics['client_ids'].append(client.client_id)
        self.train_metrics['losses'].append(train_loss)
        self.train_metrics['accuracies'].append(train_accuracy)
        
        # Log to wandb if enabled
        if self.config["wandb_flag"]:
            self.wandb.log({
                'train/loss': train_loss,
                'train/accuracy': train_accuracy,
                'train/client_id': client.client_id,
                'round': self.rounds_completed,
                'time': self.current_time
            })

        if self.config["server_lr"] >= 0:
            update = get_model_diff(model_params, self.model_version_parameters[event.data["model_version"]])
        else:
            update = model_params

        # Schedule an upload event
        upload_event = self.create_event(
            event_type=EventType.UPLOAD_UPDATE,
            start_time=self.current_time,
            duration=None,
            client=client,
            data={"update": update, "train_loss": train_loss, "train_accuracy": train_accuracy, "model_version": event.data["model_version"]}
        )

    
    def handle_upload_event(self, event):
        """
        Handle an upload event.
        Client uploads its model update to the server.
        
        Args:
            event (Event): The upload event to handle
        """ 
        client = event.client

        original_update = event.data["update"]
        
        if self.config['kd_augmentation'] is not None:
            # Calculate staleness for this update
            staleness = self.global_model_version - event.data["model_version"]
            
            # Calculate adaptive beta
            adaptive_beta = calculate_adaptive_beta(self.config, staleness)
            
            # Update KD buffer with the original update first
            self.update_kd_buffer(original_update, event.data["model_version"], client.client_id)
        
        elif self.config['async_downweighting']:
            # Calculate staleness for this update
            staleness = self.global_model_version - event.data["model_version"]
            
            # Calculate adaptive beta
            adaptive_beta = calculate_adaptive_beta(self.config, staleness)
            
        # Apply KD using all models in the updated buffer if enabled
        if self.config['kd_augmentation'] == 'simple_kd' and adaptive_beta>0:
            assert self.kd_dataloader is not None and len(self.kd_buffer) > 0, "KD loader not initialized or KD buffer is empty"

            # Perform KD using all models in the buffer
            kd_update = perform_kd_with_buffer(
                self.kd_buffer,
                self.kd_dataloader, 
                self.config,
                self.model_version_parameters[self.global_model_version],  # Use current global model as student base
                self.clients,
                teacher_models=self.kd_teacher_models[:len(self.kd_buffer)],
                student_model=self.kd_student_model,
                client_kd_dataloaders=self.client_kd_dataloaders
            )
            
            # Combine original update with KD result using beta
            processed_update = combine_kd_with_original(original_update, kd_update, adaptive_beta)
        elif (self.config['kd_augmentation'] == 'dfkd' and adaptive_beta > 0):
            assert len(self.kd_buffer) > 0, "KD buffer is empty"

            if self.config['dfkd_settings']['freeze_rounds'] > self.rounds_completed:
                self.logger.info(f"Skipping DFKD because freeze rounds ({self.config['dfkd_settings']['freeze_rounds']}) > rounds completed ({self.rounds_completed})")
                processed_update = original_update
            else:
                # Determine if this is a synthesis round based on generate_freq
                generate_freq = self.config['dfkd_settings'].get('generate_freq', 1)
                freeze_rounds = self.config['dfkd_settings']['freeze_rounds']
                rounds_after_freeze = self.rounds_completed - freeze_rounds
                synthesize_this_round = (rounds_after_freeze % generate_freq == 0)
                
                # Perform DFKD using all models in the buffer
                # Use text-specific DFKD for news dataset, otherwise use image DFKD
                if self.config['dataset'].lower() == 'news':
                    kd_update = perform_text_dfkd_with_buffer(
                        self.rounds_completed,
                        self.kd_buffer,
                        self.synthesizer,
                        self.config,
                        self.model_version_parameters[self.global_model_version],  # Use current global model as student base
                        self.clients,
                        teacher_models=self.kd_teacher_models[:len(self.kd_buffer)],
                        student_model=self.kd_student_model,
                        wandb_run=self.wandb if self.config["wandb_flag"] else None,  # Pass wandb object for logging
                        synthesize_this_round=synthesize_this_round  # Control synthesis frequency
                    )
                else:
                    kd_update = perform_dfkd_with_buffer(
                        self.rounds_completed,
                        self.kd_buffer,
                        self.synthesizer,
                        self.config,
                        self.model_version_parameters[self.global_model_version],  # Use current global model as student base
                        self.clients,
                        teacher_models=self.kd_teacher_models[:len(self.kd_buffer)],
                        student_model=self.kd_student_model,
                        wandb_run=self.wandb if self.config["wandb_flag"] else None,  # Pass wandb object for logging
                        synthesize_this_round=synthesize_this_round  # Control synthesis frequency
                    )
                
                # Combine original update with DFKD result using beta
                processed_update = combine_kd_with_original(original_update, kd_update, adaptive_beta)

        elif self.config['async_downweighting']:
            processed_update = downweight_update_based_on_staleness(original_update, adaptive_beta)
            
        else:
            processed_update = original_update


        # FedAC: compute temporal gradient cosine similarity score at upload time.
        # We do this here (not at aggregation) because model_version_parameters[x_tau_version]
        # is guaranteed to still be in memory (cleanup hasn't run yet), and the global model
        # does not change between upload events within a single buffer window.
        if self.config.get('server_optimizer', 'sgd') == 'fedac':
            x_tau_version = event.data["model_version"]
            x_tau = self.model_version_parameters.get(x_tau_version)
            x_current = self.model_version_parameters[self.global_model_version]
            if x_tau is not None and x_tau_version != self.global_model_version:
                # global_change = x^{t-1} - x^tau (direction server moved while client was training)
                global_change = {name: x_current[name] - x_tau[name] for name in x_current}
                cos_score = compute_cosine_similarity_param_dicts(global_change, original_update)
            else:
                # No staleness (x_tau == x_current) → cosine sim undefined → 0.0 (uniform weight)
                cos_score = 0.0
            self.fedac_scores.append(cos_score)
            self.logger.debug(f"FedAC: client {client.client_id} cosine score={cos_score:.4f} (staleness={self.global_model_version - x_tau_version})")

        # Add the processed update to the update buffer
        self.update_buffer.append((client.client_id, event.data["model_version"], processed_update))
        
        if self.config['algorithm'] == 'async_fl':
            # If buffer is full, schedule an aggregation event, and wait for next training assignment
            if len(self.update_buffer) >= self.config[self.config["algorithm"]+"_settings"]["buffer_size"]:
                aggregate_event = self.create_event(
                    event_type=EventType.AGGREGATE,
                    start_time=self.current_time,
                    duration=0,  # Assume aggregation happens instantly for now
                    client=None,
                    data=None
                )
            else: # directly assign a new client
                # Randomly select a client without replacement and assign an download event   
                inactive_clients = self.get_inactive_clients()
                if inactive_clients:
                    client = np.random.choice(inactive_clients)
                    model_version = self.global_model_version
                    event = self.create_event(
                        event_type=EventType.DOWNLOAD_JOB,
                        start_time=self.current_time,
                        duration=None,
                        client=client,
                        data={"model_version": model_version}
                    )
        elif self.config['algorithm'] == 'sync_fl':
            if len(self.update_buffer) == self.config[self.config['algorithm']+'_settings']['nb_of_active_jobs']:
                aggregate_event = self.create_event(
                    event_type=EventType.AGGREGATE,
                    start_time=self.current_time,
                    duration=0,  # Assume aggregation happens instantly for now
                    client=None,
                    data=None
                )

        elif self.config['algorithm'] == 'fedfa':
            # FedFa: update_buffer is a persistent FIFO sliding window of capacity K=window_size.
            # On every upload:
            #   1. Evict the oldest entry if the window has overflowed.
            #   2. Trigger an AGGREGATE event when the window is full (happens every upload
            #      once the warm-up period of K uploads is done).
            #   3. Always schedule a new client download immediately — FedFa is fully
            #      asynchronous and never waits for an aggregation to complete first.
            window_size = self.config['fedfa_settings']['window_size']

            # Evict oldest entry so window never exceeds K
            if len(self.update_buffer) > window_size:
                self.update_buffer.pop(0)

            # Schedule aggregation whenever window is at full capacity
            if len(self.update_buffer) == window_size:
                self.create_event(
                    event_type=EventType.AGGREGATE,
                    start_time=self.current_time,
                    duration=0,
                    client=None,
                    data=None
                )

            # Immediately assign a new download to keep nb_of_active_jobs clients busy
            inactive_clients = self.get_inactive_clients()
            if inactive_clients:
                new_client = np.random.choice(inactive_clients)
                self.create_event(
                    event_type=EventType.DOWNLOAD_JOB,
                    start_time=self.current_time,
                    duration=None,
                    client=new_client,
                    data={"model_version": self.global_model_version}
                )

    def handle_aggregate_event(self, event):
        """
        Handle an aggregation event.
        Server aggregates model updates from the buffer.
        
        Args:
            event (Event): The aggregation event to handle
        """ 
        
        if self.config['algorithm'] in ['async_fl', 'sync_fl']:
            if self.config['algorithm'] == 'async_fl':  
                if len(self.update_buffer) != self.config[self.config["algorithm"]+"_settings"]["buffer_size"]:
                    self.logger.warning(f"Aggregation event triggered but buffer size is {len(self.update_buffer)} instead of {self.config[self.config['algorithm']+'_settings']['buffer_size']}")

            # Log client updates and staleness information
            client_ids = [update[0] for update in self.update_buffer]
            update_versions = [update[1] for update in self.update_buffer]
            staleness = [self.global_model_version - version for version in update_versions]
            self.logger.info(f"Round {self.rounds_completed}, Time {self.current_time:.2f} | Clients in buffer: {client_ids} with staleness: {staleness}, wall clock time: {time.time() - self.prev_round_time:.2f}s")

            # Aggregate updates
            current_model_params = self.model_version_parameters[self.global_model_version]

            if self.config.get('server_optimizer', 'sgd') == 'amsgrad' and self.server_amsgrad is not None:
                # FADAS: AMSGrad server update with delay-adaptive LR
                delta_avg = average_updates([update[-1] for update in self.update_buffer])

                # Compute delay-adaptive learning rate scaling
                tau_c = self.config.get('server_optimizer_settings', {}).get('tau_c', 10)
                max_staleness_val = max(staleness) if staleness else 0
                lr_lambda = 1.0 / max_staleness_val if max_staleness_val > tau_c else 1.0

                aggregated_model_params = self.server_amsgrad.step(
                    current_model_params, delta_avg, self.config['server_lr'], lr_lambda
                )

                if self.config["wandb_flag"]:
                    self.wandb.log({
                        'fadas/lr_lambda': lr_lambda,
                        'fadas/max_staleness': max_staleness_val,
                        'fadas/optimizer_step': self.server_amsgrad.t,
                        'round': self.rounds_completed,
                        'time': self.current_time
                    })

            elif self.config.get('server_optimizer', 'sgd') == 'fedac' and self.server_fedac is not None:
                # FedAC: cosine-similarity-weighted Nadam (prospective momentum) server update.
                # Client weights are based on temporal gradient alignment:
                #   r_i = cos(x^{t-1} - x^tau, g_i^tau)  [computed at upload time]
                # Weights are clipped to [0, inf) and normalized. Uniform fallback if all <= 0.
                raw_deltas = [update[-1] for update in self.update_buffer]
                n_buf = len(raw_deltas)
                r_scores = self.fedac_scores[:n_buf]

                # Clip negative scores (misaligned clients get 0 weight)
                clipped = [max(0.0, r) for r in r_scores]
                total_weight = sum(clipped)

                if total_weight <= 1e-12:
                    # All clients misaligned or no staleness → uniform weights
                    weights = [1.0 / n_buf] * n_buf
                    self.logger.debug(f"FedAC round {self.rounds_completed}: all cosine scores <= 0, using uniform weights")
                else:
                    weights = [c / total_weight for c in clipped]

                g_t = weighted_average_updates(raw_deltas, weights)
                aggregated_model_params = self.server_fedac.step(
                    current_model_params, g_t, self.config['server_lr']
                )

                self.logger.info(f"FedAC round {self.rounds_completed}: r_scores={[f'{r:.3f}' for r in r_scores]}, weights={[f'{w:.3f}' for w in weights]}, step={self.server_fedac.t}")
                self.fedac_scores = []  # clear scores for next buffer window

                if self.config["wandb_flag"]:
                    for i, (r, w) in enumerate(zip(r_scores, weights)):
                        self.wandb.log({
                            f'fedac/r_score_{i}': r,
                            f'fedac/weight_{i}': w,
                            'round': self.rounds_completed,
                            'time': self.current_time
                        })
                    self.wandb.log({
                        'fedac/optimizer_step': self.server_fedac.t,
                        'fedac/uniform_fallback': int(total_weight <= 1e-12),
                        'round': self.rounds_completed,
                        'time': self.current_time
                    })

            else:
                # Default SGD-style update (original behavior)
                aggregated_model_params = add_update_to_params(current_model_params, [update[-1] for update in self.update_buffer], self.config['server_lr'], interpolate=self.config['server_lr'] < 0)

            self.update_buffer = []  # Clear the buffer


            aggregated_model_params = reset_stat_tracker_stats_in_state_dict(aggregated_model_params)
            # Update the global model
            self.global_model = load_model_params(self.global_model, aggregated_model_params, self.config["model_device"])
            self.global_model_version += 1
            self.model_version_parameters[self.global_model_version] = aggregated_model_params
            self.rounds_completed += 1
            
            
            if self.config["wandb_flag"]:
                for i, (client_id, staleness_val) in enumerate(zip(client_ids, staleness)):
                    self.wandb.log({
                        'aggregation/client_id': client_id,
                        'aggregation/staleness': staleness_val,
                        'round': self.rounds_completed,
                        'time': self.current_time
                    })

            # Clean up unused model versions
            self.cleanup_unused_model_versions()
            
            # Randomly select a client without replacement and assign an download event
            inactive_clients = self.get_inactive_clients()
            if inactive_clients:
                clients = np.random.choice(inactive_clients, size={'async_fl': 1, 'sync_fl': self.config[self.config['algorithm']+'_settings']['nb_of_active_jobs']}[self.config['algorithm']], replace=False)
                model_version = self.global_model_version
                for client in clients:
                    event = self.create_event(
                        event_type=EventType.DOWNLOAD_JOB,
                        start_time=self.current_time,
                        duration=None,
                        client=client,
                        data={"model_version": model_version}
                    )

        elif self.config['algorithm'] == 'fedfa':
            # FedFa: sliding-window average aggregation (FedFa-Delta variant).
            #
            # update_buffer is a persistent FIFO window of capacity K=window_size.
            # It is NOT cleared here — only the oldest entry is evicted on each new
            # upload (in handle_upload_event). The aggregation formula is:
            #   w_g^{t+1} = w_g^t + (1/K) * sum_{δ in window} δ
            # which is equivalent to add_update_to_params with server_lr as the scale.
            #
            # New clients are dispatched immediately in handle_upload_event (fully async),
            # so no client dispatch happens here.
            window_size = self.config['fedfa_settings']['window_size']

            client_ids = [update[0] for update in self.update_buffer]
            update_versions = [update[1] for update in self.update_buffer]
            staleness = [self.global_model_version - version for version in update_versions]
            self.logger.info(
                f"Round {self.rounds_completed}, Time {self.current_time:.2f} | "
                f"FedFa window ({len(self.update_buffer)}/{window_size}) | "
                f"Clients: {client_ids}, staleness: {staleness}, "
                f"wall clock: {time.time() - self.prev_round_time:.2f}s"
            )

            current_model_params = self.model_version_parameters[self.global_model_version]

            # FedFa-Delta: w_g^{t+1} = w_g^t + server_lr * (1/K) * sum_{δ in window} δ
            aggregated_model_params = add_update_to_params(
                current_model_params,
                [update[-1] for update in self.update_buffer],
                self.config['server_lr'],
                interpolate=self.config['server_lr'] < 0
            )

            # NOTE: Do NOT clear self.update_buffer — the sliding window persists.

            aggregated_model_params = reset_stat_tracker_stats_in_state_dict(aggregated_model_params)
            self.global_model = load_model_params(self.global_model, aggregated_model_params, self.config["model_device"])
            self.global_model_version += 1
            self.model_version_parameters[self.global_model_version] = aggregated_model_params
            self.rounds_completed += 1

            if self.config["wandb_flag"]:
                for i, (client_id, staleness_val) in enumerate(zip(client_ids, staleness)):
                    self.wandb.log({
                        'aggregation/client_id': client_id,
                        'aggregation/staleness': staleness_val,
                        'round': self.rounds_completed,
                        'time': self.current_time
                    })

            self.cleanup_unused_model_versions()
            # NOTE: No client dispatch here — handled in handle_upload_event for FedFa.

        # Run evaluation at regular intervals
        if self.rounds_completed > 0 and self.rounds_completed % self.config.get("eval_interval", 10) == 0:
            # Run only if this is a new evaluation point
            if len(self.test_metrics['rounds']) == 0 or self.test_metrics['rounds'][-1] != self.rounds_completed:
                self.evaluate_global_model()
        self.prev_round_time = time.time()
            
    def cleanup_unused_model_versions(self):
        """
        Clean up model versions that are no longer referenced by any events in the queue.
        This helps to save memory by removing unused model parameters.
        """
        # Get all model versions in the parameters dictionary
        model_versions = set(self.model_version_parameters.keys())
        
        # Always keep the latest model version
        model_versions.discard(self.global_model_version)
        
        # Find versions that are still in use by events in the queue
        versions_in_use = set()
        for event in self.event_queue.queue:
            if hasattr(event, 'data') and event.data and 'model_version' in event.data:
                versions_in_use.add(event.data['model_version'])
        
        # Find versions that can be deleted (not in use)
        versions_to_delete = model_versions - versions_in_use
        
        # Delete unused versions
        for version in versions_to_delete:
            if version in self.model_version_parameters:
                del self.model_version_parameters[version]
    def handle_event(self, event):
        """
        Handle an event based on its type.
        
        Args:
            event (Event): The event to handle
        """

        # Remove this event from client's tracking
        if event.client:
            event.client.remove_event(event.event_id)

        if event.event_type == EventType.DOWNLOAD_JOB:
            self.handle_download_event(event)
        elif event.event_type == EventType.LOCAL_TRAIN:
            self.handle_train_event(event)
        elif event.event_type == EventType.UPLOAD_UPDATE:
            self.handle_upload_event(event)
        elif event.event_type == EventType.AGGREGATE:
            self.handle_aggregate_event(event)

    def evaluate_global_model(self):
        """
        Evaluate the global model using the server's test data.
        Calculate overall and per-label metrics.
        """        
        device = self.config['train_device']
        
        # Set model to evaluation mode
        self.global_model.eval()
        self.global_model.to(device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize per-label metrics
        num_classes = self.config['nb_of_classes']
        label_correct = {i: 0 for i in range(num_classes)}
        label_total = {i: 0 for i in range(num_classes)}
        label_loss = {i: 0.0 for i in range(num_classes)}

        loss_fn = get_loss_fn(self.config['dataset'])
        
        # Check if this is a text dataset
        is_text_dataset = self.config['dataset'].lower() == 'news'
        
        with torch.no_grad():
            for batch in self.test_loader:
                if is_text_dataset:
                    # Text dataset: batch is a dict with 'input_ids', 'attention_mask', 'labels'
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    target = batch['labels'].to(device)
                    
                    # Forward pass for text model
                    model_output = self.global_model(input_ids=input_ids, attention_mask=attention_mask)
                    output = model_output.logits if hasattr(model_output, 'logits') else model_output
                else:
                    # Image dataset: batch is a tuple (data, target)
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    output = self.global_model(data)
                
                # Calculate overall loss
                loss = loss_fn(output, target, reduction='sum')
                total_loss += loss.item()
                
                # Calculate per-label loss
                for i in range(num_classes):
                    idx = (target == i)
                    if idx.sum() > 0:
                        label_loss[i] += loss_fn(output[idx], target[idx], reduction='sum').item()
                        label_total[i] += idx.sum().item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct_mask = pred.eq(target.view_as(pred))
                correct += correct_mask.sum().item()
                total += target.size(0)
                
                # Calculate per-label accuracy
                for i in range(num_classes):
                    idx = (target == i).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        label_correct[i] += correct_mask[idx].sum().item()
        
        # Calculate final metrics
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Calculate per-label metrics
        label_accuracies = {}
        label_avg_losses = {}
        for i in range(num_classes):
            if label_total[i] > 0:
                label_accuracies[i] = label_correct[i] / label_total[i]
                label_avg_losses[i] = label_loss[i] / label_total[i]
            else:
                label_accuracies[i] = 0.0
                label_avg_losses[i] = 0.0
        
        # Store metrics
        self.test_metrics['rounds'].append(self.rounds_completed)
        self.test_metrics['times'].append(self.current_time)
        self.test_metrics['accuracies'].append(accuracy)
        self.test_metrics['losses'].append(avg_loss)
        
        # Fix: Check if all class keys exist, not just if dictionary is empty
        if not self.test_metrics['label_wise_accuracies'] or any(i not in self.test_metrics['label_wise_accuracies'] for i in range(num_classes)):
            self.test_metrics['label_wise_accuracies'] = {i: [] for i in range(num_classes)}
            self.test_metrics['label_wise_losses'] = {i: [] for i in range(num_classes)}
            
            # Add current values
            for i in range(num_classes):
                self.test_metrics['label_wise_accuracies'][i].append(label_accuracies[i])
                self.test_metrics['label_wise_losses'][i].append(label_avg_losses[i])
        else:
            for i in range(num_classes):
                self.test_metrics['label_wise_accuracies'][i].append(label_accuracies[i])
                self.test_metrics['label_wise_losses'][i].append(label_avg_losses[i])
        
        # More detailed logging of evaluation results
        self.logger.info(f"Eval. at Round {self.rounds_completed}, Time {self.current_time:.2f} Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")
        
        # Log to wandb if enabled
        if self.config["wandb_flag"]:
            metrics_dict = {
                'test/accuracy': accuracy,
                'test/loss': avg_loss,
                'round': self.rounds_completed,
                'time': self.current_time
            }
                
            # Add per-label metrics
            for i in range(num_classes):
                metrics_dict[f'label_wise/accuracy_{i}'] = label_accuracies[i]
                metrics_dict[f'label_wise/loss_{i}'] = label_avg_losses[i]
            
            self.wandb.log(metrics_dict)
        
        self.save_metrics()
        
        # Save test plots if enabled
        if self.config.get('save_plot', False):
            self.save_test_plots()
        
        # Check if we need to save the model checkpoint
        if self.rounds_completed % self.config.get('save_model_interval', 100) == 0:
            self.save_model_checkpoint()

    def update_kd_buffer(self, update, model_version, client_id):
        """
        Update the KD buffer with a new update, maintaining most-recent-first order.
        
        New buffer design:
        - The buffer stores all models in most-recent-first order
        - Index 0 = most recent model (current)
        - Index 1+ = previous models for KD
        - Minimum buffer size is 1 (just current model)
        - When buffer is full, oldest model is removed from the end
        
        Args:
            update: The model update to add to the KD buffer
            model_version: The model version associated with this update
            client_id: The ID of the client who contributed this update
        """
        kd_buffer_size = self.config[self.config['kd_augmentation'] + '_settings']['kd_buffer_size']
        
        # Convert update to model parameters if needed
        if self.config["server_lr"] >= 0:
            # If server_lr is positive, the update is a difference
            old_version_params = self.model_version_parameters[model_version]
            updated_model_params = {}
            for name in old_version_params.keys():
                if name in update:
                    updated_model_params[name] = old_version_params[name] + update[name]
                else:
                    updated_model_params[name] = old_version_params[name]
        else:
            # If server_lr is negative, the update contains the full model parameters
            updated_model_params = update
        
        # Add to KD buffer at index 0 (most recent first) with client information
        self.kd_buffer.insert(0, (updated_model_params, client_id))
        
        # Maintain buffer size (remove oldest from the end)
        if len(self.kd_buffer) > kd_buffer_size:
            self.kd_buffer.pop()  # Remove oldest (from end)
        
    def set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # Set CUDA seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def create_event(self, event_type, start_time, duration=None, client=None, data=None):
        """
        Create a new event with a unique ID and add it to the client's tracking.
        
        Args:
            event_type (EventType): Type of the event
            start_time (float): Simulated time when the event starts
            duration (float, optional): Duration of the event
            client (Client, optional): Client involved in the event
            data (any, optional): Additional data associated with the event
            
        Returns:
            Event: The created event
        """
        event_id = self.next_event_id
        self.next_event_id += 1
        
        event = Event(
            event_type=event_type,
            start_time=start_time,
            duration=duration,
            client=client,
            data=data,
            config=self.config,
            event_id=event_id
        )
        
        # Add event to client's tracking if it's client-specific
        if client:
            client.add_event(event_id, event_type)
        self.event_queue.add_event(event)            
        return event

        
    def get_active_clients(self):
        """
        Get the list of active clients (clients with assigned events).
        
        Returns:
            list: List of active clients
        """
        return [client for client in self.clients if client.has_train_event()]
    
    def get_inactive_clients(self):
        """
        Get the list of inactive clients (clients without assigned events).
        
        Returns:
            list: List of inactive clients
        """
        return [client for client in self.clients if not client.has_train_event()]
    
    def count_active_clients(self):
        """
        Count the number of active clients.
        
        Returns:
            int: Number of active clients
        """
        return len(self.get_active_clients())
    
    def check_stopping_conditions(self):
        """
        Check if the training should stop based on the stopping conditions.
        """
        stop_flag = False
        if self.config["max_rounds"] > 0 and self.rounds_completed >= self.config["max_rounds"]:
            self.logger.info(f"Training completed after {self.rounds_completed} rounds")
            stop_flag = True
        elif self.config["max_wait_time"] > 0 and self.current_time >= self.config["max_wait_time"]:
            self.logger.info(f"Training completed after {self.current_time} seconds")
            stop_flag = True
        elif self.config["target_accuracy"] > 0 and len(self.test_metrics['accuracies']) > 0 and self.test_metrics['accuracies'][-1] >= self.config["target_accuracy"]:
            self.logger.info(f"Training completed since target accuracy {self.config['target_accuracy']} reached")
            stop_flag = True
        elif self.event_queue.is_empty():
            self.logger.info(f"Training completed since no more events")
            stop_flag = True
        if self.config.get("early_stop", None):
            for early_stop_cond in self.config["early_stop"]:
                if self.rounds_completed >= early_stop_cond["round"] and max(self.test_metrics['accuracies']) <= early_stop_cond["acc"]:
                    self.logger.info(f"Training completed since round {self.config['early_stop']['round']} reached and accuracy {self.config['early_stop']['acc']} not reached")
                    stop_flag = True
        if len(self.train_metrics['losses']) > 10 and \
                (np.isnan(np.mean(self.train_metrics['losses'][-10:])) or \
                np.isinf(np.mean(self.train_metrics['losses'][-10:]))):
            self.logger.info(f"Round {self.rounds_completed}, Time {self.current_time:.2f} | Losses are diverging, stopping training")
            stop_flag = True
        if len(self.test_metrics['losses']) > 10 and \
                (np.isnan(np.mean(self.test_metrics['losses'][-10:])) or \
                np.isinf(np.mean(self.test_metrics['losses'][-10:]))):
            self.logger.info(f"Round {self.rounds_completed}, Time {self.current_time:.2f} | Test losses are diverging, stopping training")
            stop_flag = True

        return stop_flag

    def save_model_checkpoint(self):
        """
        Save the current global model to a checkpoint file.
        """
        checkpoint_path = os.path.join(self.config['out_dir'], 'ckpt.pth')
        try:
            torch.save({
                'model_state_dict': self.global_model.state_dict(),
                'round': self.rounds_completed,
                'time': self.current_time,
                'model_version': self.global_model_version
            }, checkpoint_path)
            self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model checkpoint: {e}")
    
    def save_metrics(self):
        """
        Save all tracked metrics to a JSON file.
        """
        metrics_path = os.path.join(self.config['out_dir'], 'results.json')
        try:
            metrics_data = {
                'test_metrics': self.test_metrics,
                'train_metrics': self.train_metrics,
                'final_round': self.rounds_completed,
                'total_time': self.current_time,
                'model_version': self.global_model_version
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def save_test_plots(self):
        """
        Save test evaluation plots: accuracy and loss vs time and rounds.
        Creates a 2x2 subplot with:
        1. test accuracy vs time
        2. test accuracy vs round  
        3. test loss vs time
        4. test loss vs round
        """
        if not self.test_metrics['times'] or not self.test_metrics['rounds']:
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Test accuracy vs time
            ax1.plot(self.test_metrics['times'], self.test_metrics['accuracies'], 'b-', marker='o', markersize=4)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Test Accuracy vs Time')
            ax1.grid(True, alpha=0.3)
            
            # Test accuracy vs round
            ax2.plot(self.test_metrics['rounds'], self.test_metrics['accuracies'], 'g-', marker='s', markersize=4)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Test Accuracy')
            ax2.set_title('Test Accuracy vs Round')
            ax2.grid(True, alpha=0.3)
            
            # Test loss vs time
            ax3.plot(self.test_metrics['times'], self.test_metrics['losses'], 'r-', marker='^', markersize=4)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Test Loss')
            ax3.set_title('Test Loss vs Time')
            ax3.grid(True, alpha=0.3)
            
            # Test loss vs round
            ax4.plot(self.test_metrics['rounds'], self.test_metrics['losses'], 'm-', marker='d', markersize=4)
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Test Loss')
            ax4.set_title('Test Loss vs Round')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.config['out_dir'], 'test_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # self.logger.info(f"Test plots saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save test plots: {e}")
            # Close any open figure to prevent memory leaks
            plt.close('all')

    def initialize_metrics(self):
        """
        Initialize the metrics of the server.
        """
        if self.is_resuming:
            # Load metrics from checkpoint
            if not load_metrics(self):
                self.logger.error("Failed to load metrics from checkpoint. Cannot resume experiment.")
                raise RuntimeError("Failed to resume experiment: metrics could not be loaded")
        else:
            # Metrics tracking
            self.test_metrics = {
                'rounds': [],
                'times': [],
                'accuracies': [],
                'losses': [],
                'label_wise_accuracies': {},
                'label_wise_losses': {}
            }
            
            self.train_metrics = {
                'rounds': [],
                'times': [],
                'client_ids': [],
                'losses': [],
                'accuracies': []
            }


    def boot(self):
        """
        Boot the server, initializing data, model, and clients, and kd components.
        """
        self.logger.info("Booting server...")
        
        # Set random seeds for reproducibility
        # For client operations (data distribution, delays etc.)
        self.set_seed(self.config.get("setup_random_seed", 42))

        # Load data
        self.load_data()

        # Create clients
        self.create_clients()
        
        # Distribute data among clients
        self.distribute_data_to_clients()

        self.prev_round_time = time.time()

        # Initialize the global model using the create_model function
        self.global_model = create_model(config=self.config)
        
        reset_stat_tracker_stats(self.global_model)
        if self.config["model_device"] == "cuda":
            self.global_model.cuda()


        # Initialize a reusable local training model to avoid repeated model creation overhead
        self.local_training_model = create_model(config=self.config).to(self.config["model_device"])
        self.logger.info("Created reusable local training model for performance optimization")

        # Initialize reusable KD models if KD is enabled
        if self.config['kd_augmentation'] in ['simple_kd', 'dfkd'] and self.config[self.config['kd_augmentation'] + '_settings']['kd_buffer_size'] > 0:
            
            kd_buffer_size = self.config[self.config['kd_augmentation'] + '_settings']['kd_buffer_size']
            
            # Create teacher models (exactly kd_buffer_size models to match the buffer)
            self.kd_teacher_models = []
            for i in range(kd_buffer_size):  # Exactly kd_buffer_size models
                teacher_model = create_model(config=self.config).to(self.config["model_device"])
                self.kd_teacher_models.append(teacher_model)
            
            # Create one student model for KD
            self.kd_student_model = create_model(config=self.config).to(self.config["model_device"])
            
            self.logger.info(f"Created {len(self.kd_teacher_models)} reusable KD teacher models and 1 student model for performance optimization")

        else:
            self.kd_teacher_models = None
            self.kd_student_model = None
            
        # Note: Model reuse optimization includes lightweight state reset to maintain determinism
        self.logger.info("Model reuse optimization enabled with lightweight state reset for deterministic behavior")
        
        # Create optimized client KD dataloaders if distribution_match_batch is enabled
        if (self.config['kd_augmentation'] == 'simple_kd' and
            self.config['simple_kd_settings']['kd_version'] == 'distribution_match_batch'):
            assert self.kd_data is not None, "KD data is not initialized"
            self.logger.info("Creating optimized client-specific KD dataloaders for distribution_match_batch...")
            self.client_kd_samplers, self.client_kd_dataloaders = create_client_kd_samplers_and_dataloaders(
                self.kd_data, self.clients, self.config
            )
            
            self.logger.info(f"Created optimized KD dataloaders for {len(self.client_kd_dataloaders)} clients")
        else:
            # Initialize empty if not using distribution_match_batch
            self.client_kd_samplers = {}
            self.client_kd_dataloaders = {}
            
        # Initialize DFKD synthesizer if enabled
        if self.config['kd_augmentation'] == 'dfkd':
            self.logger.info("Initializing DFKD synthesizer and related components...")

            # Check if this is a text dataset
            if self.config['dataset'].lower() == 'news':
                # Create text synthesizer for news dataset
                self.synthesizer = FastTextMetaSynthesizer(
                    config=self.config
                )
                self.generator = None  # Text synthesizer uses trainable prompt vectors instead of generator
                self.logger.info(f"Text DFKD synthesizer initialized for dataset '{self.config['dataset']}'")
            else:
                # Create image synthesizer for image datasets (CIFAR10, CIFAR100, FEMNIST)
                dfkd_settings = self.config['dfkd_settings'] 
                self.generator = DFKDGenerator(
                    dataset_name=self.config['dataset'],
                    nz=dfkd_settings['nz'],
                    ngf=dfkd_settings['ngf']
                )
                
                # Create the synthesizer
                self.synthesizer = FastMetaSynthesizer(
                    generator=self.generator,
                    config=self.config
                )
                
                self.logger.info(f"Image DFKD synthesizer initialized with generator for dataset '{self.config['dataset']}'")
        else:
            # Initialize as None if not using DFKD
            self.generator = None
            self.synthesizer = None


        
        if self.is_resuming:
            # Load server state from checkpoint
            if not load_server_state(self):
                self.logger.error("Failed to load server state from checkpoint. Cannot resume experiment.")
                raise RuntimeError("Failed to resume experiment: server state could not be loaded")
        else:
            self.model_version_parameters = {0: get_model_params(self.global_model, self.config["model_device"])}
            self.current_time = 0
            self.global_model_version = 0
            self.next_event_id = 1
            self.rounds_completed = 0

        # Initialize server optimizers based on config
        server_opt = self.config.get('server_optimizer', 'sgd')
        settings = self.config.get('server_optimizer_settings', {})
        param_names = list(self.model_version_parameters[self.global_model_version].keys())

        if server_opt == 'amsgrad':
            # FADAS: AMSGrad with delay-adaptive learning rate
            self.server_amsgrad = ServerAMSGrad(
                param_names=param_names,
                beta1=settings.get('beta1', 0.9),
                beta2=settings.get('beta2', 0.999),
                eps=settings.get('eps', 1e-8)
            )
            self.server_fedac = None
            self.logger.info(f"FADAS server optimizer initialized (AMSGrad: beta1={settings.get('beta1', 0.9)}, beta2={settings.get('beta2', 0.99)}, tau_c={settings.get('tau_c', 10)})")
        elif server_opt == 'fedac':
            # FedAC: Nadam (prospective momentum) with cosine-similarity client weighting
            self.server_fedac = ServerFedAC(
                param_names=param_names,
                beta1=settings.get('beta1', 0.6),
                beta2=settings.get('beta2', 0.9),
                eps=settings.get('eps', 1e-8)
            )
            self.server_amsgrad = None
            self.logger.info(f"FedAC server optimizer initialized (Nadam: beta1={settings.get('beta1', 0.6)}, beta2={settings.get('beta2', 0.9)}, eps={settings.get('eps', 1e-8)})")
        else:
            self.server_amsgrad = None
            self.server_fedac = None

        # FedAC accumulates per-upload cosine similarity scores for the current buffer window
        if server_opt == 'fedac':
            if not self.is_resuming:
                self.fedac_scores = []
            # (resume case: loaded by load_fedac_state below)
        else:
            self.fedac_scores = []

        self.logger.info("Server booted successfully")

        if self.is_resuming:
            # Load the random states
            if not load_random_states(self):
                self.logger.warning("Failed to load random states from checkpoint. Random number generation may not be fully reproducible.")
            # Load event queue and update buffer from checkpoint
            if not load_event_queue(self):
                self.logger.error("Failed to load event queue from checkpoint. Cannot resume experiment.")
                raise RuntimeError("Failed to resume experiment: event queue could not be loaded")
            if not load_update_buffer(self):
                self.logger.error("Failed to load update buffer from checkpoint. Cannot resume experiment.")
                raise RuntimeError("Failed to resume experiment: update buffer could not be loaded")
            if not load_kd_buffer(self):
                self.logger.error("Failed to load KD buffer from checkpoint. Cannot resume experiment.")
                raise RuntimeError("Failed to resume experiment: KD buffer could not be loaded")
            # Load DFKD state if enabled
            if not load_dfkd_state(self, os.path.join(self.config["out_dir"], "checkpoint_exp_state")):
                raise RuntimeError("Failed to resume experiment: DFKD state could not be loaded")
            # Load FADAS optimizer state if enabled
            if not load_fadas_state(self):
                if self.config.get('server_optimizer', 'sgd') == 'amsgrad':
                    raise RuntimeError("Failed to resume experiment: FADAS optimizer state could not be loaded")
            # Load FedAC optimizer state if enabled
            if not load_fedac_state(self):
                if self.config.get('server_optimizer', 'sgd') == 'fedac':
                    raise RuntimeError("Failed to resume experiment: FedAC optimizer state could not be loaded")
        else:
            # For training and other runtime operations 
            self.set_seed(self.config["run_random_seed"])
            self.update_buffer = []
            self.kd_buffer = []  # Buffer for KD updates (stores tuples of (model_parameters, client_id))
            self.event_queue = EventQueue()
            # Schedule initial events
            self.schedule_initial_events()

        self.logger.info("Initial training events scheduled")