import os
import json
import argparse
import torch
from server import Server
from config import Config
from utils.logging_utils import setup_logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to experiment directory to resume from')
    args = parser.parse_args()

    # Initialize configuration
    if args.config:
        config = Config(args.config)
    else:
        config = Config()

    if args.resume_from:
        config["continue_exp_from"] = args.resume_from
    
    # Setup logger
    logger = setup_logger("Main")
    logger.info("Starting FL Simulation")
    
    # Print key configuration parameters
    logger.info(f"Algorithm: {config['algorithm']}")
    logger.info(f"Number of clients: {config['num_clients']}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Data distribution: {config['data_distribution']}")
    
    # Initialize server
    logger.info("Initializing server")
    server = Server(config)
    
    # Boot server
    logger.info("Booting server")
    server.boot()
    
    # Run training
    logger.info("Starting training")
    server.train()
    
    logger.info("Simulation complete")

if __name__ == "__main__":
    main() 

    exit(0)


# %%