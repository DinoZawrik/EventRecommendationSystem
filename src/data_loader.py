# src/data_loader.py
import pandas as pd
import os
import logging
from typing import Dict, Any

def load_datasets(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Loads all necessary datasets specified in the config."""
    data_config = config['data']
    raw_dir = data_config['raw_dir']
    datasets = {}
    files_to_load = {
        'train': data_config['train_csv'],
        'test': data_config['test_csv'],
        'events': data_config['events_csv'],
        'users': data_config['users_csv'],
        'event_attendees': data_config['event_attendees_csv'],
    }

    logging.info("Starting dataset loading...")
    for name, filename in files_to_load.items():
        filepath = os.path.join(raw_dir, filename)
        try:
            # Detect if file is gzipped based on extension
            compression = 'gzip' if filename.endswith('.gz') else None
            datasets[name] = pd.read_csv(filepath, compression=compression)
            logging.info(f"Loaded {name} data from {filepath} ({len(datasets[name])} rows)")
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            # Decide if this is critical - maybe exit or just warn
            # For now, we'll log error and continue, main script should handle missing data
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")

    if not datasets:
        logging.critical("No datasets were loaded. Exiting.")
        exit(1)

    logging.info("Dataset loading finished.")
    return datasets