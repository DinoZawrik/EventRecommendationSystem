# src/utils.py
import yaml
import joblib
import logging
import sys
import pandas as pd
from typing import Dict, Any, List, Optional, Union # Added Union
import os

# --- Add required imports for ModelType ---
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# --- End imports for ModelType ---


# --- Define ModelType centrally ---
ModelType = Union[lgb.LGBMClassifier, LogisticRegression, RandomForestClassifier]
# --- End ModelType Definition ---


def load_config(config_path: str = "config/params.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

def setup_logging(log_file: str = "project.log"):
    """Configures logging to file and console."""
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir: # Check if log_dir is not empty (i.e., not just the filename)
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout) # Also print to console
                        ])
    logging.info("Logging setup complete.")

def save_object(obj: Any, filepath: str):
    """Saves a Python object using joblib."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, filepath)
        logging.info(f"Object saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving object to {filepath}: {e}")

def load_object(filepath: str) -> Any:
    """Loads a Python object using joblib."""
    try:
        obj = joblib.load(filepath)
        logging.info(f"Object loaded successfully from {filepath}")
        return obj
    except FileNotFoundError:
        logging.error(f"Object file not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading object from {filepath}: {e}")
        sys.exit(1)

def count_user_ids(user_id_string: Optional[str]) -> int:
    """Counts space-separated user IDs in a string."""
    if pd.isna(user_id_string) or not isinstance(user_id_string, str) or user_id_string.strip() == "":
        return 0
    else:
        return len(user_id_string.split())