# src/model_training.py
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import logging
from typing import Dict, Any, Union
import os

from src.utils import save_object # Import helper

# Define model types more explicitly
ModelType = Union[lgb.LGBMClassifier, LogisticRegression, RandomForestClassifier]

def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]) -> ModelType:
    """Trains the specified model and saves it."""
    cfg_train = config['train']
    cfg_output = config['output']
    model_type = cfg_train['model_type']
    model_params = cfg_train.get(f"{model_type}_params", {}) # Get params for the specific type

    logging.info(f"Starting model training: {model_type}")
    logging.info(f"Using parameters: {model_params}")

    if model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**model_params)
    # Add elif blocks here for other model types if needed
    # elif model_type == 'logistic':
    #     model = LogisticRegression(**model_params)
    # elif model_type == 'random_forest':
    #      model = RandomForestClassifier(**model_params)
    else:
        logging.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

    try:
        model.fit(X_train, y_train)
        logging.info("Model training complete.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise # Re-raise the exception to stop the pipeline

    # Save the trained model
    model_path = os.path.join(cfg_output['model_dir'], cfg_output['model_name'])
    save_object(model, model_path)

    return model