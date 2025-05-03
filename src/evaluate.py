# src/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import logging
import json
import os
from typing import Dict, Any, List, Tuple

from src.recommend import recommend_events_for_users # Import recommendation logic
from src.utils import ModelType # Import model type hint

def get_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculates classification report and AUC ROC."""
    logging.info("Calculating classification metrics...")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    logging.info(f"Validation AUC-ROC: {auc:.4f}")
    # print(classification_report(y_true, y_pred, zero_division=0)) # Keep print for quick view
    return {"classification_report": report, "auc_roc": auc}

def average_precision_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """Calculates Average Precision at K for a single user."""
    if not actual:
        return 0.0

    predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]: # Check relevance and avoid double counting
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k) # Normalize by min(actual relevant, k)

def map_at_k(actual: List[List[Any]], predicted: List[List[Any]], k: int) -> float:
    """Calculates Mean Average Precision at K."""
    logging.info(f"Calculating MAP@{k}...")
    if len(actual) != len(predicted):
        logging.error("Length mismatch between actual and predicted lists for MAP@k.")
        return 0.0
    ap_sum = sum(average_precision_at_k(a, p, k) for a, p in zip(actual, predicted))
    map_score = ap_sum / len(actual)
    logging.info(f"MAP@{k}: {map_score:.4f}")
    return map_score

def evaluate_model(
    model: ModelType,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    processed_val_data: pd.DataFrame, # DataFrame containing user/event IDs for val set
    config: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Evaluates the model using classification metrics and MAP@k."""
    logging.info("Starting model evaluation on validation set...")
    cfg_eval = config['evaluate']
    cfg_output = config['output']
    k = cfg_eval['map_k']

    # --- Classification Metrics ---
    y_pred_val = model.predict(X_val)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    class_metrics = get_classification_metrics(y_val, y_pred_val, y_prob_val)

    # --- MAP@k Calculation ---
    logging.info(f"Preparing data for MAP@{k} calculation...")
    # Get recommendations for all users in the validation set
    val_users = processed_val_data[config['features']['user_id_col']].unique()

    # We need the model predictions (probabilities) for the validation set user-event pairs
    processed_val_data['probability'] = y_prob_val # Add probabilities calculated earlier

    all_predicted_lists = recommend_events_for_users(
        user_ids=val_users,
        model=model, # Pass model if needed by recommend function (might not be if using pre-computed probs)
        prediction_df=processed_val_data, # Pass df with users, events, features, and PROBABILITIES
        config=config
    )

    # Prepare actual relevant items for each user
    all_actual_lists = []
    actual_grouped = processed_val_data[processed_val_data[config['features']['target_col']] == 1].groupby(config['features']['user_id_col'])
    for user_id in val_users:
        if user_id in actual_grouped.groups:
            actual_events = actual_grouped.get_group(user_id)[config['features']['event_id_col']].astype(str).tolist()
            all_actual_lists.append(actual_events)
        else:
            all_actual_lists.append([]) # User had no positive interactions in validation

    # Calculate MAP@k
    map_score = map_at_k(all_actual_lists, all_predicted_lists, k=k)

    # --- Combine and Save Metrics ---
    all_metrics = {
        "auc_roc": class_metrics["auc_roc"],
        f"map_at_{k}": map_score,
        "classification_report": class_metrics["classification_report"] # Can be nested
    }

    metrics_path = os.path.join(cfg_output['reports_dir'], cfg_eval['metrics_file'])
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    try:
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logging.info(f"Evaluation metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {metrics_path}: {e}")

    logging.info("Model evaluation finished.")
    return all_metrics