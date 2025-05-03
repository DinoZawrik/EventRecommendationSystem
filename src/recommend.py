# src/recommend.py
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import numpy as np # Added for handling potential NaN probabilities

from src.utils import ModelType # Import model type hint

def recommend_events_for_user(
    user_id: int,
    model: ModelType, # Keep model in case direct prediction is needed, though probs might be precomputed
    prediction_df: pd.DataFrame, # Must contain user, event, features, and optionally precomputed probability
    config: Dict[str, Any]
) -> List[str]:
    """
    Recommends top_n events for a single user based on model probabilities.
    Assumes prediction_df contains the necessary user-event pairs and features/probabilities.
    """
    cfg_rec = config['recommend']
    cfg_feat = config['features']
    top_n = cfg_rec['top_n']
    user_col = cfg_feat['user_id_col']
    event_col = cfg_feat['event_id_col']

    user_data = prediction_df[prediction_df[user_col] == user_id]

    if user_data.empty:
        logging.warning(f"No data found for user {user_id} in the prediction set.")
        return []

    # Check if probabilities are already computed (preferred)
    if 'probability' not in user_data.columns:
        logging.warning(f"Probability column not found for user {user_id}. Predicting...")
         # Ensure features are present and in correct order
        features = cfg_feat['selected_features']
        missing_features = [f for f in features if f not in user_data.columns]
        if missing_features:
            logging.error(f"Missing features for prediction for user {user_id}: {missing_features}")
            return []
        X_user = user_data[features]
         # Simple NaN check/fill for safety, though should be handled in FE
        if X_user.isnull().any().any():
             logging.warning(f"NaNs found in features for user {user_id} before prediction. Imputing with 0 (consider refining).")
             X_user = X_user.fillna(0) # Basic imputation, refine if necessary
        try:
            probabilities = model.predict_proba(X_user)[:, 1]
        except Exception as e:
             logging.error(f"Error predicting probabilities for user {user_id}: {e}")
             return []
    else:
        probabilities = user_data['probability'].values

    # Handle potential NaN probabilities if they occurred
    if np.isnan(probabilities).any():
        logging.warning(f"NaN probabilities found for user {user_id}. Replacing with -1 for sorting.")
        probabilities = np.nan_to_num(probabilities, nan=-1.0) # Replace NaN with a low value

    recommendation_df = pd.DataFrame({
        event_col: user_data[event_col],
        'probability': probabilities
    })

    # Sort by probability (descending) and get top N
    top_recommendations = recommendation_df.sort_values(by='probability', ascending=False).head(top_n)

    recommended_event_ids = top_recommendations[event_col].astype(str).tolist()
    logging.debug(f"Top {top_n} recommendations for user {user_id}: {recommended_event_ids}")

    return recommended_event_ids


def recommend_events_for_users(
    user_ids: List[int],
    model: ModelType,
    prediction_df: pd.DataFrame, # Should contain all necessary user-event pairs + features/probs
    config: Dict[str, Any]
    ) -> List[List[str]]:
    """Generates recommendations for a list of users."""
    logging.info(f"Generating recommendations for {len(user_ids)} users...")
    all_recs = []
    for user_id in user_ids:
        recs = recommend_events_for_user(user_id, model, prediction_df, config)
        all_recs.append(recs)
    logging.info("Finished generating recommendations for user list.")
    return all_recs