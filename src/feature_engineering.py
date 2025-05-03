# src/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # Moved SMOTE here as it modifies features/target

from src.utils import count_user_ids # Import helper

def preprocess_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    """Converts a column to datetime objects."""
    logging.debug(f"Preprocessing datetime column: {column}")
    return pd.to_datetime(df[column], format='ISO8601', errors='coerce', utc=True)

def create_features(datasets: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Creates all features for users, events, and merges them with train/test.
    Returns processed train_df, test_df, and potentially others if needed.
    """
    logging.info("Starting feature engineering...")
    cfg_feat = config['features']
    cfg_data = config['data']
    cfg_prep = config['preprocessing']

    users_df = datasets['users'].copy()
    events_df = datasets['events'].copy()
    event_attendees_df = datasets['event_attendees'].copy()
    train_df = datasets['train'].copy()
    test_df = datasets['test'].copy()

    # --- 1. Preprocess Datetime ---
    logging.info("Preprocessing datetime columns...")
    train_df['timestamp'] = preprocess_datetime(train_df, cfg_feat['datetime_cols']['train'])
    test_df['timestamp'] = preprocess_datetime(test_df, cfg_feat['datetime_cols']['test'])
    events_df['start_time'] = preprocess_datetime(events_df, cfg_feat['datetime_cols']['events'])
    users_df['joinedAt'] = preprocess_datetime(users_df, cfg_feat['datetime_cols']['users'])
    users_df[cfg_feat['birthyear_col']] = pd.to_numeric(users_df[cfg_feat['birthyear_col']], errors='coerce')

    # --- 2. User Features ---
    logging.info("Creating user features...")
    users_df['age'] = cfg_feat['current_year'] - users_df[cfg_feat['birthyear_col']]
    users_df.loc[users_df['age'] > cfg_feat['age_outlier_threshold'], 'age'] = np.nan # Use NaN, impute later
    users_df['user_joined_month'] = users_df['joinedAt'].dt.month
    users_df['user_joined_year'] = users_df['joinedAt'].dt.year
    users_df['has_location'] = (~users_df['location'].isnull()).astype(int)
    # Impute gender NaN with mode before encoding
    gender_mode = users_df['gender'].mode()[0]
    users_df['gender'] = users_df['gender'].fillna(gender_mode)
    users_df = pd.get_dummies(users_df, columns=['gender'], prefix='gender', drop_first=True) # gender_male
    users_df = pd.get_dummies(users_df, columns=['has_location'], prefix='has_location', drop_first=True) # has_location_1

    # --- 3. Event Features ---
    logging.info("Creating event features...")
    events_df['event_start_weekday'] = events_df['start_time'].dt.weekday
    events_df['event_start_hour'] = events_df['start_time'].dt.hour
    events_df['is_weekend_event'] = events_df['event_start_weekday'].isin([5, 6]).astype(int)
    events_df['has_geo_info'] = (~events_df['city'].isnull()).astype(int)
    events_df = pd.get_dummies(events_df, columns=['is_weekend_event'], prefix='is_weekend_event', drop_first=True) # is_weekend_event_1
    events_df = pd.get_dummies(events_df, columns=['has_geo_info'], prefix='has_geo_info', drop_first=True) # has_geo_info_1

    # --- 4. Event Popularity Features ---
    logging.info("Creating event popularity features...")
    event_attendees_df['yes_count'] = event_attendees_df['yes'].apply(count_user_ids)
    event_attendees_df['maybe_count'] = event_attendees_df['maybe'].apply(count_user_ids)
    event_attendees_df['invited_count'] = event_attendees_df['invited'].apply(count_user_ids)
    event_attendees_df['no_count'] = event_attendees_df['no'].apply(count_user_ids)

    # Select only necessary columns before merging
    popularity_features = event_attendees_df[['event', 'yes_count', 'maybe_count', 'invited_count', 'no_count']]
    events_df = pd.merge(events_df, popularity_features, left_on='event_id', right_on='event', how='left')
    events_df = events_df.drop(columns=['event']) # Drop the duplicated 'event' column

    # Fill NaN counts with 0 after merge (events with no attendees data)
    count_cols = ['yes_count', 'maybe_count', 'invited_count', 'no_count']
    events_df[count_cols] = events_df[count_cols].fillna(0)

    # Calculate ratios
    epsilon = cfg_feat['event_popularity_epsilon']
    total_responses = events_df['yes_count'] + events_df['no_count'] + events_df['maybe_count'] + events_df['invited_count']
    events_df['yes_ratio'] = events_df['yes_count'] / (total_responses + epsilon)
    events_df['yes_vs_no_ratio'] = events_df['yes_count'] / (events_df['no_count'] + epsilon)
    events_df['attendance_rate'] = (events_df['yes_count'] + events_df['maybe_count']) / (events_df['invited_count'] + epsilon) # Original definition used invited_count

    # --- 5. Interaction Time Features (need merging) ---
    logging.info("Creating interaction time features...")
    # Hrs to event
    train_df = pd.merge(train_df, events_df[['event_id', 'start_time']], left_on='event', right_on='event_id', how='left')
    train_df['hrs_to_event'] = (train_df['start_time'] - train_df['timestamp']).dt.total_seconds() / 3600
    train_df = train_df.drop(columns=['event_id', 'start_time']) # Clean up merge cols

    test_df = pd.merge(test_df, events_df[['event_id', 'start_time']], left_on='event', right_on='event_id', how='left')
    test_df['hrs_to_event'] = (test_df['start_time'] - test_df['timestamp']).dt.total_seconds() / 3600
    test_df = test_df.drop(columns=['event_id', 'start_time'])

    # Mins from Join to Event View
    train_df = pd.merge(train_df, users_df[['user_id', 'joinedAt']], left_on='user', right_on='user_id', how='left')
    train_df['minsToEvent_frmJoin'] = (train_df['timestamp'] - train_df['joinedAt']).dt.total_seconds() / 60
    train_df = train_df.drop(columns=['user_id', 'joinedAt', 'timestamp']) # Clean up, timestamp no longer needed

    test_df = pd.merge(test_df, users_df[['user_id', 'joinedAt']], left_on='user', right_on='user_id', how='left')
    test_df['minsToEvent_frmJoin'] = (test_df['timestamp'] - test_df['joinedAt']).dt.total_seconds() / 60
    test_df = test_df.drop(columns=['user_id', 'joinedAt', 'timestamp'])

    # --- 6. Encode 'invited' feature ---
    logging.info("Encoding 'invited' feature...")
    train_df = pd.get_dummies(train_df, columns=['invited'], prefix='invited', drop_first=True) # invited_1
    test_df = pd.get_dummies(test_df, columns=['invited'], prefix='invited', drop_first=True)

    # --- 7. Final Merge ---
    logging.info("Merging all features into train and test sets...")
    # Select relevant columns from users and events before merge to avoid large DFs
    user_cols_to_merge = ['user_id', 'age', 'gender_male', 'has_location_1', 'user_joined_month', 'user_joined_year']
    event_cols_to_merge = ['event_id', 'is_weekend_event_1', 'has_geo_info_1', 'c_1', 'c_2', 'c_3', # Add c_1..100 if needed
                           'event_start_hour', 'yes_ratio', 'yes_vs_no_ratio', 'attendance_rate',
                           'yes_count', 'maybe_count', 'invited_count', 'no_count']

    # Merge users into train/test
    train_merged = pd.merge(train_df, users_df[user_cols_to_merge], left_on='user', right_on='user_id', how='left')
    test_merged = pd.merge(test_df, users_df[user_cols_to_merge], left_on='user', right_on='user_id', how='left')

    # Merge events into train/test
    train_final = pd.merge(train_merged, events_df[event_cols_to_merge], left_on='event', right_on='event_id', how='left')
    test_final = pd.merge(test_merged, events_df[event_cols_to_merge], left_on='event', right_on='event_id', how='left')

    # Drop intermediate IDs
    train_final = train_final.drop(columns=['user_id', 'event_id'], errors='ignore')
    test_final = test_final.drop(columns=['user_id', 'event_id'], errors='ignore')

    logging.info(f"Final train features shape: {train_final.shape}")
    logging.info(f"Final test features shape: {test_final.shape}")

    # --- 8. Impute remaining NaNs (Age) ---
    # Impute AFTER merging, using train set's median for both train and test
    age_median_train = train_final['age'].median()
    logging.info(f"Imputing NaN 'age' with median from train set: {age_median_train:.2f}")
    train_final['age'] = train_final['age'].fillna(age_median_train)
    test_final['age'] = test_final['age'].fillna(age_median_train) # Use train median for test

    # Impute potential NaNs in other numeric columns created by merges (e.g., event features for unknown events)
    numeric_cols = cfg_feat['selected_features'] # Use the final list
    for col in numeric_cols:
        if col in train_final.columns and train_final[col].isnull().any():
            if pd.api.types.is_numeric_dtype(train_final[col]):
                 # Check if column exists in train_final before calculating median
                if col in train_final.columns:
                    median_val = train_final[col].median()
                    logging.warning(f"Imputing NaNs in '{col}' with median: {median_val}")
                    train_final[col] = train_final[col].fillna(median_val)
                    # Check if column exists in test_final before filling
                    if col in test_final.columns:
                        test_final[col] = test_final[col].fillna(median_val) # Use train median
                    else:
                         logging.warning(f"Column '{col}' not found in test set for NaN imputation.")
                else:
                     logging.warning(f"Column '{col}' for median imputation not found in training set.")


    logging.info("Feature engineering finished.")
    return {"train": train_final, "test": test_final}


def prepare_data_for_model(
    processed_datasets: Dict[str, pd.DataFrame],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Selects features, splits data, applies SMOTE if configured.
    Returns dictionary with X_train, X_val, y_train, y_val, X_test, processed_test_df.
    """
    logging.info("Preparing data for modeling...")
    cfg_feat = config['features']
    cfg_prep = config['preprocessing']

    train_df = processed_datasets['train']
    test_df = processed_datasets['test'] # Keep original test for submission generation

    selected_features = cfg_feat['selected_features']
    target = cfg_feat['target_col']

    # Ensure all selected features exist
    missing_features = [f for f in selected_features if f not in train_df.columns]
    if missing_features:
        logging.error(f"Missing selected features in training data: {missing_features}")
        raise ValueError("Missing required features after processing.")

    X = train_df[selected_features]
    y = train_df[target]
    X_test = test_df[selected_features]

    # Check for NaNs before splitting/SMOTE
    if X.isnull().any().any():
        logging.warning(f"NaNs detected in features before split: \n{X.isnull().sum()[X.isnull().sum() > 0]}")
        # Consider adding more robust imputation here if needed, though handled in create_features
    if X_test.isnull().any().any():
         logging.warning(f"NaNs detected in test features: \n{X_test.isnull().sum()[X_test.isnull().sum() > 0]}")


    # Split data
    logging.info(f"Splitting data with test_size={cfg_prep['test_split_ratio']}...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg_prep['test_split_ratio'],
        random_state=cfg_prep['random_state'],
        stratify=y # Important for imbalanced data
    )
    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # Apply SMOTE if enabled
    if cfg_prep['use_smote']:
        logging.info("Applying SMOTE to the training data...")
        smote = SMOTE(**cfg_prep['smote_params'])
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logging.info(f"Data shape after SMOTE: {X_train_res.shape}")
            logging.info(f"Class distribution after SMOTE:\n{y_train_res.value_counts(normalize=True)}")
        except Exception as e:
            logging.error(f"Error during SMOTE: {e}")
            logging.warning("Proceeding without SMOTE.")
            X_train_res, y_train_res = X_train, y_train # Fallback
    else:
        logging.info("SMOTE is disabled.")
        X_train_res, y_train_res = X_train, y_train

    logging.info("Data preparation for modeling finished.")
    return {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        # Include original processed DFs needed for evaluation/submission
        "processed_train_val_split": train_df.loc[X_val.index], # For MAP@k context
        "processed_test": test_df # For submission context
    }