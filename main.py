# main.py
import argparse
import logging
import os

from src.utils import load_config, setup_logging, load_object
from src.data_loader import load_datasets
from src.feature_engineering import create_features, prepare_data_for_model
from src.model_training import train_model
from src.evaluate import evaluate_model
from src.predict import generate_predictions, create_submission_file
from src.recommend import recommend_events_for_user

def main():
    parser = argparse.ArgumentParser(description="Event Recommendation System Pipeline")
    parser.add_argument("--config", type=str, default="config/params.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'predict', 'recommend'], help="Pipeline mode: train, predict, or recommend")
    parser.add_argument("--user_id", type=int, help="User ID for recommendation mode")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_file = os.path.join(config['output']['reports_dir'], config['data']['log_file'])
    os.makedirs(config['output']['reports_dir'], exist_ok=True)
    setup_logging(log_file)

    logging.info(f"Running pipeline in mode: {args.mode}")

    if args.mode == 'train':
        # --- Training Mode ---
        logging.info("--- Starting Training Pipeline ---")
        datasets = load_datasets(config)
        processed_datasets = create_features(datasets, config)
        model_data = prepare_data_for_model(processed_datasets, config)

        model = train_model(model_data['X_train'], model_data['y_train'], config)

        evaluate_model(
            model=model,
            X_val=model_data['X_val'],
            y_val=model_data['y_val'],
            processed_val_data=model_data['processed_train_val_split'], # Pass processed data with IDs
            config=config
        )
        logging.info("--- Training Pipeline Finished ---")

    elif args.mode == 'predict':
        # --- Prediction Mode ---
        logging.info("--- Starting Prediction Pipeline ---")
        # Load only necessary data for prediction
        datasets = load_datasets(config) # Need users/events for FE
        if 'test' not in datasets:
             logging.critical("Test dataset not loaded. Cannot run prediction.")
             exit(1)

        # Reuse FE logic, ensuring consistency
        processed_datasets = create_features(datasets, config)

        # Prepare test data features (no splitting/SMOTE needed here)
        # We need the feature selection and potential imputation consistency
        # Using prepare_data_for_model might be overkill, let's refine FE output
        test_df_processed = processed_datasets['test']
        selected_features = config['features']['selected_features']
        missing_test_features = [f for f in selected_features if f not in test_df_processed.columns]
        if missing_test_features:
            logging.error(f"Missing selected features in processed test data: {missing_test_features}")
            exit(1)
        X_test = test_df_processed[selected_features]

        # Load the trained model
        model_path = os.path.join(config['output']['model_dir'], config['output']['model_name'])
        model = load_object(model_path)

        # Generate predictions
        probabilities = generate_predictions(model, X_test)

        # Create submission file (pass the df with user/event IDs)
        create_submission_file(test_df_processed, probabilities, config)
        logging.info("--- Prediction Pipeline Finished ---")

    elif args.mode == 'recommend':
        # --- Recommendation Mode ---
        if args.user_id is None:
            logging.error("User ID must be provided for recommendation mode using --user_id")
            exit(1)

        logging.info(f"--- Starting Recommendation for User ID: {args.user_id} ---")

        # Load the trained model
        model_path = os.path.join(config['output']['model_dir'], config['output']['model_name'])
        model = load_object(model_path)

        # How to get features for the user? Use configured source.
        data_source = config['recommend']['recommendation_data_source']

        if data_source == 'load_processed_test':
            # Simple approach: Load processed test data and filter
            # Assumes user is in the test set - limited but easy demo
            logging.warning("Using processed test set for recommendations. User must be in test data.")
            # We need to run FE to get the processed test set if it wasn't saved
            datasets = load_datasets(config)
            if 'test' not in datasets:
                logging.critical("Test dataset not loaded. Cannot run recommendation from test set.")
                exit(1)
            processed_datasets = create_features(datasets, config)
            prediction_df = processed_datasets['test'] # Contains features and user/event IDs
            # Add dummy probability column if not present (recommend func can predict if needed)
            if 'probability' not in prediction_df.columns:
                 prediction_df['probability'] = 0.0 # Placeholder

        # elif data_source == 'process_user_live':
            # Advanced: Load raw data, run FE specifically for this user against all events
            # More complex to implement correctly here.
            # logging.info("Processing features live for user recommendation...")
            # prediction_df = ... # Requires careful FE application
            # pass # Implement this logic if needed
        else:
            logging.error(f"Unsupported recommendation_data_source: {data_source}")
            exit(1)


        recommendations = recommend_events_for_user(
            user_id=args.user_id,
            model=model,
            prediction_df=prediction_df,
            config=config
        )

        if recommendations:
            print(f"\nTop {config['recommend']['top_n']} Recommendations for User {args.user_id}:")
            for i, event_id in enumerate(recommendations):
                print(f"{i+1}. Event ID: {event_id}")
        else:
            print(f"\nCould not generate recommendations for User {args.user_id}.")

        logging.info("--- Recommendation Finished ---")

    else:
        # This case should not be reached due to argparse choices
        logging.error(f"Invalid mode specified: {args.mode}")

if __name__ == "__main__":
    main()