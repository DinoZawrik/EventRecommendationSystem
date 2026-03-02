# src/predict.py
"""Test set prediction and Kaggle submission generation."""

import logging
import os
from typing import Any, Dict

import pandas as pd

try:
    from src.utils import ModelType
except ImportError:
    logging.warning("Could not import ModelType from src.utils — falling back to Any.")
    from typing import Any as ModelType


def generate_predictions(model: ModelType, X_test: pd.DataFrame) -> pd.Series:
    """Generate fraud probability predictions for the test set.

    Args:
        model: Trained classifier with ``predict_proba`` method.
        X_test: Feature matrix for the test set.

    Returns:
        pd.Series of predicted probabilities aligned with ``X_test.index``.
    """
    logging.info("Generating predictions for %d test samples...", len(X_test))

    if X_test.isnull().any().any():
        nan_cols = X_test.isnull().sum()
        logging.warning(
            "NaN values detected in test features before prediction:\n%s",
            nan_cols[nan_cols > 0],
        )

    try:
        probabilities = model.predict_proba(X_test)[:, 1]
        logging.info("Predictions generated successfully.")
        return pd.Series(probabilities, index=X_test.index)
    except ValueError as ve:
        logging.error("ValueError during prediction (often NaN or feature mismatch): %s", ve)
        logging.error("NaN counts in X_test:\n%s", X_test.isnull().sum())
        raise
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise


def create_submission_file(
    processed_test_df: pd.DataFrame,
    probabilities: pd.Series,
    config: Dict[str, Any],
) -> None:
    """Create a Kaggle-format submission CSV.

    Args:
        processed_test_df: Processed test DataFrame containing user/event ID columns.
        probabilities: Predicted probabilities aligned with ``processed_test_df``.
        config: Project configuration dictionary.
    """
    cfg_pred = config["predict"]
    cfg_feat = config["features"]
    user_col = cfg_feat["user_id_col"]
    event_col = cfg_feat["event_id_col"]
    submission_path = cfg_pred["submission_file"]

    logging.info("Creating submission file...")

    if not processed_test_df.index.equals(probabilities.index):
        logging.warning("Index mismatch between test data and probabilities — reindexing.")
        probabilities = probabilities.reindex(processed_test_df.index)
        if probabilities.isnull().any():
            logging.error("Could not reliably align probabilities after reindexing.")
            raise ValueError("Data mismatch for submission.")

    submission_df = processed_test_df[[user_col, event_col]].copy()
    submission_df["Probability"] = probabilities.values

    logging.info("Ranking events per user...")
    submission_df = submission_df.sort_values(
        by=[user_col, "Probability"], ascending=[True, False]
    )

    ranked_events = submission_df.groupby(user_col)[event_col].apply(
        lambda x: " ".join(x.astype(str))
    )

    final_submission = pd.DataFrame(ranked_events).reset_index()
    final_submission.columns = ["User", "Events"]
    final_submission["User"] = final_submission["User"].astype(int)

    try:
        final_submission.to_csv(submission_path, index=False)
        logging.info("Submission file saved to %s", submission_path)
    except Exception as e:
        logging.error("Error saving submission file: %s", e)
        raise