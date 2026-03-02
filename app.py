# app.py
import gradio as gr
import pandas as pd
import json
import logging
import os
import time

from src.utils import load_config, load_object, setup_logging
from src.data_loader import load_datasets
from src.feature_engineering import create_features, preprocess_datetime

try:
    from src.utils import ModelType
except ImportError:
    from typing import Any as ModelType

# ---------------------------------------------------------------------------
# Global configuration & caches
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/params.yaml"
config = None
model = None
processed_data_cache = None
events_data = None
example_user_ids = None
last_processed_time = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    logging.info("Loading configuration...")
    config = load_config(CONFIG_PATH)
    logging.info("Configuration loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load configuration {CONFIG_PATH}: {e}")
    exit()

# ---------------------------------------------------------------------------
# Helper loaders
# ---------------------------------------------------------------------------


def load_example_user_ids():
    global example_user_ids, config
    if example_user_ids is None:
        logging.info("Loading user IDs from test.csv...")
        try:
            test_path = os.path.join(config["data"]["raw_dir"], config["data"]["test_csv"])
            test_df_ids = pd.read_csv(test_path, usecols=[config["features"]["user_id_col"]])
            example_user_ids = sorted(test_df_ids[config["features"]["user_id_col"]].unique().tolist())
            MAX_EXAMPLES = 1000
            if len(example_user_ids) > MAX_EXAMPLES:
                example_user_ids = example_user_ids[:MAX_EXAMPLES]
            logging.info(f"Loaded {len(example_user_ids)} unique user IDs.")
        except FileNotFoundError:
            logging.error("test.csv not found.")
            example_user_ids = []
        except Exception as e:
            logging.error(f"Error loading user IDs: {e}")
            example_user_ids = []
    return example_user_ids


def load_event_details():
    global events_data, config
    if events_data is None:
        logging.info("Loading event details (events.csv)...")
        try:
            events_path = os.path.join(config["data"]["raw_dir"], config["data"]["events_csv"])
            compression = "gzip" if config["data"]["events_csv"].endswith(".gz") else None
            cols_to_load = ["event_id", "start_time", "city", "state", "country"]
            events_data = pd.read_csv(
                events_path, compression=compression, usecols=cols_to_load, dtype={"event_id": str}
            )
            events_data["start_time"] = preprocess_datetime(events_data, "start_time")
            events_data.set_index("event_id", inplace=True)
            logging.info(f"Loaded details for {len(events_data)} events.")
        except FileNotFoundError:
            logging.error("events.csv not found.")
            events_data = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading event details: {e}")
            events_data = pd.DataFrame()
    return events_data


def load_resources():
    global model, config
    load_event_details()
    if model is None:
        model_path = os.path.join(config["output"]["model_dir"], config["output"]["model_name"])
        logging.info(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = load_object(model_path)
        logging.info("Model loaded successfully.")


def get_processed_data(force_reload=False):
    global processed_data_cache, last_processed_time, config
    current_time = time.time()
    cache_ttl = 3600
    if force_reload or processed_data_cache is None or (current_time - last_processed_time > cache_ttl):
        logging.info("Refreshing processed-data cache...")
        raw_datasets = load_datasets(config)
        required_keys = ["users", "events", "event_attendees", "test"]
        if not all(k in raw_datasets for k in required_keys):
            missing = [k for k in required_keys if k not in raw_datasets]
            raise ValueError(f"Missing raw datasets: {missing}")
        processed_datasets = create_features(raw_datasets, config)
        if "test" not in processed_datasets:
            raise KeyError("Key 'test' missing in processed datasets.")
        processed_data_cache = processed_datasets["test"].copy()
        last_processed_time = current_time
        logging.info("Processed data cache updated.")
    user_col = config["features"]["user_id_col"]
    event_col = config["features"]["event_id_col"]
    if user_col not in processed_data_cache.columns or event_col not in processed_data_cache.columns:
        raise KeyError(f"ID columns '{user_col}' or '{event_col}' missing in processed data.")
    return processed_data_cache


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------


def get_recommendations_for_gradio(user_id_input, progress=gr.Progress(track_tqdm=True)):
    global config, model, events_data
    empty_df = pd.DataFrame(columns=["#", "Event ID", "Location", "Start Time", "Score"])
    try:
        progress(0, desc="Validating user ID...")
        if user_id_input is None or user_id_input == "":
            return empty_df
        try:
            user_id = int(user_id_input)
        except (ValueError, TypeError):
            return empty_df

        progress(0.15, desc="Loading processed data...")
        prediction_df = get_processed_data()

        progress(0.35, desc="Filtering user events...")
        user_col = config["features"]["user_id_col"]
        event_col = config["features"]["event_id_col"]
        user_data = prediction_df[prediction_df[user_col] == user_id]
        if user_data.empty:
            return empty_df

        progress(0.55, desc="Running model inference...")
        selected_features = config["features"]["selected_features"]
        missing_features = [f for f in selected_features if f not in user_data.columns]
        if missing_features:
            logging.error(f"Missing features: {missing_features}")
            return empty_df
        X_user = user_data[selected_features].fillna(0)
        probabilities = model.predict_proba(X_user)[:, 1]

        progress(0.75, desc="Ranking recommendations...")
        top_n = config["recommend"]["top_n"]
        recommendation_df = pd.DataFrame(
            {event_col: user_data[event_col].astype(str), "probability": probabilities}
        ).sort_values(by="probability", ascending=False).head(top_n)

        progress(0.9, desc="Formatting results...")
        rows = []
        for rank, (_, row) in enumerate(recommendation_df.iterrows(), start=1):
            event_id = row[event_col]
            score = f"{row['probability']:.3f}"
            location = "N/A"
            start_time_str = "N/A"
            if events_data is not None and not events_data.empty and event_id in events_data.index:
                evt = events_data.loc[event_id]
                city = evt.get("city")
                state = evt.get("state")
                if pd.isna(city) and pd.isna(state):
                    location = "Unknown"
                elif pd.isna(state):
                    location = str(city)
                elif pd.isna(city):
                    location = str(state)
                else:
                    location = f"{city}, {state}"
                if pd.notna(evt["start_time"]):
                    start_time_str = evt["start_time"].strftime("%Y-%m-%d %H:%M")
            rows.append(
                {
                    "#": rank,
                    "Event ID": event_id,
                    "Location": location,
                    "Start Time": start_time_str,
                    "Score": score,
                }
            )

        progress(1, desc="Done!")
        return pd.DataFrame(rows)

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        return empty_df
    except Exception as e:
        logging.exception("Unexpected error during recommendation.")
        return empty_df


# ---------------------------------------------------------------------------
# Load static metrics for the Model Performance tab
# ---------------------------------------------------------------------------

METRICS_PATH = "reports/evaluation_metrics.json"
_metrics: dict = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH) as fh:
            _metrics = json.load(fh)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

example_ids = load_example_user_ids()

custom_css = ".gradio-container { max-width: 1080px !important; margin: auto !important; }"

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Event Recommendation System
        Select a user ID from the dropdown (or type your own) to get personalised
        top-5 event recommendations powered by a **LightGBM** classifier.

        > Dataset: [Kaggle — Event Recommendation Engine Challenge](https://www.kaggle.com/c/event-recommendation-engine-challenge)
        """
    )

    with gr.Tabs():

        # ── Tab 1: Recommendations ──────────────────────────────────────────
        with gr.Tab("Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    user_id_input = gr.Dropdown(
                        label="Select or enter User ID",
                        choices=example_ids,
                        allow_custom_value=True,
                        filterable=True,
                    )
                    submit_button = gr.Button("Get Recommendations", variant="primary")
                    gr.Markdown(
                        "_The dropdown lists user IDs from the test set. "
                        "Custom IDs outside this set will return no results._"
                    )

                with gr.Column(scale=2):
                    output_table = gr.DataFrame(
                        label="Recommended Events",
                        headers=["#", "Event ID", "Location", "Start Time", "Score"],
                        interactive=False,
                        wrap=True,
                    )

            submit_button.click(
                fn=get_recommendations_for_gradio,
                inputs=[user_id_input],
                outputs=[output_table],
            )

        # ── Tab 2: Model Performance ─────────────────────────────────────────
        with gr.Tab("Model Performance"):
            gr.Markdown("### Validation Metrics")
            with gr.Row():
                gr.Number(
                    label="ROC AUC",
                    value=_metrics.get("roc_auc", 0),
                    precision=3,
                    interactive=False,
                )
                gr.Number(
                    label="MAP@200",
                    value=_metrics.get("map_at_200", 0),
                    precision=3,
                    interactive=False,
                )
                gr.Number(
                    label="Accuracy",
                    value=_metrics.get("accuracy", 0),
                    precision=3,
                    interactive=False,
                )
                gr.Number(
                    label="F1 Score",
                    value=_metrics.get("f1_score", 0),
                    precision=3,
                    interactive=False,
                )
            with gr.Row():
                gr.Number(
                    label="Precision",
                    value=_metrics.get("precision", 0),
                    precision=3,
                    interactive=False,
                )
                gr.Number(
                    label="Recall",
                    value=_metrics.get("recall", 0),
                    precision=3,
                    interactive=False,
                )
            gr.Markdown(
                """
                **Model:** LightGBM binary classifier with SMOTE oversampling.

                **Features (21):** user age, gender, join month/year, location flag,
                event weekday/hour, weekend flag, geo flag, event popularity counts
                (yes / maybe / invited / no), attendance ratios, hours-to-event,
                minutes-to-event-from-join.

                **Training split:** 80 / 20 stratified, evaluated on held-out validation fold.
                `current_year = 2013` in `config/params.yaml` is intentional — the dataset
                originates from 2013 and age calculation uses that reference year.
                """
            )

        # ── Tab 3: About ─────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## Project Overview

                End-to-end event recommendation pipeline for the Kaggle
                *Event Recommendation Engine Challenge*.

                | Stage | Tool |
                |---|---|
                | Data loading | pandas |
                | Feature engineering | pandas, scikit-learn |
                | Class imbalance | SMOTE (imbalanced-learn) |
                | Model training | LightGBM |
                | Evaluation | scikit-learn, custom MAP@K |
                | Serving | Gradio 5 |

                ### Quickstart

                ```bash
                pip install -r requirements.txt
                python main.py --mode train --config config/params.yaml
                python app.py
                ```

                ### Dataset
                Download from
                [Kaggle](https://www.kaggle.com/c/event-recommendation-engine-challenge/data)
                and place the CSV files in `data/`.
                """
            )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.info("Starting Gradio application...")
    try:
        load_resources()
        demo.launch(theme=gr.themes.Soft(), css=custom_css)
    except FileNotFoundError as e:
        print(f"Critical error — model or data files not found: {e}")
    except Exception as e:
        print(f"Critical error: {e}")
