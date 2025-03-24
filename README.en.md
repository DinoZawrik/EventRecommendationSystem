
# Project: Event Recommendation System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.1+-green.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.23+-orange.svg)](https://scikit-learn.org/stable/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.0+-lightgreen.svg)](https://lightgbm.readthedocs.io/en/latest/)

## Project Description

This project implements an **event recommendation system** designed to predict user interest in online events. The system leverages machine learning to personalize recommendations, based on historical data of user-event interactions, user demographics, and event characteristics.

**Project Goal:**

To develop and evaluate a recommendation system capable of effectively ranking events for each user, increasing the likelihood of their interest and improving user experience in an event discovery application.

**Key Achievements:**

* **Complete Data Science Pipeline:** Implementation of all stages from data loading and Exploratory Data Analysis (EDA) to Feature Engineering, machine learning model training, and submission file creation.
* **Effective Feature Engineering:** Creation of informative features based on time, event popularity, and user demographics, enhancing prediction quality.
* **Machine Learning Model Training and Comparison:**  Utilization of Logistic Regression, Random Forest, and LightGBM models, identifying LightGBM as the most effective for this task.
* **MAP@200 Metric Implementation:** Integration of the Mean Average Precision at 200 (MAP@200) metric to evaluate the ranking quality of recommendations, achieving a MAP@200 score of **0.3285** on the validation set.
* **`recommend_events()` Recommendation Function:** Development of a function to generate personalized event recommendations for a specific user, leveraging the trained LightGBM model.

## Technologies Used

* **Python:** Primary programming language.
* **pandas:** For data processing and DataFrame manipulation.
* **matplotlib and seaborn:** For data visualization and EDA.
* **scikit-learn:** Machine learning library for models (Logistic Regression, RandomForest), data splitting, and evaluation metrics.
* **LightGBM:** Gradient Boosting Machine library used for training the main model.
* **imblearn (imbalanced-learn):** For handling class imbalance using SMOTE.

## Data

The project utilizes data from the **"Event Recommendation Engine Challenge" Kaggle competition:**

[Kaggle Competition Link](https://www.kaggle.com/competitions/event-recommendation-engine-challenge)

The dataset includes the following files:

* `train.csv`: Training data about user-event interactions.
* `test.csv`: Test data for which recommendations need to be made.
* `users.csv`: Demographic information about users.
* `events.csv`: Information about events and their characteristics.
* `event_attendees.csv`: Data on event attendance (user responses to invitations).
* `user_friends.csv`: Information about user social connections (friend lists).

To run the code, download the dataset and place the CSV files in a `data` folder in the project root directory.

## Feature Engineering

Key Feature Engineering steps included:

* **Time-based Features:**
    * `hrs_to_event`: Hours until the event starts from the time of user viewing.
    * `minsToEvent_frmJoin`: Minutes from user joining time to event viewing time.
    * `event_start_weekday`, `event_start_hour`, `user_joined_month`, `user_joined_year`: Day of the week, hour of event start, month and year of user joining.
* **Event Popularity Features:**
    * `yes_count`, `maybe_count`, `invited_count`, `no_count`: Counts of responses to event invitations.
    * `yes_ratio`, `yes_vs_no_ratio`, `attendance_rate`: Popularity ratios calculated from response counts.
* **Binary Features:**
    * `is_weekend_event`: Indicates if the event day is a weekend.
    * `has_geo_info`, `has_location`: Indicate the presence of geographic information for the event and location information for the user.
    * `invited_1`, `gender_male`, `has_location_1`, `is_weekend_event_1`, `has_geo_info_1`: Results of One-Hot Encoding for categorical features.
* **Demographic Features:**
    * `age`, `gender_male`, `has_location_1`, `user_joined_month`, `user_joined_year`: Age, gender, presence of location information, and user joining time.

## Machine Learning Models

The following models were trained and evaluated:

* **Logistic Regression:** Used as a baseline model.
* **Random Forest:** A more complex model based on decision trees.
* **LightGBM (Gradient Boosting Machine):** Gradient Boosting showed the best performance and was selected as the main model for the recommendation system.


## Recommendation Function `recommend_events()`

The project implements a function `recommend_events(user_id, top_n=5)` that retrieves a list of top-N recommended events for a given user. The function uses the trained LightGBM model to predict the probability of user interest in each event and ranks events based on these probabilities.

**Example Usage of Recommendation Function:**

```python
user_id_example = 3044012  # Example User ID
recommendations = recommend_events(user_id_example, top_n=5)
print(f"Recommendations for User ID {user_id_example}: {recommendations}")
```

## MAP@200 Evaluation Metric

The **Mean Average Precision at 200 (MAP@200)** metric was used to evaluate the ranking quality of recommendations. The MAP@200 score achieved on the validation set using the LightGBM model is **0.3285**.

MAP@200 is a suitable metric for evaluating recommendation systems as it considers the order of recommendations and focuses on accuracy in the top-k positions of the recommendation list, which is particularly important for tasks where users typically view only the first few suggested options.

## Setup and Run Instructions

**Prerequisites:**

* **Python 3.7 or higher**
* **Installed Python Libraries:** (see `requirements.txt` or install manually)

**Library Installation:**

It is recommended to create a Python virtual environment and install the necessary libraries using `pip`:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate for Windows

pip install -r requirements.txt
```

(The `requirements.txt` file can be created by listing all used libraries or installing them manually: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`, `lightgbm`, `imblearn`)

**Running the Code:**

1. **Download the project repository to your computer.**
2. **Download the "Event Recommendation Engine Challenge" dataset from Kaggle and place the CSV files in a `data` folder in the project root directory.**
3. **Run the Jupyter Notebook `event_recommendation_system.ipynb` or the Python script `event_recommendation_system.py` containing the project code.**

**The `submission.csv` file will be automatically created after running the code.**
