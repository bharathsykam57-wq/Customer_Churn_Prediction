# Customer Churn Prediction

Predicting customer churn for a telecom company using machine learning. The dataset contains 7043 customers with 20 features. The goal is to identify customers likely to leave before they do, which is more cost-effective than acquiring new ones.

Live demo: https://customerchurnprediction-csuh83vw3c6ygqvygacap4.streamlit.app


## Problem Statement

The dataset has a 73:27 class imbalance — 73% of customers did not churn. A naive model that always predicts "No Churn" achieves 73% accuracy while being completely useless. The real challenge is maximising recall for the minority class (churners) without sacrificing too much precision. This drove most of the modelling decisions.


## What I Built

- Exploratory data analysis to understand churn drivers and clean the data
- A model comparison pipeline testing 13 different approaches to handle class imbalance
- A deployed Streamlit app where you can input customer details and get a churn prediction in real time


## Key Findings from EDA

The features most correlated with churn:

- Month-to-month contract customers churn at 41% vs under 5% for two-year contracts
- Customers in their first 12 months are the highest risk group
- Fiber optic internet service has disproportionately high churn despite being the premium tier
- Electronic check payment method correlates strongly with churn
- No online security and no tech support are strong churn indicators
- Gender has almost no predictive value

The most actionable insight: a new customer on a month-to-month contract with fiber optic and no support services is the highest churn risk profile.


## Modelling Approach

Tried 13 models across different imbalance handling strategies:

| Model | Imbalance Strategy |
|---|---|
| Decision Tree | Baseline, no handling |
| Decision Tree | StandardScaler |
| Decision Tree | SMOTEENN |
| Random Forest (500 trees) | SMOTEENN |
| XGBoost | scale_pos_weight |
| XGBoost | SMOTEENN |
| XGBoost | SMOTE |
| XGBoost | ADASYN |
| AdaBoost | sample_weight |
| Random Forest | class_weight=balanced |
| CatBoost | auto_class_weights=Balanced |
| LightGBM | scale_pos_weight |
| XGBoost | RandomizedSearchCV + Optuna tuning |

The production model is AdaBoost with sample weighting. The minority class (churners) is assigned a weight proportional to the class ratio so the model penalises missing a churner more than a false alarm. This gave the best balance of recall and precision for the business use case.

Model performance on the held-out test set:
- Accuracy: 72%
- Recall for churners: 79%
- Precision for churners: 49%

Recall of 79% is what matters here — the model catches 4 out of 5 actual churners.


## Project Structure

```
customer-churn-prediction/
    data/
        Customer-Churn.csv
    models/
        ada_boost_churn_model.pkl
    notebooks/
        01_EDA.ipynb
        02_ML_Model_Building.ipynb
    app/
        app.py
    requirements.txt
    .gitignore
    README.md
```


## Running Locally

```bash
git clone <your-repo-url>
cd customer-churn-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks in order:

```bash
jupyter notebook
```

Open `notebooks/01_EDA.ipynb` first, then `notebooks/02_ML_Model_Building.ipynb`. The EDA notebook generates `data/tel_churn.csv` and the ML notebook saves the model to `models/`.

Run the app:

```bash
streamlit run app/app.py
```


## Tech Stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, Optuna, Streamlit
