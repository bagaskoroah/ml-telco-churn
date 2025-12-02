# ====================
# Paths
# ====================
DATA_PATH = 'data/raw/telco-customer-churn.csv'
PROCESSED_PATH = 'data/processed/cleaned.csv'
MODEL_PATH = 'outputs/final_model.pkl'

# ====================
# Data Columns
# ====================
TARGET = 'Churn'

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

# ====================
# Train Test Split
# ====================
TEST_SIZE = 0.2
RANDOM_STATE = 123

# ====================
# Model Parameters
# ====================
XGBOOST_PARAMS = {
    'model__n_estimators': [100, 200, 500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 10],
    'model__reg_lambda': [0.5, 1, 2],
    'model__reg_alpha': [0, 0.1, 0.5]
    }

CATBOOST_PARAMS = {
    'model__iterations': [200, 500, 800],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__depth': [4, 6, 8, 10]
    }

LIGHTGBM_PARAMS = {
    'model__n_estimators': [200, 400, 600, 800],
    'model__learning_rate': [0.005, 0.01, 0.05, 0.1],
    'model__max_depth': [-1, 5, 10, 15],
    'model__reg_alpha': [0, 0.1, 0.5, 1],
    'model__reg_lambda': [0, 0.1, 0.5, 1]
}

# ====================
# CV arguments
# ====================
VERBOSE = 0
N_JOBS = -1
N_ITER = 40
CV = 5