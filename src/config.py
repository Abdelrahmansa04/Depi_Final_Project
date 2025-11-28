import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "artifacts")

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model_20251122_103557.json")
CAT_MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model_20251122_103557.cbm")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns_20251122_103557.pkl")
ENSEMBLE_CONFIG_PATH = os.path.join(MODEL_DIR, "ensemble_config_20251122_103557.json")
