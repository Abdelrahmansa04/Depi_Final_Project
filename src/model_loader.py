import pickle
import json
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from .config import XGB_MODEL_PATH, CAT_MODEL_PATH, FEATURES_PATH, ENSEMBLE_CONFIG_PATH

def load_models():
    # Load XGBoost
    xgb = XGBRegressor()
    xgb.load_model(XGB_MODEL_PATH)

    # Load CatBoost
    cat = CatBoostRegressor()
    cat.load_model(CAT_MODEL_PATH)

    # Load feature columns
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    # Load ensemble config
    with open(ENSEMBLE_CONFIG_PATH, "r") as f:
        ensemble_config = json.load(f)

    return xgb, cat, feature_columns, ensemble_config
