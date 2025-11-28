import streamlit as st
import pandas as pd
import pickle
import json
import os
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# -----------------------------
# Load Models and Config
# -----------------------------
@st.cache_resource
def load_models():
    # Resolve artifacts directory relative to this file (project root/artifacts)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")

    # Load XGBoost
    xgb = XGBRegressor()
    xgb_path = os.path.join(artifacts_dir, "xgboost_model_20251122_103557.json")
    xgb.load_model(xgb_path)

    # Load CatBoost
    cat = CatBoostRegressor()
    cat_path = os.path.join(artifacts_dir, "catboost_model_20251122_103557.cbm")
    cat.load_model(cat_path)

    # Load feature columns
    features_path = os.path.join(artifacts_dir, "feature_columns_20251122_103557.pkl")
    with open(features_path, "rb") as f:
        feature_columns = pickle.load(f)

    # Load ensemble config
    ensemble_path = os.path.join(artifacts_dir, "ensemble_config_20251122_103557.json")
    with open(ensemble_path, "r") as f:
        ensemble_config = json.load(f)

    return xgb, cat, feature_columns, ensemble_config


xgb_model, cat_model, feature_columns, ensemble_config = load_models()


# -----------------------------
# Prediction Function
# -----------------------------
def predict(df):
    # Make a copy
    df = df.copy()
    
    # Check if any feature is missing
    missing = [c for c in feature_columns if c not in df.columns]
    
    # Attempt to compute features from 'date' and 'sales'
    if missing:
        if "date" not in df.columns or "sales" not in df.columns:
            st.warning(
                f"Missing required features: {missing}. Cannot derive because 'date' or 'sales' column is missing."
            )
        else:
            st.info("Computing lag and rolling features automatically...")
            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            # Compute lags
            for lag in [1, 7, 14, 30]:
                df[f"sales_lag_{lag}"] = df["sales"].shift(lag)
            
            # Compute rolling mean and std
            for window in [7, 14, 30]:
                df[f"sales_rolling_mean_{window}"] = df["sales"].rolling(window).mean()
                df[f"sales_rolling_std_{window}"] = df["sales"].rolling(window).std()
            
            # Re-check missing
            missing = [c for c in feature_columns if c not in df.columns]
    
    # Fill any remaining missing features with 0
    for col in missing:
        df[col] = 0

    # Keep only required features
    df_features = df[feature_columns].fillna(0)

    # Predict with both models
    xgb_pred = xgb_model.predict(df_features)
    cat_pred = cat_model.predict(df_features)

    # Weighted ensemble
    weights = ensemble_config.get("weights", {"xgb": 0.5, "cat": 0.5})
    final_pred = weights["xgb"] * xgb_pred + weights["cat"] * cat_pred

    return final_pred


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 Sales Forecasting App")
st.write("Upload your dataset (CSV) and get predicted sales instantly.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data")
        st.dataframe(df)

        # Predict
        st.subheader("📌 Forecasted Sales")
        predictions = predict(df)
        df["Predicted_Sales"] = predictions
        st.dataframe(df)

        # Download CSV
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df(df)
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name="sales_predictions.csv",
            mime="text/csv",
        )

        st.success("Prediction completed successfully!")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to start.")
