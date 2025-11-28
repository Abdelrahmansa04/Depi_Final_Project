def predict(df, xgb_model, cat_model, feature_columns, ensemble_config):
    # Keep only necessary features
    df_features = df[feature_columns]

    # Individual predictions
    xgb_pred = xgb_model.predict(df_features)
    cat_pred = cat_model.predict(df_features)

    # Weighted ensemble
    weights = ensemble_config.get("weights", {"xgb": 0.5, "cat": 0.5})
    final_pred = weights["xgb"] * xgb_pred + weights["cat"] * cat_pred

    return final_pred
