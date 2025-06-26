from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def compute_model_metrics(model, X_test, y_test, y_scaler=None, name=None):
    """
    Calcule les métriques d'évaluation pour le modèle sur l'ensemble de test.
    """
    y_pred = model.predict(X_test)

    if y_scaler:
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    if name:
        print(f"Métrique du modèle {name}:")
    else:
        print("Métriques:")
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return r2, rmse, mae

def compile_models_metrics(models, X_test, y_test, y_scaler=None):
    """
    Compile les métriques pour plusieurs modèles.
    """
    metrics = {}
    for i, (name, model) in enumerate(models.items()):
        metrics[name] = compute_model_metrics(model, X_test[i], y_test[i], y_scaler, name)
    return pd.DataFrame(metrics, index=['R²', 'RMSE', 'MAE', 'Score']).T

def compute_estimators_metrics(df_estimators, 
                                y_true, 
                                sort_mode = "asset_first"):
    """
    Calcule les métriques R², RMSE, MAE et Score pour chaque estimateur de spread.
    """
    df = df_estimators.copy()

    for time_col in ["day", "date"]:
        if time_col in df.columns:
            df = df.drop(columns=[time_col])

    if sort_mode not in {"asset_first", "day_first"}:
        raise ValueError("sort_mode doit être 'asset_first' ou 'day_first'")

    if sort_mode == "day_first":
        nb_assets = len(df) // len(y_true)
        if len(df) % len(y_true) != 0:
            raise ValueError("Impossible de déterminer nb_assets depuis les tailles de df et y_true")
        idx = np.arange(len(df)).reshape(len(y_true), nb_assets).T.flatten()
        df = df.iloc[idx].reset_index(drop=True)

    if df.shape[0] != len(y_true):
        raise ValueError(f"Incohérence de taille : {df.shape[0]} estimateurs vs {len(y_true)} y_true")

    results = {}
    for col in df.columns:
        y_pred = df[col].values
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        results[col] = {"R²": r2, "RMSE": rmse, "MAE": mae}

    return pd.DataFrame(results)
