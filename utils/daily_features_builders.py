import numpy as np
import pandas as pd
import requests

from parametric_estimators.hurst_estimator import estimateur_Hurst


def get_serial_dependancy_features_v2(df_features: pd.DataFrame)->pd.DataFrame:
    """
    Méthode permettant de calculer et d'ajouter des features de dépendance sérielle pour enrichir l'estimation des modèles
    """

    # Création d'un dataframe pour stocker les features de dépendance sérielle
    df_serial_dependance: pd.DataFrame = pd.DataFrame()

    # Récupération de la série des prix close et passage en log
    close_arr = df_features["close"]
    log_close = np.log(close_arr)

    # Première feature : incrément du prix, en logarithme
    delta1 = np.zeros_like(log_close)
    delta1[1:] = log_close[1:] - log_close[:-1]
    df_serial_dependance["delta1"] = delta1

    # 2eme feature : convariance non laggées
    cov1 = np.zeros_like(log_close)
    cov1[2:] = (log_close[2:] - log_close[1:-1]) * (log_close[1:-1] - log_close[:-2])
    df_serial_dependance["cov1"] = cov1

    # 3eme feature : covariance avec un lag
    cov2 = np.zeros_like(log_close)
    if len(log_close) >= 5:
        # On calcule pour i>=4
        for i in range(4, len(log_close)):
            cov2[i] = (log_close[i] - log_close[i-2]) * (log_close[i-2] - log_close[i-4])
    df_serial_dependance["cov2"] = cov2

    # 4eme feature : variance glissante sur 10 minutes des incréments delta1
    var10 = np.zeros_like(delta1)
    window = 10
    for i in range(window, len(delta1)):
        var10[i] = np.var(delta1[i-window:i])
    mu_var10 = var10.mean()
    sigma_var10 = var10.std() if var10.std() > 0 else 1.0
    var10 = (var10 - mu_var10) / sigma_var10
    df_serial_dependance["var10"] = var10

    # Concaténation avec le dataframe de features pris en entrée
    df_all_features: pd.DataFrame = pd.concat([df_features, df_serial_dependance], axis=1)
    return df_all_features

def get_serial_dependency_features(date, group, features):
    try:
        close = group["close"].to_numpy()
        high = group["high"].to_numpy()
        low = group["low"].to_numpy()
        volume = group["volume"].to_numpy()
        
        log_ret = np.log(close[1:] / close[:-1])
        
        for lag in [1, 5, 10, 30]:
            if len(log_ret) > lag:
                features[f'autocorr_return_lag{lag}'] = np.corrcoef(log_ret[lag:], log_ret[:-lag])[0, 1]
            else:
                features[f'autocorr_return_lag{lag}'] = np.nan
        
        vol_changes = np.diff(volume)
        for lag in [1, 5, 10, 30]:
            if len(vol_changes) > lag:
                features[f'autocorr_volume_lag{lag}'] = np.corrcoef(vol_changes[lag:], vol_changes[:-lag])[0, 1]
            else:
                features[f'autocorr_volume_lag{lag}'] = np.nan
        
        if len(log_ret) >= 100:
            features['hurst_exponent'] = estimateur_Hurst(log_ret,L=5)
        else:
            features['hurst_exponent'] = np.nan
        
        abs_returns = np.abs(log_ret)
        for lag in [1, 5, 10]:
            if len(abs_returns) > lag:
                features[f'vol_cluster_lag{lag}'] = np.corrcoef(abs_returns[lag:], abs_returns[:-lag])[0, 1]
            else:
                features[f'vol_cluster_lag{lag}'] = np.nan
        
        for window in [10, 30, 60, 120]:
            if len(close) > window:
                features[f'momentum_{window}m'] = (close[-1] / close[-window-1]) - 1
            else:
                features[f'momentum_{window}m'] = np.nan
        
        for window in [30, 60, 120, 240]:
            if len(log_ret) >= window:
                features[f'realized_vol_{window}m'] = np.std(log_ret[-window:]) * np.sqrt(1440)  # Annualized
            else:
                features[f'realized_vol_{window}m'] = np.nan
        
        hl_range = (high - low) / close
        for lag in [1, 5, 10]:
            if len(hl_range) > lag:
                features[f'hl_range_persist_lag{lag}'] = np.corrcoef(hl_range[lag:], hl_range[:-lag])[0, 1]
            else:
                features[f'hl_range_persist_lag{lag}'] = np.nan
        
        for lag in [0, 1, 5]:
            if len(log_ret) > lag and lag < len(volume):
                vol_slice = volume[lag:] if lag > 0 else volume[:len(log_ret)]
                ret_slice = log_ret[:len(vol_slice)]
                features[f'vol_ret_corr_lag{lag}'] = np.corrcoef(vol_slice, ret_slice)[0, 1]
            else:
                features[f'vol_ret_corr_lag{lag}'] = np.nan
        
        if len(close) >= 720:
            morning_ret = np.log(close[360] / close[0])
            afternoon_ret = np.log(close[720] / close[360])
            features['morning_afternoon_ret_ratio'] = morning_ret / afternoon_ret if afternoon_ret != 0 else np.nan
            features['morning_vol'] = np.sum(volume[:360])
            features['afternoon_vol'] = np.sum(volume[360:720])
            features['vol_morning_afternoon_ratio'] = features['morning_vol'] / features['afternoon_vol'] if features['afternoon_vol'] > 0 else np.nan
        
        hour_returns = []
        for hour in range(24):
            start_idx = hour * 60
            end_idx = (hour + 1) * 60
            if start_idx < len(close) and end_idx <= len(close):
                hour_ret = np.log(close[end_idx-1] / close[start_idx])
                hour_returns.append(hour_ret)
                features[f'hour_{hour}_return'] = hour_ret
            else:
                features[f'hour_{hour}_return'] = np.nan
        
        if len(log_ret) > 1:
            sign_changes = np.sum(np.sign(log_ret[1:]) != np.sign(log_ret[:-1])) / (len(log_ret) - 1)
            features['return_sign_changes'] = sign_changes
        
        if len(close) > 0:
            features['time_at_daily_high'] = np.argmax(high) / len(high) if len(high) > 0 else np.nan
            features['time_at_daily_low'] = np.argmin(low) / len(low) if len(low) > 0 else np.nan
        
    except Exception as e:
        print(f"Erreur calcul features séries temporelles pour {date}: {e}")
    
    return features