import pandas as pd
import numpy as np

from parametric_estimators import (
    AR_estimator,
    ardia_estimator,
    CS_estimator,
    grb_estimator,
    grb_estimator2,
    grb_estimator3,
    grb_estimator4,
    roll_estimator,
    hurst_estimator
)

def get_parametric_estimators_series(X, meta, use_opposed):
    """
    Calcule les estimateurs de spreads à partir de X minute-level pour un actif
    (X = valeurs de df avec colonnes: date, open, high, low, close, volume).
    """
    minutes_per_day = 1440
    _, d = X.shape

    if not {"day", "symbol"}.issubset(meta.columns):
        raise ValueError("meta doit contenir 'day' et 'symbol'.")

    df_full = pd.DataFrame(X, columns=["date", "open", "high", "low", "close", "volume"])
    df_full["day"] = meta["day"].values

    list_estimators = [
        "S1", "S2", "S3", "S4",
        "Roll Spread", "CS Spread",
        "AGK1", "AGK2", "AR Spread"
    ]

    results = []
    for day in sorted(df_full["day"].unique()):
        df_day = df_full[df_full["day"] == day]
        if len(df_day) < minutes_per_day:
            print(f"Données incomplètes pour le jour {day} → ignoré.")
            continue

        ohlc = df_day[["open", "high", "low", "close"]].values[:minutes_per_day]
        df_ohlc = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
        est = compute_parametric_estimators(df_ohlc, use_opposed)
        results.append([day] + list(est))

    return pd.DataFrame(results, columns=["day"] + list_estimators)


def compute_parametric_estimators(data, use_opposed=False):
    """
    Calcule les estimateurs paramètriques du spread pour une journée donnée
    à partir des klines 1 minute
    """
    estimators = np.zeros(9, dtype=float)

    S1 = S2 = S3 = S4 = 0.0
    Roll_spread = CS_spread = AGK1 = AGK2 = AR_spread = 0.0

    try:
        close_arr = np.asarray(data["close"], dtype=float)
        open_arr  = np.asarray(data["open"],  dtype=float)
        high_arr  = np.asarray(data["high"],  dtype=float)
        low_arr   = np.asarray(data["low"],   dtype=float)
    except Exception as e:
        print("Erreur de colonne ou de contenu dans data", e)
        return np.zeros(9, dtype=float)

    if close_arr.size == 0 or open_arr.size == 0 or high_arr.size == 0 or low_arr.size == 0:
        print("data est vide pour cette journée.")
        return np.zeros(9, dtype=float)

    if np.any(close_arr <= 0) or np.any(open_arr <= 0) or np.any(high_arr <= 0) or np.any(low_arr <= 0):
        print("valeur négative détectée dans data.")
        return np.zeros(9, dtype=float)

    if np.isnan(close_arr).any() or np.isnan(open_arr).any() or np.isnan(high_arr).any() or np.isnan(low_arr).any():
        print("un NaN détecté dans data.")
        return np.zeros(9, dtype=float)

    try:
        log_close = np.log(close_arr)
        log_open  = np.log(open_arr)
        mid_arr   = (high_arr + low_arr) / 2.0
        log_mid   = np.log(mid_arr)
    except Exception as e:
        print("Erreur lors du calcul des log-prices : ", e)
        return np.zeros(9, dtype=float)

    try:
        H = hurst_estimator.estimateur_Hurst(log_close, L=5)
    except Exception as e:
        print("Erreur dans estimateur_Hurst : ", e)
        H = 0.0

    try:
        S1 = grb_estimator.estimateur_grb1_close(log_close, 10, use_opposed)
    except Exception as e:
        print("Erreur avec estimateur_grb1_close : ", e)
        S1 = 0.0

    try:
        S2 = grb_estimator2.estimateur_grb2_close(log_close, 10, H, use_opposed)
    except Exception as e:
        print("Erreur avec estimateur_grb2_close : ", e)
        S2 = 0.0

    try:
        S3 = grb_estimator3.estimateur_grb3_close(log_close, 10, use_opposed)
    except Exception as e:
        print("Erreur avec estimateur_grb3_close : ", e)
        S3 = 0.0

    try:
        tmp = grb_estimator4.estimateur_grb4_close(log_close, 10)
        S4 = tmp["Spread"] if isinstance(tmp, dict) and "Spread" in tmp else float(tmp)
    except Exception as e:
        print("Erreur avec estimateur_grb4_close : ", e)
        S4 = 0.0

    try:
        Roll_spread = roll_estimator.Roll_estimateur(log_close, use_opposed)
    except Exception as e:
        print("Erreur avec Roll_estimateur : ", e)
        Roll_spread = 0.0

    try:
        CS_spread = CS_estimator.corwinSchultz(
            data[["high", "low"]], 
            sl=1, 
            use_oposed=use_opposed
            )
    except Exception as e:
        print("Erreur avec corwinSchultz : ", e)
        CS_spread = 0.0

    try:
        AGK1 = ardia_estimator.edge_close_price(log_close, use_opposed)
    except Exception as e:
        print("Erreur avec edge_close_price : ", e)
        AGK1 = 0.0

    try:
        AGK2 = ardia_estimator.edge(open_arr, high_arr, low_arr, close_arr, use_opposed)
    except Exception as e:
        print("Erreur avec edge : ", e)
        AGK2 = 0.0

    try:
        AR_spread = AR_estimator.AR_estimateur(log_close, log_mid, use_opposed)
    except Exception as e:
        print("Erreur avec AR_estimator : ", e)
        AR_spread = 0.0

    estimators[:] = [
        S1, S2, S3, S4, Roll_spread, CS_spread, AGK1, AGK2, AR_spread
    ]

    return estimators