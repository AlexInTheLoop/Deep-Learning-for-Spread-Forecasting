import pandas as pd
import numpy as np

def getBeta(series, sl):
    if not isinstance(series, pd.DataFrame) or set(series.columns) != {"high","low"}:
        raise ValueError("getBeta attend un DataFrame avec colonnes ['high','low']")
    arr = series[["high","low"]].to_numpy(dtype=float)  
    ratios = arr[:,0] / arr[:,1]
    log_hl = np.log(ratios)       
    hl_sq = log_hl**2
    hl_series = pd.Series(hl_sq, index=series.index)
    beta = hl_series.rolling(2).sum()
    beta = beta.rolling(sl).mean()
    return beta.dropna()

def getGamma(series):
    if not isinstance(series, pd.DataFrame) or set(series.columns) != {"high","low"}:
        raise ValueError("getGamma attend un DataFrame avec colonnes ['high','low']")
    h2 = series["high"].rolling(2).max()
    l2 = series["low"].rolling(2).min()
    h2_arr = h2.to_numpy(dtype=float)
    l2_arr = l2.to_numpy(dtype=float)
    ratios = h2_arr / l2_arr
    log_rl = np.log(ratios)
    gamma_vals = log_rl**2
    gamma_series = pd.Series(gamma_vals, index=series.index)
    return gamma_series.dropna()


def getAlpha(beta, gamma, use_opposed=False):
    """
    Combine β et γ pour estimer α (voir Corwin & Schultz, p.727),
    et prend la moyenne pour éviter les α négatifs si use_opposed=False.
    """
    if not (isinstance(beta, pd.Series) and isinstance(gamma, pd.Series)):
        raise ValueError("getAlpha attend deux Series Pandas (beta, gamma).")
    idx = beta.index.intersection(gamma.index)
    if idx.empty:
        raise ValueError("getAlpha : β et γ n'ont aucun index en commun.")
    beta_cut = beta.loc[idx].to_numpy(dtype=float)
    gamma_cut = gamma.loc[idx].to_numpy(dtype=float)
    den = 3 - 2 * 2**0.5
    alpha_vals = (2**0.5 - 1) * np.sqrt(beta_cut) / den - np.sqrt(gamma_cut / den)
    if not use_opposed:
        alpha_vals[alpha_vals < 0] = 0.0
    else:
        alpha_vals = np.abs(alpha_vals)

    return pd.Series(alpha_vals, index=idx)

def corwinSchultz(series, sl=1, use_oposed=False):
    """
    Implémente l'estimateur Corwin & Schultz (2012) à partir des prix High/Low.
    series : DataFrame Pandas avec EXACTEMENT deux colonnes "High" et "Low", indexées par minute.
    sl : fenêtre de lissage (typiquement 1).
    use_oposed : bool, si True on autorise alpha négatif.
    Renvoie la moyenne de la colonne "Spread" calculée.
    """
    if not isinstance(series, pd.DataFrame):
        raise ValueError(f"corwinSchultz attend un DataFrame, reçu {type(series)}")
    if set(series.columns) != {"high", "low"}:
        raise ValueError(f"corwinSchultz attend colonnes ['high','low'], reçu {series.columns.tolist()}")
    beta  = getBeta(series, sl)
    gamma = getGamma(series)
    alpha = getAlpha(beta, gamma, use_oposed)
    exp_a = np.exp(alpha.to_numpy(dtype=float))
    spread_vals = 2 * (exp_a - 1) / (1 + exp_a)
    spread_series = pd.Series(spread_vals, index=alpha.index)
    start_times = pd.Series(series.index[0 : spread_series.shape[0]], index=spread_series.index)
    df_out = pd.concat([spread_series, start_times], axis=1)
    df_out.columns = ["Spread", "Start_Time"]

    return df_out["Spread"].mean()

# dico_sim=simu.HF_simulation(50,3/100,0.25/100,10)
# tab_spread = []
# for day in dico_sim.keys():
#     series_hl = dico_sim[day][["High", "Low"]].astype(float)
#     tab_spread.append(corwinSchultz(series_hl)) 