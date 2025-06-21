import numpy as np
import requests

from parametric_estimators.hurst_estimator import estimateur_Hurst


def get_basic_features(symbol, date, group):
    try:
        close = group["close"].to_numpy()
        open = group["open"].iloc[0]
        close_last = group["close"].iloc[-1]
        high = group["high"].max()
        low = group["low"].min()
        volume_total = group["volume"].sum()
        log_ret = np.log(close[1:] / close[:-1])
        volatility = np.std(log_ret)
        max_dd = np.min(close / np.maximum.accumulate(close) - 1)
        vol_opening = group.iloc[:30]["volume"].sum()
        vol_closing = group.iloc[-30:]["volume"].sum()
        x = np.arange(len(close))
        slope = np.polyfit(x, close, 1)[0]
        above_open = np.mean(close > open)
        below_open = np.mean(close < open)
    except Exception as e:
        print(f"Erreur traitement {date} : {e}")
    
    tick_size = np.nan
    try:
        exchange_info = requests.get(
            f"https://fapi.binance.com/fapi/v1/exchangeInfo?symbol={symbol}"
        ).json()
        for f in exchange_info["symbols"][0]["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
                break
    except Exception as e:
        print(f"⚠️ Erreur récupération tick_size : {e}")
    
    row = {
            "day": date.day,
            "daily_return": np.log(close_last / open),
            "daily_range": (high - low) / close_last,
            "volatility": volatility,
            "max_drawdown": max_dd,
            "volume_total": volume_total,
            "volume_opening": vol_opening,
            "volume_closing": vol_closing,
            "trend_slope": slope,
            "pct_time_above_open": above_open,
            "pct_time_below_open": below_open,
            "tick_size": tick_size
        }
    return row

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