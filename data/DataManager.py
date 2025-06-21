import os
import pandas as pd
import numpy as np
import requests
import zipfile
import tensorflow as tf
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.daily_features_builders import (
    get_basic_features, 
    get_serial_dependency_features
)
from utils.res_comp import get_parametric_estimators_series

RAW_DATA_DIR = "raw data"
LABELS_DATA_DIR = "labels data"
DAILY_FEATURES_DATA_DIR = "daily features data"
MINUTE_FEATURES_DATA_DIR = "minute features data"
PARAMETRIC_ESTIMATORS_DIR = "parametric estimators"


class DataManager:
    def __init__(self, symbols, year, month,light = False):
        self.symbols = [s.upper() for s in symbols]
        self.year = year
        self.month = month
        self.freq = "1m"
        self.nb_assets = len(self.symbols)
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_dir = os.path.join(self.module_dir, RAW_DATA_DIR)
        self.daily_features_data_dir = os.path.join(self.module_dir, DAILY_FEATURES_DATA_DIR)
        self.labels_data_dir = os.path.join(self.module_dir, LABELS_DATA_DIR)
        self.minute_features_data_dir = os.path.join(self.module_dir, MINUTE_FEATURES_DATA_DIR)
        self.parametric_estimators_dir = os.path.join(self.module_dir, PARAMETRIC_ESTIMATORS_DIR)
        self.light = light

        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.daily_features_data_dir, exist_ok=True)
        os.makedirs(self.labels_data_dir, exist_ok=True)
        os.makedirs(self.minute_features_data_dir, exist_ok=True)
        os.makedirs(self.parametric_estimators_dir, exist_ok=True)

        self.kline_base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        self.bookticker_base_url = "https://data.binance.vision/data/futures/um/monthly/bookTicker"

    def download_and_prepare_data(self):
        for symbol in self.symbols:
            month_str = f"{self.month:02d}"
            year_str = str(self.year)

            base_name = f"{symbol}-bookTicker-{year_str}-{month_str}"
            label_path = os.path.join(self.labels_data_dir, f"{base_name}_labels.parquet")

            # KLINES
            kline_filename = f"{symbol}-{self.freq}-{year_str}-{month_str}"
            kline_parquet_path = os.path.join(self.raw_data_dir, f"{kline_filename}.parquet")

            if not os.path.exists(kline_parquet_path):
                kline_zip_url = f"{self.kline_base_url}/{symbol}/{self.freq}/{kline_filename}.zip"
                kline_zip_path = os.path.join(self.raw_data_dir, f"{kline_filename}.zip")
                print(f"⬇️ Téléchargement Klines : {kline_zip_url}")
                response = requests.get(kline_zip_url)
                
                if response.status_code != 200:
                    raise Exception(f"Erreur téléchargement : {kline_zip_url}")
                with open(kline_zip_path, "wb") as f:
                    f.write(response.content)
                with zipfile.ZipFile(kline_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_data_dir)
                os.remove(kline_zip_path)

                csv_path = os.path.join(self.raw_data_dir, f"{kline_filename}.csv")
                df_kline = pd.read_csv(csv_path)
                df_kline.columns = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "nb_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ]
                df_kline.to_parquet(kline_parquet_path, index=False)
                os.remove(csv_path)
                print(f"✅ Klines sauvegardé : {kline_parquet_path}")
            else:
                print(f"⚠️ Klines déjà existant : {kline_parquet_path}")

            # BOOKTICKER
            bookticker_filename = f"{symbol}-bookTicker-{year_str}-{month_str}"
            bookticker_zip_filename = f"{bookticker_filename}.zip"
            bookticker_csv_filename = f"{bookticker_filename}.csv"

            bookticker_zip_url = f"{self.bookticker_base_url}/{symbol}/{bookticker_zip_filename}"

            bookticker_zip_path = os.path.join(self.raw_data_dir, bookticker_zip_filename)
            csv_path = os.path.join(self.raw_data_dir, bookticker_csv_filename)
            bookticker_parquet_path = os.path.join(self.raw_data_dir, f"{bookticker_filename}.parquet")

            if not os.path.exists(bookticker_parquet_path) and not os.path.exists(label_path):
                print(f"⬇️ Téléchargement BookTicker : {bookticker_zip_url}")
                response = requests.get(bookticker_zip_url, stream=True)

                if response.status_code != 200:
                    print(f"⚠️ BookTicker indisponible : {bookticker_zip_url} (code {response.status_code})")
                    continue 

                with open(bookticker_zip_path, "wb") as f:
                    f.write(response.content)

                with zipfile.ZipFile(bookticker_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_data_dir)
                os.remove(bookticker_zip_path)

                if self.light:
                    reader = pd.read_csv(
                    csv_path,
                    usecols=["transaction_time", "best_bid_price", "best_ask_price"],
                    dtype={
                        "transaction_time": np.int64,
                        "best_bid_price": np.float32,
                        "best_ask_price": np.float32
                    },
                    chunksize=1_000_000
                    )
                    df_bt = pd.concat(reader, ignore_index=True)
                else:
                    df_bt = pd.read_csv(csv_path)

                n_rows = len(df_bt)
                if n_rows < 10_000_000:
                    print(f"⚠️ Attention : fichier bookTicker anormalement petit ({n_rows} lignes)")
                df_bt.to_parquet(bookticker_parquet_path, index=False)
                os.remove(csv_path)

                print(f"✅ BookTicker sauvegardé : {bookticker_parquet_path}")
            else:
                print(f"⚠️ BookTicker déjà existant : {bookticker_parquet_path}")

    def build_features(self, serial_dependency = False, daily = True):
        processed_paths = []
        for symbol in self.symbols:
            month_str = f"{self.month:02d}"
            filename = f"{symbol}-{self.freq}-{self.year}-{month_str}"
            parquet_path = os.path.join(self.raw_data_dir, f"{filename}.parquet")
            if daily:
                processed_path = os.path.join(
                    self.daily_features_data_dir, 
                    f"{filename}_features_{serial_dependency}.parquet"
                    )
            else:
                processed_path = os.path.join(
                    self.minute_features_data_dir, 
                    f"{filename}_features_1min_klines.parquet"
                )

            if os.path.exists(processed_path):
                print(f"✅ Fichier déjà existant, ignoré : {processed_path}")
                processed_paths.append(processed_path)
                continue
            print(f"📊 Construction des features pour : {symbol} {self.year}-{month_str}")
            df = pd.read_parquet(parquet_path)
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
            df["date"] = df["datetime"].dt.date
            df["close"] = df["close"].astype(float)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["volume"] = df["volume"].astype(float)

            rows = []
            if daily:
                for date, group in df.groupby("date"):
                    if len(group) < 1440:
                        continue
                    features_row = get_basic_features(symbol, date, group)
                    if serial_dependency:
                        features_row = get_serial_dependency_features(date, group, features_row)
                    rows.append(features_row)
                df_out = pd.DataFrame(rows)
                df_out = df_out.fillna(0)
                df_out.to_parquet(processed_path, index=False)
            else:
                df_temp = df[["date", "open", "high", "low", "close", "volume"]].copy()
                df_temp = df_temp.dropna()
                df_temp["return"] = df_temp["close"].pct_change().fillna(0)
                df_temp["log_return"] = np.log(df_temp["close"]).diff().fillna(0)
                df_temp["high_low_spread"] = df_temp["high"] - df_temp["low"]             
                df_temp["volatility"] = df_temp["return"].rolling(window=10, min_periods=1).std().fillna(0)
                df_temp["volume_change"] = df_temp["volume"].pct_change().fillna(0)
                df_temp["rolling_sum_volume"] = df_temp["volume"].rolling(window=10, min_periods=1).sum().fillna(df_temp["volume"])
                df_temp["day"] = pd.to_datetime(df_temp["date"]).dt.day
                df_out = df_temp[[
                    "open", "high", "low", "close", "volume",
                    "return", "log_return", "high_low_spread",
                    "volume_change", "rolling_sum_volume", "day"
                    ]]
                df_out.to_parquet(processed_path, index=False)
            processed_paths.append(processed_path)           
            print(f"✅ Features générées : {processed_path}")
        return processed_paths
        
    def build_labels(self)->list:

        os.makedirs(self.labels_data_dir, exist_ok=True)

        label_paths = []

        for symbol in self.symbols:
            month_str = f"{self.month:02d}"
            year_str = str(self.year)
            base_name = f"{symbol}-bookTicker-{year_str}-{month_str}"
            parquet_path = os.path.join(self.raw_data_dir, f"{base_name}.parquet")
            label_path = os.path.join(self.labels_data_dir, f"{base_name}_labels.parquet")

            if os.path.exists(label_path):
                print(f"✅ Fichier déjà existant, ignoré : {label_path}")
                continue

            if not os.path.exists(parquet_path):
                print(f"❌ Fichier bookTicker introuvable : {parquet_path}")
                continue

            print(f"📊 Construction des labels pour : {symbol} {year_str}-{month_str}")

            df = pd.read_parquet(parquet_path)
            df.columns = [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time"
            ]

            if not np.issubdtype(df["transaction_time"].dtype, np.datetime64):
                try:
                    df["datetime"] = pd.to_datetime(df["transaction_time"].astype(np.int64), unit="ms")
                except Exception as e:
                    print("⛔ Erreur conversion datetime:", e)
                    return
            else:
                df["datetime"] = df["transaction_time"]

            df["day"] = df["datetime"].dt.day 

            df["spread"] = df["best_ask_price"] - df["best_bid_price"]

            df_daily = df.groupby("day")["spread"].mean().reset_index()
            df_daily.columns = ["day", "spread_real"]

            df_daily.to_parquet(label_path, index=False)
            print(f"✅ Labels générés : {label_path}")
            label_paths.append(label_path)

            os.remove(parquet_path)
            print(f"Le fichier {parquet_path} a été supprimé avec succès")
        return label_paths

    def build_training_data(self, symbols=None, serial_dependency=False, daily=True):
        if symbols is None:
            symbols = self.symbols

        X_all = []
        y_all = []
        meta_all = []

        for symbol in symbols:
            month_str = f"{self.month:02d}"
            year_str = str(self.year)
            base_name = f"{symbol}-1m-{year_str}-{month_str}"

            if daily:
                feature_path = os.path.join(
                    self.daily_features_data_dir,
                    f"{base_name}_features_{serial_dependency}.parquet"
                )
            else:
                feature_path = os.path.join(
                    self.minute_features_data_dir,
                    f"{base_name}_features_1min_klines.parquet"
                )

            label_name = f"{symbol}-bookTicker-{year_str}-{month_str}_labels.parquet"
            label_path = os.path.join(self.labels_data_dir, label_name)

            if not os.path.exists(feature_path):
                print(f"❌ Fichier features manquant : {feature_path}")
                continue
            if not os.path.exists(label_path):
                print(f"❌ Fichier labels manquant : {label_path}")
                continue

            df_feat = pd.read_parquet(feature_path)
            df_label = pd.read_parquet(label_path)

            if daily:
                if "day" not in df_feat.columns:
                    df_feat["datetime"] = pd.to_datetime(df_feat["date"])
                    df_feat["day"] = df_feat["datetime"].dt.day

                df_merged = pd.merge(df_feat, df_label, on="day", how="inner")

                X = df_merged.drop(columns=["spread_real"] + [col for col in df_merged.columns if col.startswith("date")], errors="ignore")
                y = df_merged["spread_real"]

                meta_all.append(pd.DataFrame({
                    "symbol": [symbol] * len(df_merged),
                    "day": df_merged["day"]
                }))

            else:
                df_merged = pd.merge(df_feat, df_label, on="day", how="inner")
                features = [
                    "day", "open", "high", "low", "close", "volume",
                    "return", "log_return", "high_low_spread",
                    "volume_change", "rolling_sum_volume"
                ]
                X = df_merged[features]
                y = df_merged["spread_real"]

                meta_all.append(pd.DataFrame({
                    "symbol": [symbol] * len(df_merged),
                    "day": df_merged["day"]
                }))

            X_all.append(X)
            y_all.append(y)

        if not X_all:
            raise ValueError("Aucune donnée disponible pour entraîner le modèle.")

        X_final = pd.concat(X_all, ignore_index=True).to_numpy(dtype=np.float32)
        y_final = pd.concat(y_all, ignore_index=True).to_numpy(dtype=np.float32)
        self.meta = pd.concat(meta_all, ignore_index=True)

        max_float32 = np.finfo(np.float32).max
        min_float32 = np.finfo(np.float32).min
        X_final = np.nan_to_num(X_final, nan=0.0, posinf=max_float32, neginf=min_float32)
        y_final = np.nan_to_num(y_final, nan=0.0, posinf=max_float32, neginf=min_float32)

        max_float32 = np.finfo(np.float32).max
        min_float32 = np.finfo(np.float32).min
        X_final = np.clip(X_final, min_float32, max_float32)
        y_final = np.clip(y_final, min_float32, max_float32)

        print(f"✅ Données prêtes : X.shape = {X_final.shape}, y.shape = {y_final.shape}")
        return X_final, y_final


    def time_series_features(self, X, y, daily = True, test_size=0.2, val_size=0.2):
        """
        Trie les données dans l'ordre [day, symbol] et applique un split temporel train / val / test.
        """
        if not daily:
            return self._time_series_split_minute(X, y, test_size, val_size)
        else:
            return self._time_series_split_daily(X, y, test_size, val_size)


    def _reduce_labels_by_day_symbol(self, y_part, day_part, symbol_part):
        """
        Réduit les labels à un seul par (day, symbol), en conservant l'ordre d'apparition [day, symbol].
        """
        import pandas as pd

        df = pd.DataFrame({
            "y": y_part,
            "day": day_part,
            "symbol": symbol_part
        })

        df_sorted = df.sort_values(by=["day", "symbol"], kind="stable")

        grouped = df_sorted.groupby(["day", "symbol"], sort=False)["y"].mean()

        return grouped.values.astype(np.float32)


    def _time_series_split_minute(self, X, y, test_size, val_size):
        df_meta = self.meta.copy()
        df_meta["row_idx"] = np.arange(len(df_meta))

        df_meta_sorted = df_meta.sort_values(by=["day", "symbol"]).reset_index(drop=True)
        sorted_indices = df_meta_sorted["row_idx"].values

        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        days_sorted = df_meta_sorted["day"].values
        symbols_sorted = df_meta_sorted["symbol"].values

        unique_days = np.unique(days_sorted)
        n_days = len(unique_days)

        rows_per_day = self.nb_assets * 1440
        assert len(X_sorted) == rows_per_day * n_days, "X incohérent avec nb_assets et minute-level"

        n_test = int(np.ceil(test_size * n_days)) if test_size < 1 else int(test_size)
        n_val  = int(np.ceil(val_size * (n_days - n_test))) if val_size < 1 else int(val_size)
        n_train = n_days - n_test - n_val

        def get_day_indices(start_day_idx, n_days_split):
            day_indices = []
            for i in range(start_day_idx, start_day_idx + n_days_split):
                start = i * rows_per_day
                end = (i + 1) * rows_per_day
                day_indices.append(np.arange(start, end))
            return np.concatenate(day_indices)

        idx_train = get_day_indices(0, n_train)
        idx_val   = get_day_indices(n_train, n_val)
        idx_test  = get_day_indices(n_train + n_val, n_test)

        X_train = X_sorted[idx_train]
        X_val = X_sorted[idx_val]
        X_test = X_sorted[idx_test]

        y_train_full = y_sorted[idx_train]
        y_val_full = y_sorted[idx_val]
        y_test_full = y_sorted[idx_test]

        days_train = days_sorted[idx_train]
        days_val = days_sorted[idx_val]
        days_test = days_sorted[idx_test]

        symbols_train = symbols_sorted[idx_train]
        symbols_val = symbols_sorted[idx_val]
        symbols_test = symbols_sorted[idx_test]

        def reduce_labels(y_part, day_part, symbol_part):
            df = pd.DataFrame({
                "y": y_part,
                "day": day_part,
                "symbol": symbol_part
            })
            return df.groupby(["day", "symbol"], sort=False)["y"].mean().values.astype(np.float32)

        y_train = reduce_labels(y_train_full, days_train, symbols_train)
        y_val = reduce_labels(y_val_full,   days_val,   symbols_val)
        y_test = reduce_labels(y_test_full,  days_test,  symbols_test)

        print(f"✅ Split : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print(f"✅ Labels : y_train={len(y_train)}, y_val={len(y_val)}, y_test={len(y_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _time_series_split_daily(self, X, y, test_size, val_size):
        df_meta = self.meta.copy()
        df_meta["row_idx"] = np.arange(len(df_meta))
        df_meta_sorted = df_meta.sort_values(by=["day", "symbol"]).reset_index(drop=True)

        sorted_indices = df_meta_sorted["row_idx"].values
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        days_sorted = df_meta_sorted["day"].values

        unique_days = np.unique(days_sorted)
        n_days = len(unique_days)

        n_test = int(np.ceil(test_size * n_days)) if test_size < 1 else int(test_size)
        n_val = int(np.ceil(val_size * (n_days - n_test))) if val_size < 1 else int(val_size)
        n_train = n_days - n_test - n_val

        train_days = unique_days[:n_train]
        val_days = unique_days[n_train:n_train + n_val]
        test_days = unique_days[n_train + n_val:]

        def get_mask(days_subset):
            return np.isin(days_sorted, days_subset)

        mask_train = get_mask(train_days)
        mask_val = get_mask(val_days)
        mask_test = get_mask(test_days)

        X_train = X_sorted[mask_train]
        X_val = X_sorted[mask_val]
        X_test = X_sorted[mask_test]

        y_train = y_sorted[mask_train]
        y_val = y_sorted[mask_val]
        y_test = y_sorted[mask_test]

        print(f"✅ Split : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print(f"✅ Labels : y_train={len(y_train)}, y_val={len(y_val)}, y_test={len(y_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def make_batches(X_rnn, y_rnn, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((X_rnn, y_rnn))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X_rnn))
        dataset = dataset.batch(batch_size)
        return dataset
    
    @staticmethod
    def format_data(
        X,
        y,
        model_type,
        daily=True,
        nb_assets=None,
        minutes_per_day=1440,
        window=None
    ):
        if nb_assets is None:
            raise ValueError("nb_assets doit être spécifié")

        model_type = model_type.lower()
        N, d = X.shape

        if daily:
            nb_days = N
            if y.shape[0] != nb_days:
                raise ValueError(f"[daily=True] y.shape[0] doit être égal à X.shape[0] (= {nb_days})")
            y_out = y
            rows_per_day = 1

        else:
            rows_per_day = nb_assets * minutes_per_day
            if N % rows_per_day != 0:
                raise ValueError(f"[daily=False] X.shape[0]={N} n'est pas divisible par nb_assets*minutes_per_day={rows_per_day}")
            nb_days = N // rows_per_day

            if y.shape[0] != nb_days * nb_assets:
                raise ValueError(f"[daily=False] y.shape[0]={y.shape[0]} attendu = nb_days*nb_assets = {nb_days}*{nb_assets}")
            y_out = y.reshape(nb_days, nb_assets)

        if model_type == "mlp":
            X_out = X.reshape(nb_days, rows_per_day * d)

        elif model_type == "cnn":
            X_out = X.reshape(nb_days, rows_per_day, d)

        elif model_type == "rnn":
            X_out = X.reshape(nb_days, rows_per_day, d)

        elif model_type == "seq":
            if window is None:
                raise ValueError("window doit être spécifié pour model_type='seq'")

            X_r = X.reshape(nb_days, rows_per_day, d)

            if nb_days <= window:
                return np.empty((0, window, rows_per_day * d), dtype=np.float32), np.empty((0, nb_assets), dtype=np.float32)

            X_out = np.array([
                X_r[t:t + window].reshape(window, -1)
                for t in range(nb_days - window)
            ], dtype=np.float32)

            y_out = y_out[window:] 

        else:
            raise ValueError(f"Modèle inconnu : '{model_type}'")

        return X_out.astype(np.float32), y_out.astype(np.float32)

    def compute_and_save_parametric_estimators(
            self, 
            X, 
            use_opposed=False, 
            sort_mode="asset_first"
            ):
        if sort_mode not in {"asset_first", "day_first"}:
            raise ValueError("sort_mode doit être 'asset_first' ou 'day_first'.")

        crypto_names = sorted(self.symbols)
        crypto_str = "-".join(crypto_names)
        filename = f"param_est__{crypto_str}_{self.year:04d}_{self.month:02d}_" \
                f"opposed={use_opposed}_{sort_mode}.parquet"
        file_path = os.path.join(self.parametric_estimators_dir, filename)

        if os.path.exists(file_path):
            print(f"✅ Fichier déjà existant, pas de recalcul : {file_path}")
            return pd.read_parquet(file_path)

        df = pd.DataFrame(X, columns=["day", "open", "high", "low", "close", "volume"])

        meta_subset = self.generate_meta_from_X(X, sort_mode=sort_mode)
        list_dfs = []

        for asset in self.symbols:
            print(f"📊 Traitement de l'actif {asset}...")

            asset_mask = meta_subset["symbol"] == asset
            meta_asset = meta_subset[asset_mask]
            df_asset = df.loc[asset_mask.values]

            if df_asset.empty:
                print(f"⚠️ Aucun point trouvé pour {asset}, on passe.")
                continue

            df_result = get_parametric_estimators_series(df_asset.values, meta_asset, use_opposed)
            list_dfs.append(df_result)

        if not list_dfs:
            raise ValueError("Aucune donnée valide trouvée dans X.")

        df_concat = pd.concat(list_dfs, axis=0, ignore_index=True)
        os.makedirs(self.parametric_estimators_dir, exist_ok=True)
        df_concat.to_parquet(file_path, index=False)
        print(f"✅ Estimations sauvegardées dans : {file_path}")
        return df_concat

    def generate_meta_from_X(self, X, sort_mode="asset_first", minutes_per_day=1440):
        nb_assets = len(self.symbols)
        nb_rows = X.shape[0]
        total_minutes_per_day = minutes_per_day * nb_assets

        if nb_rows % total_minutes_per_day != 0:
            raise ValueError("X ne couvre pas un nombre entier de jours pour tous les actifs.")

        nb_days = nb_rows // total_minutes_per_day

        meta = []
        if sort_mode == "asset_first":
            for symbol in self.symbols:
                for day in range(nb_days):
                    meta.extend([{"symbol": symbol, "day": day}] * minutes_per_day)
        elif sort_mode == "day_first":
            for day in range(nb_days):
                for symbol in self.symbols:
                    meta.extend([{"symbol": symbol, "day": day}] * minutes_per_day)
        else:
            raise ValueError("sort_mode doit être 'asset_first' ou 'day_first'.")

        return pd.DataFrame(meta)
