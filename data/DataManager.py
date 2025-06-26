import os
import pandas as pd
import numpy as np
import requests
import zipfile
import tensorflow as tf
from numpy.random import normal
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.daily_features_builders import (get_serial_dependancy_features_v2,normalize)
from utils.res_comp import get_parametric_estimators_series

RAW_DATA_DIR = "raw data"
LABELS_DATA_DIR = "labels data"
DAILY_FEATURES_DATA_DIR = "daily features data"
MINUTE_FEATURES_DATA_DIR = "minute features data"
PARAMETRIC_ESTIMATORS_DIR = "parametric estimators"

LABELS_LABEL = "labels"
FEATURES_LABEL = "features"

class DataManager:
    """
    Classe permettant de récupérer les données et de préparer construire les features pour les modèles

    Arguments :
    - symbols : liste contenant un ensemble de cryptos (pour le train ou le test)
    - dates : liste contenant un ensemble de dates (pour le train ou le test)
    - light : booléen pour alléger l'ouverture des fichiers
    """
    def __init__(self, symbols: list, dates:list,light = False):
        self.symbols:list = [s.upper() for s in symbols]
        self.dates: list = dates
        self.freq:str = "1m"
        self.nb_assets:int = len(self.symbols)
        self.light = light

        # Récupération des différents paths vers les répertoires où seront stockées les features, les labels, ...
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_dir = os.path.join(self.module_dir, RAW_DATA_DIR)
        self.labels_data_dir = os.path.join(self.module_dir, LABELS_DATA_DIR)
        self.minute_features_data_dir = os.path.join(self.module_dir, MINUTE_FEATURES_DATA_DIR)
        self.parametric_estimators_dir = os.path.join(self.module_dir, PARAMETRIC_ESTIMATORS_DIR)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.labels_data_dir, exist_ok=True)
        os.makedirs(self.minute_features_data_dir, exist_ok=True)
        os.makedirs(self.parametric_estimators_dir, exist_ok=True)

        # Récupération des url pour télécharger les barres d'une minute et les carnets d'ordres
        self.kline_base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        self.bookticker_base_url = "https://data.binance.vision/data/futures/um/monthly/bookTicker"

    def download_and_prepare_data(self):
        """
        Méthode central de télécharger les barres d'une minute et les données de carnet d'ordres
        """

        # Boucle par actif
        for symbol in self.symbols:
            # Boucle par date
            for year, month in self.dates:

                month_str = f"{month:02d}"
                year_str = str(year)

                base_name = f"{symbol}-bookTicker-{year_str}-{month_str}"
                label_path = os.path.join(self.labels_data_dir, f"{base_name}_labels.parquet")

                # Première partie : import des barres d'une minute et construction des features
                kline_filename = f"{symbol}-{self.freq}-{year_str}-{month_str}"
                kline_parquet_path = os.path.join(self.raw_data_dir, f"{kline_filename}.parquet")

                # Si les données n'ont pas déjà été importées, on les importe
                if not os.path.exists(kline_parquet_path):
                    kline_zip_url = f"{self.kline_base_url}/{symbol}/{self.freq}/{kline_filename}.zip"
                    kline_zip_path = os.path.join(self.raw_data_dir, f"{kline_filename}.zip")
                    print(f"Téléchargement Klines : {kline_zip_url}")
                    response = requests.get(kline_zip_url)

                    # En l'absence d'erreur, dézipage des données
                    if response.status_code != 200:
                        raise Exception(f"Erreur téléchargement : {kline_zip_url}")
                    with open(kline_zip_path, "wb") as f:
                        f.write(response.content)
                    with zipfile.ZipFile(kline_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.raw_data_dir)
                    os.remove(kline_zip_path)

                    # Ouverture du fichier CSV et sauvegarde en parquet
                    csv_path = os.path.join(self.raw_data_dir, f"{kline_filename}.csv")
                    df_kline = pd.read_csv(csv_path)
                    df_kline.columns = [
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "nb_trades",
                        "taker_buy_base", "taker_buy_quote", "ignore"
                    ]
                    df_kline.to_parquet(kline_parquet_path, index=False)

                    # Destruction du CSV
                    os.remove(csv_path)
                    print(f"Klines sauvegardé : {kline_parquet_path}")
                else:
                    print(f"Klines déjà existant : {kline_parquet_path}")

                # Deuxième partie : import des orders books
                bookticker_filename = f"{symbol}-bookTicker-{year_str}-{month_str}"
                bookticker_zip_filename = f"{bookticker_filename}.zip"
                bookticker_csv_filename = f"{bookticker_filename}.csv"

                bookticker_zip_url = f"{self.bookticker_base_url}/{symbol}/{bookticker_zip_filename}"

                bookticker_zip_path = os.path.join(self.raw_data_dir, bookticker_zip_filename)
                csv_path = os.path.join(self.raw_data_dir, bookticker_csv_filename)
                bookticker_parquet_path = os.path.join(self.raw_data_dir, f"{bookticker_filename}.parquet")

                # Si les données n'ont pas déjà été importées, elles sont importées à nouveau
                if not os.path.exists(bookticker_parquet_path) and not os.path.exists(label_path):
                    print(f"Téléchargement BookTicker : {bookticker_zip_url}")
                    response = requests.get(bookticker_zip_url, stream=True)

                    if response.status_code != 200:
                        print(f"BookTicker indisponible : {bookticker_zip_url} (code {response.status_code})")
                        continue

                    with open(bookticker_zip_path, "wb") as f:
                        f.write(response.content)

                    with zipfile.ZipFile(bookticker_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.raw_data_dir)
                    os.remove(bookticker_zip_path)

                    # Pour accélerer les traitements, seules les colonnes requises pour le calcul du spread journalier sont conservées
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

                    # Vérification sur le nombre de lignes
                    n_rows = len(df_bt)
                    if n_rows < 10_000_000:
                        print(f"Attention : fichier bookTicker anormalement petit ({n_rows} lignes)")

                    # Sauvegarde en parquet et destruction du CSV
                    df_bt.to_parquet(bookticker_parquet_path, index=False)
                    os.remove(csv_path)
                    print(f"BookTicker sauvegardé : {bookticker_parquet_path}")
                else:
                    print(f"BookTicker déjà existant : {bookticker_parquet_path}")

    def load_features(self, serial_dependency:bool = False, clean_features: bool = False, use_tick_size: bool = False):
        """
        Méthode permettant de sauvegarder les features intra-day utilisées pour estimer les modèles.
        """

        processed_paths = []

        # Retraitement sur les jours pour garantir l'uniformisation des données
        nb_days:int = 30
        nb_minute_per_day:int = 1440
        nb_sequences:int = nb_days * nb_minute_per_day

        # Si on souhaite supprimer les features créées précédemment
        if clean_features:
            self._clean_features("features")

        # Double boucle par actif et date
        for symbol in self.symbols:
            for year, month in self.dates:
                month_str = f"{month:02d}"
                filename = f"{symbol}-{self.freq}-{year}-{month_str}"
                parquet_path = os.path.join(self.raw_data_dir, f"{filename}.parquet")

                # Récupération du chemin vers les features en minutes
                processed_path = os.path.join(
                    self.minute_features_data_dir,
                    f"{filename}_features_1min_klines.parquet"
                )

                # Si le fichier existe déjà, pas besoin de refaire les calculs, sinon on construit les features
                if os.path.exists(processed_path):
                    print(f"Fichier déjà existant, ignoré : {processed_path}")
                    processed_paths.append(processed_path)
                    continue
                print(f"Construction des features pour : {symbol} {year}-{month_str}")

                # Construction des features intraday avec / sans dépendance sérielle
                df_out = self.build_features(parquet_path, use_serial_dependency=serial_dependency)

                # Si on souhaite utiliser le ticksize comme feature
                if use_tick_size:
                    tick_size: float = self._load_ticksize(symbol)
                    df_out["ticksize"] = tick_size

                # Si on travaille sur un mois à moins de 30 jours, on passe (uniformisation des données) ==> à revoir ; cas à +30 jours gérés dans la méthode
                if df_out.shape[0] < nb_sequences:
                    continue

                # Cas d'un mois > 30 jours : seuls les 30 premiers jours sont conservés
                if df_out.shape[0] > nb_sequences:
                    df_out = df_out.iloc[0:nb_sequences, :]

                # Sauvegarde
                df_out.to_parquet(processed_path, index=False)
                processed_paths.append(processed_path)
                print(f"Features générées pour l'actif {symbol} et la date {year}-{month_str}")
        return processed_paths

    def _clean_features(self, elem_model:str):
        """
        Méthode permettant de supprimer tous les fichiers contenant les features / labels créés lors des
        run précédents
        """
        print(f"Suppression des fichiers de {elem_model} créés précédemment")
        if elem_model == LABELS_LABEL:
            filelist: list = [f for f in os.listdir(self.labels_data_dir)]
        elif elem_model == FEATURES_LABEL:
            filelist: list = [f for f in os.listdir(self.minute_features_data_dir)]
        else:
            raise Exception(f"Aucun répertoire n'est associé à {elem_model}")

        # Boucle sur tous les fichiers et suppression
        for f in filelist:
            if elem_model == LABELS_LABEL:
                # Récupération du path absolu du fichier
                file_path = os.path.join(self.labels_data_dir, f)
            else:
                file_path = os.path.join(self.minute_features_data_dir, f)
            os.remove(file_path)
        print("Suppression des fichiers de features terminée")

    @staticmethod
    def _load_ticksize(symbol: str)->float:
        """
        Méthode permettant de télécharger le ticksize associé au ticker
        pour tester son pouvoir prédictif dans le modèle de Deep Learning
        """

        # Création du ticksize (nan par défaut)
        tick_size: float = np.nan
        try:
            # Import du ticksize depuis l'API binance
            exchange_info = requests.get(
                f"https://fapi.binance.com/fapi/v1/exchangeInfo?symbol={symbol}"
            ).json()
            for f in exchange_info["symbols"][0]["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                    break
        except Exception as e:
            print(f"Erreur récupération tick_size : {e}")

        # Récupération du ticksize
        return tick_size

    @staticmethod
    def build_features(klines_path, use_serial_dependency: bool = False):
        """
        Méthode permettant de construire les features intraday

        Les features intraday utilisées sont :
        - OHCL + Vol
        - Les rendements entre deux minutes
        - Le spread H-L au cours d'une minute
        - L'évolution du spread H-L
        - La volatilité des rendements
        - L'évolution du volume
        - Le volume moyen sur une fenêtre passée
        - Le jour de la semaine
        - Le numéro de l'actif (= pseudo encodage ==> à voir)
        """

        # Ouverture du fichier
        df = pd.read_parquet(klines_path)

        # Création d'un dataframe pour stocker les features
        df_features: pd.DataFrame = pd.DataFrame()

        # Récupération de la date au format datetime
        df_features["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df_features["date"] = df_features["datetime"].dt.date

        # Récupération des barres + volume
        df_features["close"] = df["close"].astype(float)
        df_features["open"] = df["open"].astype(float)
        df_features["high"] = df["high"].astype(float)
        df_features["low"] = df["low"].astype(float)
        df_features["volume"] = df["volume"].astype(float)

        # Calcul du rendement et de la vol sur 10 minutes
        df_features["returns"] = df_features["close"].pct_change().fillna(0)
        df_features["volatility"] = df_features["returns"].rolling(window=10, min_periods=1).std().fillna(0)

        # Calcul de l'évolution du volume / moyenne du volume sur les 10 minutes passées
        df_features["volume_change"] = df_features["volume"].pct_change().fillna(0)
        df_features["rolling_mean_volume"] = df_features["volume"].rolling(window=10, min_periods=1).mean().fillna(df_features["volume"])

        # Calcul du spread High - Low du jour et de son évolution entre deux dates (en VA)
        df_features["spread_high_low"] = df_features["high"] - df_features["low"]
        df_features["delta_spread_high_low"] = np.abs(df_features["spread_high_low"].pct_change().fillna(0))

        # Récupération des informations temporelles
        df_features["year"] = df_features["datetime"].dt.year
        df_features["month"] = df_features["datetime"].dt.month
        df_features["day"] = df_features["datetime"].dt.day
        df_features["hour"] = df_features["datetime"].dt.hour

        # Intégration éventuelle d'indicateurs de dépendance sérielle
        if use_serial_dependency:

            # Récupération du dataframe avec dépendance sérielle
            df_features = get_serial_dependancy_features_v2(df_features)


        # Suppression de la colonne de dates
        df_features.drop("datetime", axis=1, inplace=True)
        df_features.drop("date", axis=1, inplace=True)

        # récupération du dataframe
        return df_features

    def build_labels(self, clean_labels: bool = False)->list:
        """
        Méthode permettant de calculer les spreads journaliers
        """
        os.makedirs(self.labels_data_dir, exist_ok=True)

        label_paths = []

        # Retraitement sur les jours pour garantir l'uniformisation des données
        nb_days:int = 30

        # Cas où l'utilisateur souhaite supprimer les labels créés précédemment
        if clean_labels:
            self._clean_features("labels")

        # Double boucle actif / date
        for symbol in self.symbols:
            for (year, month) in self.dates:
                month_str = f"{month:02d}"
                year_str = str(year)
                base_name = f"{symbol}-bookTicker-{year_str}-{month_str}"
                parquet_path = os.path.join(self.raw_data_dir, f"{base_name}.parquet")
                label_path = os.path.join(self.labels_data_dir, f"{base_name}_labels.parquet")

                # Vérification de l'existence des fichiers (s'ils n'existent pas, on réalise les calculs)
                if os.path.exists(label_path):
                    print(f"Fichier déjà existant, ignoré : {label_path}")
                    continue

                if not os.path.exists(parquet_path):
                    print(f"Fichier bookTicker introuvable : {parquet_path}")
                    continue

                print(f"Construction des labels pour : {symbol} {year_str}-{month_str}")

                # Chargement des booktickers
                df = pd.read_parquet(parquet_path)
                if self.light:
                    df.columns = [
                        "best_bid_price",
                        "best_ask_price",
                        "transaction_time",
                        ]
                else:
                    df.columns = [
                        "update_id",
                        "best_bid_price",
                        "best_bid_qty",
                        "best_ask_price",
                        "best_ask_qty",
                        "transaction_time",
                        "event_time"
                        ]

                # si transaction_time n'est pas au format date, conversion
                if not np.issubdtype(df["transaction_time"].dtype, np.datetime64):
                    try:
                        df["datetime"] = pd.to_datetime(df["transaction_time"].astype(np.int64), unit="ms")
                    except Exception as e:
                        print("Erreur conversion datetime:", e)
                        return []
                else:
                    df["datetime"] = df["transaction_time"]

                # Récupération du jour
                df["day"] = df["datetime"].dt.day

                # Calcul des séries bid / ask normalisée
                df["normalized_ask"] = normalize(df["best_ask_price"])
                df["normalized_bid"] = normalize(df["best_bid_price"])

                # Calcul du spread entre les séries de bid / ask normalisées en valeur absolue (pour garantir que le spread soit positif)
                df["spread"] = np.abs(df["normalized_ask"] - df["normalized_bid"])
                #df["spread"] = df["best_ask_price"] - df["best_bid_price"]
                #df["spread"] = normalize(df["spread"])

                # Regroupement par jour ==> pas sur pour la normalisation, peut être nul
                df_daily = df.groupby("day")["spread"].mean().reset_index()
                df_daily.columns = ["day", "spread_real"]

                # Si on travaille sur un mois à moins de 30 jours, on passe (uniformisation des données) ==> à revoir ; cas à +30 jours gérés dans la méthode
                if df_daily.shape[0] < nb_days:
                    continue

                # Cas d'un mois > 30 jours : seuls les 30 premiers jours sont conservés
                if df_daily.shape[0] > nb_days:
                    df_daily = df_daily.iloc[0:nb_days, :]

                # Export des données
                df_daily.to_parquet(label_path, index=False)
                print(f"Labels générés : {label_path}")
                label_paths.append(label_path)

                # Suppression du fichier contenant les booktickers pour alléger la mémoire
                # os.remove(parquet_path)
                # print(f"Le fichier {parquet_path} a été supprimé avec succès")
        return label_paths

    def build_training_data(self, symbols=None, do_aggregate: bool = False):
        """
        Méthode permettant de construire les dataframes d'entrainement

        output : format (nb_actif * nb_mois * 1440, nb_features) organisées avec les données pour chaque actif, mois par mois
        """

        if symbols is None:
            symbols = self.symbols

        # Liste pour stocker les résultats successifs
        X_all = []
        y_all = []
        meta_all = []

        # Boucle sur les périodes
        for year, month in self.dates:
            # Boucle sur les actifs
            for symbol in symbols:
                month_str = f"{month:02d}"
                year_str = str(year)
                base_name = f"{symbol}-1m-{year_str}-{month_str}"

                feature_path = os.path.join(
                    self.minute_features_data_dir,
                    f"{base_name}_features_1min_klines.parquet"
                )

                # Vérification de l'existence des fichiers
                label_name = f"{symbol}-bookTicker-{year_str}-{month_str}_labels.parquet"
                label_path = os.path.join(self.labels_data_dir, label_name)

                if not os.path.exists(feature_path):
                    print(f"Fichier features manquant : {feature_path}")
                    continue
                if not os.path.exists(label_path):
                    print(f"Fichier labels manquant : {label_path}")
                    continue

                # Import des données
                df_feat = pd.read_parquet(feature_path)
                df_label = pd.read_parquet(label_path)

                # Agrégation de la séquence (cas du MLP, ...) ==> à ajouter ici
                # df_feat = ...

                # Fusion des dataframes et récupération des features / spread
                df_merged = pd.merge(df_feat, df_label, on="day", how="inner")
                X = df_merged.loc[:, df_merged.columns != "spread_real"]
                y = df_merged["spread_real"]

                # Spread du jour = moyenne du spread intra-journalier
                # y = df_merged.groupby("day")["spread_real"].mean()

                meta_all.append(pd.DataFrame({
                    "symbol": [symbol] * len(df_merged),
                    "day": df_merged["day"],
                    "month":df_merged["month"]
                }))

                # Ajout dans la liste
                X_all.append(X)
                y_all.append(y)

        if not X_all:
            raise ValueError("Aucune donnée disponible pour entraîner le modèle.")

        # Concaténation des dataframes et conversion en array pour les traitements ultérieurs
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

        print(f"Données prêtes : X.shape = {X_final.shape}, y.shape = {y_final.shape}")
        return X_final, y_final

    def build_train_val_dataset(self, val_size = 0.2, is_test: bool = False, do_aggregate: bool = False):
        """
        Méthode permettant de construire les dataframe d'input / output requis pour les modèles

        output :
        - un dataframe contenant les features pour tous les actifs et toutes les dates
        """

        # Récupération des arrays avec les features/label pour tous les actifs date par date
        X, y = self.build_training_data(do_aggregate = do_aggregate)

        # Récupération du nombre d'actifs du datamanager et de la longueur d'une séquence
        nb_assets: int = len(self.symbols)
        len_sequence: int = 1440

        # Cas où l'on travaille sur l'ensemble de test
        if is_test:

            # Vérification : le nombre de ligne doit être divisible par nb_assets * 1440 pour continuer les traitements
            while X.shape[0]%(nb_assets * len_sequence)!= 0:
                X = X[:-1,:]
            while y.shape[0]%nb_assets != 0:
                y = y[:-1,:]

            # transformation de y en spread moyen quotidien
            y = self.reduce_labels(y, len_sequence)
            return X,y

        # Autre cas : train / val split usuel
        else:
            if X.shape[0]%(nb_assets * len_sequence)!= 0:
                raise Exception("Problème pour effectuer le split entre train et test")
            # Séparation entre train / val et récupération
            X_train, y_train, X_val, y_val = self._time_series_split_minute(X, y, val_size)
        return X_train, X_val, y_train, y_val

    def time_series_features(self, X, y, daily = True, test_size=0.2, val_size=0.2):
        """
        Trie les données dans l'ordre [day, symbol] et applique un split temporel train / val / test.
        """
        if not daily:
            return self._time_series_split_minute(X, y, test_size, val_size)
        else:
            return self._time_series_split_daily(X, y, test_size, val_size)

    @staticmethod
    def reduce_labels(y_part, n_min: int):
        """
        fonction  permettant de passer des spread intraday au spread journalier moyen

        arguments :
        - nb_rows_per_day : nombre de lignes par mois
        """

        # Modification de la dimension de y : passage en 3D (nb_days * nb_months * nb_assets, 1440,1)
        spread_per_day = y_part.reshape(int(np.ceil(y_part.shape[0] / n_min)), n_min, 1)

        # Calcul de la moyenne intraday
        avg_daily_spread = np.mean(spread_per_day, axis=1)

        return avg_daily_spread

    def _time_series_split_minute(self, X, y, val_size):
        df_meta = self.meta.copy()
        df_meta["row_idx"] = np.arange(len(df_meta))

        df_meta_sorted = df_meta.sort_values(by=["day", "symbol"]).reset_index(drop=True)
        sorted_indices = df_meta_sorted["row_idx"].values

        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        days_sorted = df_meta_sorted["day"].values
        symbols_sorted = df_meta_sorted["symbol"].values

        # Nombre de minutes par jour
        n_min: int = 1440

        # Récupération du nombre de jours / mois
        unique_days = np.unique(days_sorted)
        n_days = len(unique_days)

        # Récupération du nombre de mois
        n_months: int = len(self.dates)

        # Récupération du nombre de périodes
        n_periods: int = n_months * n_days

        # Nombre de lignes par jours / Nombre de lignes dans le train
        rows_per_day = self.nb_assets * n_min
        nb_val_train: int = rows_per_day * n_days * n_months
        assert len(X_sorted) == nb_val_train, "X incohérent avec nb_assets et minute-level"

        # Nombre de données temporelles à conserver pour l'ensemble de train et de validation
        n_val = int(np.ceil(val_size * n_periods))
        n_train = n_periods - n_val

        def get_day_indices(start_day_idx, n_days_split):
            day_indices = []
            for i in range(start_day_idx, start_day_idx + n_days_split):
                start = i * rows_per_day
                end = (i + 1) * rows_per_day
                day_indices.append(np.arange(start, end))
            return np.concatenate(day_indices)

        idx_train = get_day_indices(0, n_train)
        idx_val   = get_day_indices(n_train, n_val)

        X_train = X_sorted[idx_train]
        X_val = X_sorted[idx_val]

        y_train_full = y_sorted[idx_train]
        y_val_full = y_sorted[idx_val]

        y_train = self.reduce_labels(y_train_full, n_min=n_min)
        y_val = self.reduce_labels(y_val_full, n_min=n_min)

        print(f"Split : train={len(X_train)}, val={len(X_val)}")
        print(f"Labels : y_train={len(y_train)}, y_val={len(y_val)}")

        return X_train, y_train, X_val, y_val

    def _time_series_split_daily(self, X, y, val_size):
        df_meta = self.meta.copy()
        df_meta["row_idx"] = np.arange(len(df_meta))
        df_meta_sorted = df_meta.sort_values(by=["day", "symbol"]).reset_index(drop=True)

        sorted_indices = df_meta_sorted["row_idx"].values
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        days_sorted = df_meta_sorted["day"].values

        unique_days = np.unique(days_sorted)

        # Découpage du dataframe
        n_days = len(unique_days)
        n_val = int(np.ceil(val_size * n_days)) if val_size < 1 else int(val_size)
        n_train = n_days - n_val

        train_days = unique_days[:n_train]
        val_days = unique_days[n_train:n_train + n_val]

        def get_mask(days_subset):
            return np.isin(days_sorted, days_subset)

        mask_train = get_mask(train_days)
        mask_val = get_mask(val_days)

        X_train = X_sorted[mask_train]
        X_val = X_sorted[mask_val]

        y_train = y_sorted[mask_train]
        y_val = y_sorted[mask_val]

        print(f"Split : train={len(X_train)}, val={len(X_val)}")
        print(f"Labels : y_train={len(y_train)}, y_val={len(y_val)}")
        return X_train, y_train, X_val, y_val

    @staticmethod
    def make_batches(X_rnn, y_rnn, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((X_rnn, y_rnn))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X_rnn))
        dataset = dataset.batch(batch_size)
        return dataset
    
    @staticmethod
    def format_data(X,y,model_type,daily=True,nb_assets=None,minutes_per_day=1440,window=None):
   
    
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
            print(f"Fichier déjà existant, pas de recalcul : {file_path}")
            return pd.read_parquet(file_path)

        df = pd.DataFrame(X, columns=["day", "open", "high", "low", "close", "volume"])

        meta_subset = self.generate_meta_from_X(X, sort_mode=sort_mode)
        list_dfs = []

        for asset in self.symbols:
            print(f"Traitement de l'actif {asset}...")

            asset_mask = meta_subset["symbol"] == asset
            meta_asset = meta_subset[asset_mask]
            df_asset = df.loc[asset_mask.values]

            if df_asset.empty:
                print(f"Aucun point trouvé pour {asset}, on passe.")
                continue

            df_result = get_parametric_estimators_series(df_asset.values, meta_asset, use_opposed)
            list_dfs.append(df_result)

        if not list_dfs:
            raise ValueError("Aucune donnée valide trouvée dans X.")

        df_concat = pd.concat(list_dfs, axis=0, ignore_index=True)
        os.makedirs(self.parametric_estimators_dir, exist_ok=True)
        df_concat.to_parquet(file_path, index=False)
        print(f"Estimations sauvegardées dans : {file_path}")
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
