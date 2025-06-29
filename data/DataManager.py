import os
import pandas as pd
import numpy as np
import requests
import zipfile
import tensorflow as tf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

        # Récupération des paths vers les répertoires où seront stockées les features, les labels, ...
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_dir = os.path.join(self.module_dir, RAW_DATA_DIR)
        self.labels_data_dir = os.path.join(self.module_dir, LABELS_DATA_DIR)
        self.minute_features_data_dir = os.path.join(self.module_dir, MINUTE_FEATURES_DATA_DIR)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.labels_data_dir, exist_ok=True)
        os.makedirs(self.minute_features_data_dir, exist_ok=True)

        # Récupération des url pour télécharger les barres d'une minute et les carnets d'ordres
        self.kline_base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        self.bookticker_base_url = "https://data.binance.vision/data/futures/um/monthly/bookTicker"

    def download_and_prepare_data(self):
        """
        Méthode centrale permettant de télécharger les barres d'une minute / données de carnet d'ordre
        pour la construction des features et des labels
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

                # Si les données ont déjà été importées, on passe
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

                # Si les données ont déjà été importées, on passe
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
        Méthode permettant de sauvegarder en parquet les features intra-day utilisées pour estimer les modèles.
        Arguments :
        - serial_dependancy: bool pour déterminer s'il faut construire des indicateurs de dépendance sérielle
        - clean_features: bool pour déterminer s'il faut supprimer tous les fichiers de features créés précédemment (utile pour modifier les features)
        - use_tick_size: bool pour déterminer s'il faut utiliser le ticksize parmi les features du modèle

        Cette méthode permet de sauvegarder, en parquet, toutes les features pour chaque couple mois / actif
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

                # Si on travaille sur un mois à moins de 30 jours, on passe (uniformisation des données) 
                if df_out.shape[0] < nb_sequences:
                    continue

                # Cas d'un mois > 30 jours : seuls les 30 premiers jours sont conservés
                if df_out.shape[0] > nb_sequences:
                    df_out = df_out.iloc[0:nb_sequences, :]

                # Sauvegarde en parquet
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
        # Récupération de tous les fichiers de label sous forme de liste
        if elem_model == LABELS_LABEL:
            filelist: list = [f for f in os.listdir(self.labels_data_dir)]
        # Récupération de tous les fichiers de features sous forme de liste
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
            # Suppression
            os.remove(file_path)
        print(f"Suppression des fichiers de {elem_model} terminée")

    @staticmethod
    def _load_ticksize(symbol: str)->float:
        """
        Méthode permettant de télécharger le ticksize associé au ticker
        pour tester son pouvoir prédictif dans les modèle de Deep Learning
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

    def build_features(self, klines_path, use_serial_dependency: bool = False)->pd.DataFrame:
        """
        Méthode permettant de construire les features intraday utilisées pour l'estimation des modèles

        Les features intraday utilisées dans le modèle sont :
        - OHCL + Volume
        - Les rendements entre deux minutes
        - Le spread H-L au cours d'une minute
        - L'évolution du spread H-L sur 2 minutes
        - La volatilité des rendements
        - L'évolution du volume
        - Le volume moyen sur une fenêtre passée
        - La date de l'étude (heure, jour, mois, année)
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
            df_features = self.get_serial_dependancy_features(df_features)

        # Suppression de la colonne de dates
        df_features.drop("datetime", axis=1, inplace=True)
        df_features.drop("date", axis=1, inplace=True)

        # récupération du dataframe
        return df_features

    def build_labels(self, clean_labels: bool = False)->list:
        """
        Méthode permettant de calculer les spreads journaliers
        arguments:
        -   clean_labels: booléen pour déterminer s'il faut supprimer les fichiers de label précédemment créés ou non

        return:
        - list contenant le path de tous les fichiers de label
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

                # Calcul du prix mid
                df["mid_price"] = (df["best_ask_price"] + df["best_bid_price"])/2

                # Calcul du spread en % du prix (pour tenir compte de l'hétérogénéité des données crypto)
                df["spread"] = (df["best_ask_price"] - df["best_bid_price"]) / df["mid_price"]

                # Regroupement par jour
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
        return label_paths

    def build_training_data(self, symbols=None):
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

                # Fusion des dataframes et récupération des features / spread
                df_merged = pd.merge(df_feat, df_label, on="day", how="inner")
                X = df_merged.loc[:, df_merged.columns != "spread_real"]
                y = df_merged["spread_real"]

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

        # Retraitement éventuel des valeurs manquantes
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

    def build_train_val_dataset(self, val_size = 0.2, is_test: bool = False):
        """
        Méthode permettant de construire les dataframe de train / validation requis pour l'estimation des modèles

        Arguments:
        - val_size: split train/val
        - is_test: booléen pour indiquer s'il faut effectuer un split ou non
        """

        # Récupération des arrays avec les features/label pour tous les actifs date par date
        X, y = self.build_training_data()

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

    def compute_spread_pred(self, y_pred, scaler_y = None):
        """
        Méthode permettant de convertir la sortie du modèle de Deep Learning
        en spread pour comparer avec les estimateurs de Garcin.
        """
        os.makedirs(self.labels_data_dir, exist_ok=True)

        # Liste pour stocker les prix mid pour tous les actif
        mid_price_list: list = []

        # Double boucle actif / date
        for symbol in self.symbols:
            for (year, month) in self.dates:
                month_str = f"{month:02d}"
                year_str = str(year)
                base_name = f"{symbol}-bookTicker-{year_str}-{month_str}"
                parquet_path = os.path.join(self.raw_data_dir, f"{base_name}.parquet")

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

                # Calcul de l'écart-moyen
                df["mid_price"] = (df["best_ask_price"] + df["best_bid_price"]) / 2

                # Regroupement par jour
                df_daily = df.groupby("day")["mid_price"].mean().reset_index()
                df_daily.columns = ["day", "mid_price"]

                mid_price_list.append(df_daily["mid_price"].values)


        # Rescale du spread en % du prix prédit par le modèle
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred).ravel()

        # Calcul du spread / jour
        mid_price = np.concatenate(mid_price_list).astype(dtype = np.float32)
        
        # Ajustement des éléments à utiliser les les prévisions (60 éléments dans tous les cas, sauf le TKAN où la taille de la fenêtre considérée dans le projet est de 5)
        print(len(mid_price))
        print(len(y_pred))
        if len(y_pred) == 60:
            spread_pred = y_pred * mid_price
        elif len(y_pred) == 55 and len(mid_price) == 60:
            spread_pred = y_pred * mid_price[5:]
        else:
            raise ValueError(f"Taille inattendue : y_pred={len(y_pred)}, mid_price={len(mid_price)}")
        return spread_pred    

    def time_series_features(self, X, y, daily = True, test_size=0.2, val_size=0.2):
        """
        Trie les données dans l'ordre [day, symbol] et applique un split temporel train / val / test.
        """
        return self._time_series_split_minute(X, y, test_size, val_size)

    @staticmethod
    def reduce_labels(y_part, n_min: int):
        """
        fonction  permettant de passer des spread intraday au spread journalier moyen
        """

        spread_per_day = y_part.reshape(int(np.ceil(y_part.shape[0] / n_min)), n_min, 1)
        avg_daily_spread = np.mean(spread_per_day, axis=1)

        return avg_daily_spread

    def _time_series_split_minute(self, X, y, val_size):
        """
        Méthode permettant de réaliser un split pour les données intraday selon 
        les actifs et les périodes
        """
        df_meta = self.meta.copy()
        df_meta["row_idx"] = np.arange(len(df_meta))

        df_meta_sorted = df_meta.sort_values(by=["day", "symbol"]).reset_index(drop=True)
        sorted_indices = df_meta_sorted["row_idx"].values

        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        days_sorted = df_meta_sorted["day"].values

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

        # Fonction pour récupérer les indices des données à utiliser dans le train / val
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

        # Réduction des données sur le spread rapport à l'actif par jour
        y_train = self.reduce_labels(y_train_full, n_min=n_min)
        y_val = self.reduce_labels(y_val_full, n_min=n_min)

        print(f"Split : train={len(X_train)}, val={len(X_val)}")
        print(f"Labels : y_train={len(y_train)}, y_val={len(y_val)}")

        return X_train, y_train, X_val, y_val

    @staticmethod
    def get_serial_dependancy_features(df_features: pd.DataFrame)->pd.DataFrame:
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
    
    @staticmethod
    def format_data(X,y,model_type,nb_assets=None,minutes_per_day=1440,window=None):
   
        if nb_assets is None:
            raise ValueError("nb_assets doit être spécifié")
        
        model_type = model_type.lower()
        N, d = X.shape

        # Récupération du nombre de données par jour (pour les tests)
        rows_per_day = nb_assets * minutes_per_day
        if N % rows_per_day != 0:
            raise ValueError(f"[daily=False] X.shape[0]={N} n'est pas divisible par nb_assets*minutes_per_day={rows_per_day}")
        nb_days = N // rows_per_day

        if y.shape[0] != nb_days * nb_assets:
            raise ValueError(f"[daily=False] y.shape[0]={y.shape[0]} attendu = nb_days*nb_assets = {nb_days}*{nb_assets}")
        y_out = y.reshape(nb_days * nb_assets, 1)

        # Pour un MLP, input = applatissement de la séquence journalière
        if model_type == "mlp":
            X_out = X.reshape(nb_days * nb_assets, minutes_per_day * d)

        # Pour les autres modèles : (nb_jours * nb_actifs, 1440, nb_features)
        elif model_type == "cnn":
            X_out = X.reshape(nb_days * nb_assets, minutes_per_day, d)

        elif model_type == "rnn":
            X_out = X.reshape(nb_days * nb_assets, minutes_per_day, d)

        elif model_type == "seq":
            if window is None:
                raise ValueError("window doit être spécifié pour model_type='seq'")

            X_r = X.reshape(nb_days * nb_assets, minutes_per_day, d)

            if nb_days <= window:
                return np.empty((0, window, rows_per_day * d), dtype=np.float32), np.empty((0, nb_assets), dtype=np.float32)

            X_out = np.array([
                X_r[t:t + window].reshape(window, -1)
                for t in range(nb_days * nb_assets - window)
            ], dtype=np.float32)

            y_out = y_out[window:] 

        else:
            raise ValueError(f"Modèle inconnu : '{model_type}'")

        return X_out.astype(np.float32), y_out.astype(np.float32)