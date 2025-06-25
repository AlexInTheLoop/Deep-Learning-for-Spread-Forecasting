################################################################
####### IMPORTATION DES MODULES ################################
################################################################

# Set up
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importation des modules nécessaires
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

# Gestionnaire de données
from data.DataManager import DataManager

# Modèles de Deep Learning
from dl_models.MLPs import create_mlp_model,  LRHistory
from dl_models.CNNs import create_cnn_model
from dl_models.RNNs import LSTM, GRU, TKAN, create_rnn_model

# Métriques
from utils.metrics import compile_models_metrics, compute_estimators_metrics
from utils.visualization_tools import plot_model_metrics

###################################################
####### VARIABLES DE BASE ET CALLBACKS ############
###################################################

keras.utils.set_random_seed(72)

# Nombre d'itération / taille de batch
N_MAX_EPOCH = 100
BATCH_SIZE = 32

# Liste de callbacks à utiliser dans les modèles
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience = 5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.05,
        patience = 5,
        min_delta = 1e-6,
        verbose = 0
    ),

    LRHistory()
]

################################################################
####### CONSTRUCTION DES DONNEES (FEATURES ET LABEL) ###########
################################################################

# Paramètres

# Définition des périodes de récupération
train_period: list = [(2023,7), (2023,8), (2023, 9), (2023,10)]
test_period: list = [(2023,11)]

# Choix lié à l'utilisation des indicateurs de dépendance sérielle
use_serial_dependance: bool = False

# Définition des actifs à récupérer
cryptos_train: list = ["ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
cryptos_test: list = ["MATICUSDT", "TRXUSDT"]

# Création du datamanager pour gérer les données d'entrainement + import
manager_train: DataManager = DataManager(symbols=cryptos_train, dates=train_period, light=True)
manager_train.download_and_prepare_data()

# Construction des features / labels pour les données d'entrainement
feature_paths = manager_train.load_features(use_serial_dependance, clean_features=True, use_tick_size=True)
labels_paths = manager_train.build_labels(clean_labels=True)

# Création du datamanager pour gérer les données de test (et import)
manager_test: DataManager = DataManager(symbols=cryptos_test, dates=test_period, light=True)
manager_test.download_and_prepare_data()

# Construction des features et labels pour les données de test
feature_paths_test = manager_test.load_features(use_serial_dependance, clean_features=True, use_tick_size=True)
labels_paths_test = manager_test.build_labels(clean_labels=True)


# Test dataframe de construction de test et val
X_train, X_val, y_train, y_val = manager_train.build_train_val_dataset()


# Construction des tests
X_test, y_test = manager_test.build_train_val_dataset(is_test=True)

# Normalisation des features
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Normalisation des features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val).reshape(X_val.shape)
X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape)

# Train CNN
X_tr_cnn, y_tr_cnn = manager_train.format_data(
    X_train_scaled, y_train_scaled,
    model_type='cnn',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TRAIN CNN :", X_tr_cnn.shape, y_tr_cnn.shape)

# Création et entraînement du CNN
input_shape = X_tr_cnn.shape[1:]     # (1440, 11)
cnn = create_cnn_model(input_shape=input_shape, model_type="simple")
cnn.compile(optimizer='adam', loss='mse')

history = cnn.fit(
    X_tr_cnn, y_tr_cnn,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(patience=5,
                                             restore_best_weights=True)]
)

