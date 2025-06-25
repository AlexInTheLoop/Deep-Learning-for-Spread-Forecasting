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
from utils.visualization_tools import evaluate_and_plot

from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error

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
feature_paths = manager_train.load_features(use_serial_dependance)
labels_paths = manager_train.build_labels()

# Création du datamanager pour gérer les données de test (et import)
manager_test: DataManager = DataManager(symbols=cryptos_test, dates=test_period, light=True)
manager_test.download_and_prepare_data()

# Construction des features et labels pour les données de test
feature_paths_test = manager_test.load_features(use_serial_dependance)
labels_paths_test = manager_test.build_labels()


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

X_val_cnn, y_val_cnn = manager_train.format_data(
    X_val_scaled, y_val_scaled,
    model_type='cnn',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TRAIN CNN :", X_val_cnn.shape, y_val_cnn.shape)


# Création et entraînement du CNN
input_shape = X_tr_cnn.shape[1:]     # (1440, 11)
cnn = create_cnn_model(input_shape=input_shape, model_type="simple")
cnn.compile(optimizer='adam', loss='mse')

history = cnn.fit(
    X_tr_cnn, y_tr_cnn,
    validation_data=(X_val_cnn, y_val_cnn),
    epochs=N_MAX_EPOCH,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks
)

# Formatage des features de test
X_te_cnn, y_te_cnn = manager_test.format_data(
    X_test_scaled,
    y_test_scaled,
    model_type='cnn',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TEST CNN :", X_te_cnn.shape, y_te_cnn.shape)

# Test sur CNN
# Make predictions with MLP
y_pred_cnn_train = cnn.predict(X_tr_cnn)
y_pred_cnn_test = cnn.predict(X_te_cnn)

# Compute metrics for MLP
cnn_mse_train = mean_squared_error(y_tr_cnn, y_pred_cnn_train)
cnn_mse_test = mean_squared_error(y_te_cnn, y_pred_cnn_test)
cnn_r2_train = r2_score(y_tr_cnn, y_pred_cnn_train)
cnn_r2_test = r2_score(y_te_cnn, y_pred_cnn_test)

# Print comparison
print("Model Performance Comparison:")
print("\nLinear Regression:")
print(f"Train MSE: {cnn_mse_train:.6f}")
print(f"Test MSE: {cnn_mse_test:.6f}")
print(f"Train R²: {cnn_r2_train:.6f}")
print(f"Test R²: {cnn_r2_test:.6f}")

evaluate_and_plot(
    model    = cnn,
    X        = X_te_cnn,
    y        = y_te_cnn,
    scaler_y = scaler_y,
    title    = "CNN simple - actifs de test",
    history=history
)


# Mise en forme des données pour le MLP
X_tr_mlp, y_tr_mlp = manager_train.format_data(
    X_train_scaled, y_train_scaled,
    model_type='mlp',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TRAIN MLP :", X_tr_mlp.shape, y_tr_mlp.shape)


# Construction et entraînement du modèle MLP
input_shape = X_tr_mlp.shape
mlp = create_mlp_model(
    input_shape=input_shape,
    model_type="simple",
    hidden_dims=[128, 64, 32]
)
mlp.compile(optimizer='adam', loss='mse')

history_mlp = mlp.fit(
    X_tr_mlp, y_tr_mlp,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    verbose=1,
    callbacks=callbacks
)

#Préparation du set de test au même format
X_te_mlp, y_te_mlp = manager_test.format_data(
    X_test_scaled, y_test_scaled,
    model_type='mlp',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TEST  MLP :", X_te_mlp.shape, y_te_mlp.shape)


#Évaluation + visualisation
evaluate_and_plot(
    model    = mlp,
    X        = X_te_mlp,
    y        = y_te_mlp,
    scaler_y = scaler_y,
    title    = "MLP simple - actifs de test",
    history=history_mlp
)

### Mise en forme des données pour un RNN (LSTM)        ###
X_tr_rnn, y_tr_rnn = manager_train.format_data(
    X_train_scaled, y_train_scaled,
    model_type='rnn',          # ← reshape (nb_days, 1440, n_feat)
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TRAIN RNN :", X_tr_rnn.shape, y_tr_rnn.shape)


# Construction & fit du modèle RNN
input_shape = X_tr_rnn.shape[1:]        # (1440, n_feat)
rnn = create_rnn_model(
    input_shape=input_shape,
    nb_assets=1,                         # 1 valeur de spread par séquence
    rnn_layer=LSTM(units=100, return_sequences=False, dropout=0.2),
    use_conv=False
)
rnn.compile(optimizer='adam', loss='mse')

history_rnn = rnn.fit(
    X_tr_rnn, y_tr_rnn,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    verbose=1,
    callbacks=callbacks                 # même liste que pour le CNN/MLP
)


#Préparation du set de test au même format
X_te_rnn, y_te_rnn = manager_test.format_data(
    X_test_scaled, y_test_scaled,
    model_type='rnn',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TEST  RNN :", X_te_rnn.shape, y_te_rnn.shape)


## Évaluation + visualisation
evaluate_and_plot(
    model    = rnn,
    X        = X_te_rnn,
    y        = y_te_rnn,
    scaler_y = scaler_y,                 # pour dé-normaliser le spread
    title    = "LSTM – actifs de test",
    history=history_rnn
)