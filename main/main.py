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
import keras
from sklearn.preprocessing import MinMaxScaler

# Gestionnaire de données
from data.DataManager import DataManager

# Modèles de Deep Learning
from dl_models.MLPs import create_mlp_model,  LRHistory
from dl_models.CNNs import create_cnn_model
from dl_models.RNNs import LSTM, GRU, TKAN, create_rnn_model

# Métriques
from utils.visualization_tools import evaluate_and_plot
from utils.papers_runner import PaperEstimatorsRunner

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
        min_delta=0.00001,
        patience = 10,
        mode = "min",
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.25,
        patience = 5,
        min_delta = 0.00001,
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

# Récupération du nombre d'actifs
nb_asset_train: int = len(cryptos_train)
nb_asset_test: int = len(cryptos_test)

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

# Standardisation des features
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val).reshape(X_val.shape)
X_test_scaled = scaler_X.transform(X_test).reshape(X_test.shape)

# Standardisation des labels (pour pas avoir des valeurs trop proches de 0)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

#Estimation papier
runner = PaperEstimatorsRunner(
    symbols = cryptos_test,
    periods = [(2023, 11)] * len(cryptos_test),
    mode    = "pair",
    light_download = True
)

df_est  = runner.get_estimates()
df_lab  = runner._load_labels_all()
df_all  = df_est.join(df_lab["spread_real"])
y_true  = df_all["spread_real"].values
paper_daily = df_all.drop(columns="spread_real")

# Métriques (toutes)
df_perf = runner.evaluate()


# Train CNN
X_tr_cnn, y_tr_cnn = manager_train.format_data(
    X_train_scaled, y_train_scaled,
    model_type='cnn',
    daily=False,
    nb_assets=nb_asset_train,
    minutes_per_day=1440
)
print("TRAIN CNN :", X_tr_cnn.shape, y_tr_cnn.shape)

X_val_cnn, y_val_cnn = manager_train.format_data(
    X_val_scaled, y_val_scaled,
    model_type='cnn',
    daily=False,
    nb_assets=nb_asset_train,
    minutes_per_day=1440
)
print("VAL CNN :", X_val_cnn.shape, y_val_cnn.shape)

optimizer = keras.optimizers.Adam(1e-3)

# Création et entraînement du CNN
input_shape = X_tr_cnn.shape[1:]     # (1440, 11)
cnn = create_cnn_model(input_shape=input_shape, model_type="simple")
cnn.summary()

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
    y_test,
    model_type='cnn',
    daily=False,
    nb_assets=nb_asset_test,
    minutes_per_day=1440
)
print("TEST CNN :", X_te_cnn.shape, y_te_cnn.shape)

# Test sur CNN
y_pred_cnn_train = cnn.predict(X_tr_cnn)
y_pred_cnn_test = cnn.predict(X_te_cnn)

# Calcul du spread sur les tests
cnn_spread_pred = manager_test.compute_spread_pred(
    y_pred=y_pred_cnn_test,
    scaler_y = scaler_y
)

evaluate_and_plot(
    model          = cnn,
    X              = X_te_cnn,
    y              = y_te_cnn,
    scaler_y       = scaler_y,
    title          = "CNN",
    history        = history,
    paper_daily    = paper_daily,
    paper_metrics  = df_perf ,
    y_true_daily   = y_true,
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

# Construction du dataframe de validation pour le MLP
X_val_mlp, y_val_mlp = manager_train.format_data(
    X_val_scaled, y_val_scaled,
    model_type='mlp',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("VAL MLP :", X_val_mlp.shape, y_val_mlp.shape)

# Construction et entraînement du modèle MLP
input_shape = X_tr_mlp.shape
mlp = create_mlp_model(
    input_shape=input_shape,
    model_type="simple",
    hidden_dims=[128, 64, 32]
)

history_mlp = mlp.fit(
    X_tr_mlp, y_tr_mlp,
    validation_data=(X_val_mlp, y_val_mlp),
    epochs=N_MAX_EPOCH,
    batch_size=BATCH_SIZE,
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

# Test sur CNN
y_pred_mlp_test = mlp.predict(X_te_mlp)

# Calcul du spread sur les tests
mlp_spread_pred = manager_test.compute_spread_pred(
    y_pred=y_pred_mlp_test,
    scaler_y = scaler_y
)

#Évaluation + visualisation
evaluate_and_plot(
    model          = mlp,
    X              = X_te_mlp,
    y              = mlp_spread_pred,
    title          = "MLP",
    history        = history_mlp,
    paper_daily    = paper_daily,
    paper_metrics  = df_perf ,
    y_true_daily   = y_true,
)

"""
LSTM
"""

# Mise en forme TRAIN / VAL pour le RNN (ici LSTM)
X_tr_rnn, y_tr_rnn = manager_train.format_data(
    X_train_scaled, y_train_scaled,
    model_type='rnn',
    daily=False,
    nb_assets=6,
    minutes_per_day=1440
)
X_val_rnn, y_val_rnn = manager_train.format_data(
    X_val_scaled, y_val_scaled,
    model_type='rnn',
    daily=False,
    nb_assets=6,
    minutes_per_day=1440
)

print("TRAIN RNN :", X_tr_rnn.shape, y_tr_rnn.shape)
print("VAL   RNN :", X_val_rnn.shape, y_val_rnn.shape)

input_shape = X_tr_rnn.shape[1:]
lstm_model = create_rnn_model(
    input_shape = input_shape,
    nb_assets   = 6,
    rnn_layer   = LSTM(units=100, return_sequences=False, dropout=0.2),
    use_conv    = False
)

history_lstm = lstm_model.fit(
    X_tr_rnn, y_tr_rnn,
    validation_data = (X_val_rnn, y_val_rnn),
    epochs          = N_MAX_EPOCH,
    batch_size      = BATCH_SIZE,
    verbose         = 1,
    callbacks       = callbacks
)


# Préparation du JEU DE TEST
X_te_lstm, y_te_lstm = manager_test.format_data(
    X_test_scaled, y_test_scaled,
    model_type='rnn',
    daily=False,
    nb_assets=2,
    minutes_per_day=1440
)
print("TEST RNN :", X_te_lstm.shape, y_te_lstm.shape)

# Test sur LSTM
y_pred_lstm_test = lstm_model.predict(X_te_lstm)

# Calcul du spread sur les tests
lstm_spread_pred = manager_test.compute_spread_pred(
    y_pred=y_pred_lstm_test,
    scaler_y = scaler_y
)


# Évaluation + visualisation + comparaison papier
df_compare_rnn = evaluate_and_plot(
    model          = lstm_model,
    X              = X_te_lstm,
    y              = lstm_spread_pred,
    scaler_y       = scaler_y,
    title          = "LSTM - actifs de test",
    history        = history_lstm,
    paper_metrics  = df_perf,
    paper_daily    = paper_daily,
    y_true_daily   = y_true
)

"""
GRU
"""

X_tr_gru, y_tr_gru = manager_train.format_data(
    X_train_scaled, y_train_scaled,
    model_type='rnn',          # → (nb_days, 1440, n_feat)
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)

print("TRAIN GRU :", X_tr_gru.shape, y_tr_gru.shape)

input_shape = X_tr_gru.shape[1:]   # (1440, n_feat)
gru_model = create_rnn_model(
    input_shape = input_shape,
    nb_assets   = 1,
    rnn_layer   = GRU(units=100, return_sequences=False, dropout=0.3),
    use_conv    = False
)

gru_model.compile(optimizer='adam', loss='mse')

history_gru = gru_model.fit(
    X_tr_gru, y_tr_gru,
    validation_split=0.2,
    epochs          = 60,
    batch_size      = 32,
    verbose         = 1,
    callbacks       = callbacks
)

X_val_gru, y_val_gru = manager_train.format_data(
    X_val_scaled, y_val_scaled,
    model_type='rnn',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)

print("VAL   GRU :", X_val_gru.shape, y_val_gru.shape)

X_te_gru, y_te_gru = manager_test.format_data(
    X_test_scaled, y_test_scaled,
    model_type='rnn',
    daily=False,
    nb_assets=1,
    minutes_per_day=1440
)
print("TEST GRU :", X_te_gru.shape, y_te_gru.shape)

# Test sur CNN
y_pred_gru_test = gru_model.predict(X_te_gru)

# Calcul du spread sur les tests
gru_spread_pred = manager_test.compute_spread_pred(
    y_pred=y_pred_gru_test,
    scaler_y = scaler_y
)



df_compare_gru = evaluate_and_plot(
    model          = gru_model,
    X              = X_te_gru,
    y              = gru_spread_pred,
    scaler_y       = scaler_y,
    title          = "GRU – actifs de test",
    history        = history_gru,
    paper_metrics  = df_perf,
    paper_daily    = paper_daily,
    y_true_daily   = y_true
)

a = 3