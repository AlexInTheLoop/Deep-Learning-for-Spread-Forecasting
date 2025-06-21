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
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

# Gestionnaire de données
from data.DataManager import DataManager

# Modèles de Deep Learning
from dl_models.MLPs import create_mlp_model
from dl_models.CNNs import create_cnn_model
from dl_models.RNNs import LSTM, GRU, TKAN, create_rnn_model

# Métriques
from utils.metrics import compile_models_metrics, compute_estimators_metrics
from utils.visualization_tools import plot_model_metrics

################################################################
####### CONSTRUCTION DES DONNEES (FEATURES ET LABEL) ###########
################################################################

# Paramètres
cryptos = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", 
    "SOLUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT", 
    "MATICUSDT", "TRXUSDT"
    ]
nb_assets = len(cryptos)
nb_epochs = 100
year = 2023
month = 9
daily = True

# Téléchargement et préparation des données
manager = DataManager(cryptos, year, month)
manager.download_and_prepare_data()

# Construction des features journalières
daily_features_paths=manager.build_features(daily=daily,serial_dependency=False)

# Créations d'une variable de features et d'une variable de labels
X, y = manager.build_training_data(daily=daily,serial_dependency=False)

# Création des ensembles d'entraînement, de validation et de test
X_train, y_train, X_val, y_val, X_test, y_test = manager.time_series_features(X,y, daily=daily)

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


################################################################
####### Modèles MLP ############################################
################################################################

# Formatage des données pour les modèles MLP
X_train_mlp, y_train_mlp = manager.format_data(
    X_train_scaled,
    y_train_scaled, 
    model_type='mlp',
    daily=daily, 
    nb_assets=nb_assets
    )
X_val_mlp, y_val_mlp = manager.format_data(
    X_val_scaled,
    y_val_scaled, 
    model_type='mlp',
    daily=daily, 
    nb_assets=nb_assets
    )
X_test_mlp, y_test_mlp = manager.format_data(
    X_test_scaled,
    y_test_scaled, 
    model_type='mlp',
    daily=daily, 
    nb_assets=nb_assets
    )

# Création et entraînement du modèle MLP simple
input_shape = X_train_mlp.shape
simple_mlp_model = create_mlp_model(input_shape, model_type="simple", hidden_dims=[128, 64, 32])
simple_mlp_model.fit(
    X_train_mlp, 
    y_train_mlp, 
    validation_data=(X_val_mlp, y_val_mlp), 
    epochs=nb_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
)

# Création et entraînement du modèle MLP résiduel
input_shape = X_train_mlp.shape
res_mlp_model = create_mlp_model(input_shape, model_type="residual", hidden_dims=[128, 64, 32])
res_mlp_model.fit(
    X_train_mlp, 
    y_train_mlp, 
    validation_data=(X_val_mlp, y_val_mlp), 
    epochs=nb_epochs,
    callbacks=[
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
)

################################################################
####### Modèles CNN ############################################
################################################################

# Formatage des données pour les modèles CNN
X_train_cnn, y_train_cnn = manager.format_data(
    X_train_scaled,
    y_train_scaled, 
    model_type='cnn',
    daily=daily, 
    nb_assets=nb_assets
    )
X_val_cnn, y_val_cnn = manager.format_data(
    X_val_scaled,
    y_val_scaled, 
    model_type='cnn',
    daily=daily, 
    nb_assets=nb_assets
    )
X_test_cnn, y_test_cnn = manager.format_data(
    X_test_scaled,
    y_test_scaled, 
    model_type='cnn',
    daily=daily, 
    nb_assets=nb_assets
    )

# Création et entraînement du modèle CNN simple
input_shape = X_train_cnn.shape[1:]
simple_cnn_model = create_cnn_model(input_shape, model_type="simple")
simple_cnn_model.fit(
    X_train_cnn, 
    y_train_cnn, 
    validation_data=(X_val_cnn, y_val_cnn), 
    epochs=nb_epochs
    )

# Création et entraînement du modèle CNN-BGR
input_shape = X_train_cnn.shape[1:]
bgr_cnn_model = create_cnn_model(input_shape, model_type="bgr")
bgr_cnn_model.fit(
    X_train_cnn, 
    y_train_cnn, 
    validation_data=(X_val_cnn, y_val_cnn), 
    epochs=nb_epochs,
    callbacks=[
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
)

################################################################
####### Modèles RNNs ###########################################
################################################################

# Formatage des données pour les modèles LSTM et GRU
X_train_rnn, y_train_rnn = manager.format_data(
    X_train_scaled,
    y_train_scaled, 
    model_type='rnn',
    daily=daily, 
    nb_assets=nb_assets,
    )
X_val_rnn, y_val_rnn = manager.format_data(
    X_val_scaled,
    y_val_scaled, 
    model_type='rnn',
    daily=daily, 
    nb_assets=nb_assets,
    )
X_test_rnn, y_test_rnn = manager.format_data(
    X_test_scaled,
    y_test_scaled, 
    model_type='rnn',
    daily=daily, 
    nb_assets=nb_assets,
    )

# Création et entraînement du modèle LSTM
input_shape = X_train_rnn.shape[1:]
lstm_model = create_rnn_model(
    input_shape, 
    nb_assets,
    LSTM(units=100, return_sequences=False, dropout=0.2),
    False
)
lstm_model.fit(
    X_train_rnn, 
    y_train_rnn, 
    validation_data=(X_val_rnn, y_val_rnn), 
    epochs=nb_epochs,
    callbacks=[
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ] 
)

# Création et entraînement du modèle GRU
input_shape = X_train_cnn.shape[1:]
gru_model = create_rnn_model(
    input_shape, 
    nb_assets,
    GRU(units=100, return_sequences=False, dropout=0.3),
    False
)
gru_model.fit(
    X_train_rnn, 
    y_train_rnn, 
    validation_data=(X_val_rnn, y_val_rnn), 
    epochs=nb_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
)


# Formatage des données pour le modèle TKAN
X_train_tkan, y_train_tkan = manager.format_data(
    X_train_scaled,
    y_train_scaled, 
    model_type='seq',
    daily=daily, 
    nb_assets=nb_assets,
    window=5
    )

X_val_tkan, y_val_tkan = manager.format_data(
    X_val_scaled,
    y_val_scaled, 
    model_type='seq',
    daily=daily, 
    nb_assets=nb_assets,
    window=5
    )

X_test_tkan, y_test_tkan = manager.format_data(
    X_test_scaled,
    y_test_scaled, 
    model_type='seq',
    daily=daily, 
    nb_assets=nb_assets,
    window=5
    )

# Création et entraînement du modèle TKAN
input_shape = X_train_tkan.shape[1:]
tkan_model = create_rnn_model(
    input_shape, 
    nb_assets,
    TKAN(units=100, num_heads=4, return_sequences=False),
    False
)
tkan_model.fit(
    X_train_tkan, 
    y_train_tkan, 
    validation_data=(X_val_tkan, y_val_tkan), 
    epochs=nb_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
)

################################################################
####### Modèles CNN-RNNs #######################################
################################################################

input_shape = X_train_rnn.shape[1:]

cnn_lstm_model = create_rnn_model(
    input_shape=input_shape,
    nb_assets=nb_assets,
    rnn_layer=LSTM(units=100, return_sequences=False, dropout=0.2),
    use_conv=True,
    conv_filters=64,
    conv_kernel_size=3,
    conv_activation="relu"
)
cnn_lstm_model.fit(
    X_train_rnn, 
    y_train_rnn, 
    validation_data=(X_val_rnn, y_val_rnn), 
    epochs=nb_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
)

cnn_gru_model = create_rnn_model(
    input_shape=input_shape,
    nb_assets=nb_assets,
    rnn_layer=GRU(units=100, return_sequences=False, dropout=0.3),
    use_conv=True,                             
    conv_filters=64,                            
    conv_kernel_size=3,                
    conv_activation="relu"      
)
cnn_gru_model.fit(
    X_train_rnn, 
    y_train_rnn, 
    validation_data=(X_val_rnn, y_val_rnn), 
    epochs=nb_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
)

################################################################
####### Estimateurs paramétriques ##############################
################################################################

X, y = manager.build_training_data(daily=False)
X_train, y_train, X_val, y_val, X_test, y_test = manager.time_series_features(X,y,daily=False)
df_est = manager.compute_and_save_parametric_estimators(X_test[:,:6],sort_mode="day_first")
df_est_metrics = compute_estimators_metrics(df_est, y_test,sort_mode="day_first")


################################################################
####### Comparaison des résultats ##############################
################################################################

df_dl_models = compile_models_metrics(
    models={
        "MLP": simple_mlp_model,
        "Residual Regressor MLP": res_mlp_model,
        "CNN": simple_cnn_model,
        "BGR-CNN": bgr_cnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model,
        "TKAN": tkan_model,
        "CNN-LSTM": cnn_lstm_model,
        "CNN-GRU": cnn_gru_model
    },
    X_test=[
        X_test_mlp, X_test_mlp, 
        X_test_cnn, X_test_cnn,
        X_test_rnn, X_test_rnn, X_test_tkan, 
        X_test_rnn, X_test_rnn
    ],
    y_test=[
        y_test_mlp.reshape(-1,1), y_test_mlp.reshape(-1,1), 
     y_test_cnn.reshape(-1,1), y_test_cnn.reshape(-1,1),
     y_test_rnn.reshape(-1,1), y_test_rnn.reshape(-1,1), y_test_tkan.reshape(-1,1),
     y_test_rnn.reshape(-1,1), y_test_rnn.reshape(-1,1)
    ],
    y_scaler=scaler_y
)

df_res = pd.concat([df_est_metrics.T, df_dl_models], axis=0, ignore_index=False)
df_res.sort_values(by='Score', ascending=False, inplace=True)

fig = plot_model_metrics(df_res)
fig.show()