import os
#os.environ["KERAS_BACKEND"] = "jax"
from keras import ops, layers, Input, models, optimizers, Model

class LSTM(layers.Layer):
    def __init__(self, units, return_sequences=False, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.layernorm = layers.LayerNormalization()
        self.dropout_layer = layers.Dropout(self.dropout)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Poids d'entrée
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_i')
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_f')
        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_c')
        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_o')

        # Poids récurrents
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_i')
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_f')
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_c')
        self.U_o = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_o')

        # Biais
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')
        self.b_f = self.add_weight(shape=(self.units,), initializer='ones', name='b_f')
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')

        self.built = True

    def lstm_step(self, x_t, h_prev, c_prev, training=False):

        if training:
            x_t = self.dropout_layer(x_t)

        i = ops.sigmoid(self.layernorm(ops.dot(x_t, self.W_i) + ops.dot(h_prev, self.U_i) + self.b_i))
        f = ops.sigmoid(self.layernorm(ops.dot(x_t, self.W_f) + ops.dot(h_prev, self.U_f) + self.b_f))
        c_tilde = ops.tanh(self.layernorm(ops.dot(x_t, self.W_c) + ops.dot(h_prev, self.U_c) + self.b_c))
        o = ops.sigmoid(self.layernorm(ops.dot(x_t, self.W_o) + ops.dot(h_prev, self.U_o) + self.b_o))
        c_t = f * c_prev + i * c_tilde
        h_t = o * ops.tanh(c_t)
        return h_t, c_t

    def call(self, inputs, training=False):
        time_steps = ops.shape(inputs)[1]
        batch_size = ops.shape(inputs)[0]

        h_t = ops.zeros((batch_size, self.units))
        c_t = ops.zeros((batch_size, self.units))

        outputs = []

        for t in range(time_steps):
            x_t = inputs[:, t, :]
            h_t, c_t = self.lstm_step(x_t, h_t, c_t, training=training)
            #h_t = self.layernorm(h_t)
            #if training:
            #    h_t = self.dropout_layer(h_t)
            if self.return_sequences:
                outputs.append(h_t)
        
        # Récupération de la séquence ou de l'output final
        if self.return_sequences:
            return ops.stack(outputs, axis=1)
        else:
            return h_t

class GRU(layers.Layer):
    def __init__(self, units, return_sequences=False, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.layernorm = layers.LayerNormalization()
        self.dropout_layer = layers.Dropout(self.dropout)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_z = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_z')
        self.U_z = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_z')
        self.b_z = self.add_weight(shape=(self.units,), initializer='zeros', name='b_z')

        self.W_r = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_r')
        self.U_r = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_r')
        self.b_r = self.add_weight(shape=(self.units,), initializer='zeros', name='b_r')

        self.W_h = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_h')
        self.U_h = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_h')
        self.b_h = self.add_weight(shape=(self.units,), initializer='zeros', name='b_h')

        self.built = True

    def gru_step(self, x_t, h_prev, training):
        if training:
            x_t = self.dropout_layer(x_t)

        z = ops.sigmoid(self.layernorm(ops.dot(x_t, self.W_z) + ops.dot(h_prev, self.U_z) + self.b_z))
        r = ops.sigmoid(self.layernorm(ops.dot(x_t, self.W_r) + ops.dot(h_prev, self.U_r) + self.b_r))
        h_hat = ops.tanh(self.layernorm(ops.dot(x_t, self.W_h) + ops.dot(r * h_prev, self.U_h) + self.b_h))
        h_t = (1 - z) * h_hat + z * h_prev
        return h_t

    def call(self, inputs, training = False):
        time_steps = ops.shape(inputs)[1]
        batch_size = ops.shape(inputs)[0]

        h_t = ops.zeros((batch_size, self.units))
        outputs = []

        for t in range(time_steps):
            x_t = inputs[:, t, :]
            h_t = self.gru_step(x_t, h_t, training)
            outputs.append(h_t)

        outputs = ops.stack(outputs, axis=1)
        if not self.return_sequences:
            return outputs[:, -1, :]
        return outputs

def create_native_lstm_model(
    input_shape,
    nb_assets   = 1,
    units       = 100,
    dropout     = 0.3,
    lr          = 1e-3,
    clipnorm    = 1.0
):
    """
    LSTM Keras natif + tête Dense(1, softplus).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.LSTM(
        units,
        dropout        = dropout,   # dropout sur les entrées
        recurrent_dropout = 0.0,    # éventuellement 0.1
        return_sequences = False
    )(inputs)

    outputs = layers.Dense(nb_assets, activation="softplus")(x)

    model = models.Model(inputs, outputs)

    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def create_rnn_model_v2(
    input_shape,
    nb_assets,
    rnn_layer,
    use_simple_model=False,
    conv_filters=32,
    conv_kernel_size=3,
    conv_activation="relu",
    lr = 1e-3,
    clipnorm = 1.0
):
    """
    Fonction utilisée pour construire des modèles RNN simples / plus complet
    """
    # Création d'une couche d'inputs
    inputs = Input(shape=input_shape)
    x = inputs

    # Cas où l'utilisateur souhaite utiliser le modèle simple (une seule couche RNN)
    if use_simple_model:
            # Ajout d'une couche RNN (LSTM, GRU)
            x = rnn_layer(x)
            # Récupération des outputs (softplus pour être > 0)
            outputs = layers.Dense(1, activation="softplus")(x)
    # Cas où l'utilisateur souhaite un modèle plus complet (plusieurs couches RNN)
    else:
        # Convolution 1D
        x = layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            padding="same",
            activation=conv_activation
        )(x)

        # Deux couches de RNN (LSTM / GRU / TKAN selon le cas)
        x = rnn_layer()(x)
        x = rnn_layer()(x)

        # Couche d'attention
        x = layers.MultiHeadAttention(num_heads=4, key_dim=x.shape[-1])(x,x,x)

        # Prévision avec un flatten
        x  = layers.Flatten()(x)

        # Ajout de plusieurs couches denses avec fonction d'activation ReLU
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        # Récupération de l'output final avec fonction d'activation softplus
        outputs = layers.Dense(1, activation="softplus")(x)

    # Construction du modèle et optimisation
    model = Model(inputs, outputs)
    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mae"], jit_compile=False)

    # Résumé et récupération
    model.summary()
    return model

def create_rnn_model(
    input_shape,
    nb_assets,
    rnn_layer,
    use_conv=False,
    conv_filters=32,
    conv_kernel_size=3,
    conv_activation="relu"
):
    """
    Fonction utilisée pour construire des convolutions 1D
    """
    inputs = Input(shape=input_shape)

    x = inputs

    if use_conv:
        x = layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            padding="same",
            activation=conv_activation
        )(x)
    x = rnn_layer(x)
    outputs = layers.Dense(1, activation="softplus")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


