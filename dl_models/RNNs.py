import os
#os.environ["KERAS_BACKEND"] = "jax"
from keras import ops, layers, Sequential, Input, Model,initializers
import keras
import keras.backend as K


 

class LSTM(layers.Layer):
    def __init__(self, units, return_sequences=False, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.layernorm = layers.LayerNormalization()

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

    def lstm_step(self, x_t, h_prev, c_prev):
        i = ops.sigmoid(ops.dot(x_t, self.W_i) + ops.dot(h_prev, self.U_i) + self.b_i)
        f = ops.sigmoid(ops.dot(x_t, self.W_f) + ops.dot(h_prev, self.U_f) + self.b_f)
        c_tilde = ops.tanh(ops.dot(x_t, self.W_c) + ops.dot(h_prev, self.U_c) + self.b_c)
        o = ops.sigmoid(ops.dot(x_t, self.W_o) + ops.dot(h_prev, self.U_o) + self.b_o)
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
            h_t, c_t = self.lstm_step(x_t, h_t, c_t)
            h_t = self.layernorm(h_t)
            outputs.append(h_t)

            if self.return_sequences:
                outputs.append(h_t)
        
        # Récupération de la séquence ou de l'output final
        if self.return_sequences:
            return ops.stack(outputs, axis=1)
        return h_t

def create_lstm_model(input_shape, nb_assets, units=100, dropout=0.3):
    model = keras.Sequential([
        layers.Input(shape=input_shape), 
        LSTM(units=units, return_sequences=False, dropout=dropout),
        layers.Dense(nb_assets,activation='softplus')               
    ])
    model.compile(optimizer="adam", loss='mse', metrics=["mae"])
    return model

class GRU(layers.Layer):
    def __init__(self, units, return_sequences=False, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.layernorm = layers.LayerNormalization()

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

    def gru_step(self, x_t, h_prev):
        z = ops.sigmoid(ops.dot(x_t, self.W_z) + ops.dot(h_prev, self.U_z) + self.b_z)
        r = ops.sigmoid(ops.dot(x_t, self.W_r) + ops.dot(h_prev, self.U_r) + self.b_r)
        h_hat = ops.tanh(ops.dot(x_t, self.W_h) + ops.dot(r * h_prev, self.U_h) + self.b_h)
        h_t = (1 - z) * h_hat + z * h_prev
        return self.layernorm(h_t)

    def call(self, inputs, training = False):
        time_steps = ops.shape(inputs)[1]
        batch_size = ops.shape(inputs)[0]

        h_t = ops.zeros((batch_size, self.units))
        outputs = []

        for t in range(time_steps):
            x_t = inputs[:, t, :]
            h_t = self.gru_step(x_t, h_t)
            outputs.append(h_t)

        outputs = ops.stack(outputs, axis=1)
        if not self.return_sequences:
            return outputs[:, -1, :]
        return outputs

class TKAN_last(layers.Layer):
    def __init__(self, units, num_heads=4, dropout=0.1, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.return_sequences = return_sequences
        self.dropout_rate = dropout

        self.input_projection = layers.Dense(units)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.ffn = Sequential([
            layers.Dense(units * 2, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        projected_inputs = self.input_projection(inputs)

        attn_output = self.attn(projected_inputs, projected_inputs)
        out1 = self.layernorm1(projected_inputs + attn_output)

        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)

        if not self.return_sequences:
            return out2[:, -1, :]

        return out2 



class PolyActivation(layers.Layer):
    """Activation polynomiale (ordre 2)"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        # trois coefficients par unité
        self.a0 = self.add_weight( shape=(units,),initializer='zeros',name='a0')
        self.a1 = self.add_weight(shape=(units,),initializer='ones',name='a1')
        self.a2 = self.add_weight( shape=(units,),initializer=initializers.RandomNormal(0., 0.1),name='a2') 
        
    def call(self, x):
        return self.a0 + self.a1 * x + self.a2 * (x * x)


class KANFeedForward(layers.Layer):
    """Sous-couche KAN : Dense puis PolyActivation puis Dense"""
    def __init__(self, units, num_branches=32, **kwargs):
        super().__init__(**kwargs)
        self.units        = units
        self.num_branches = num_branches

        self.linear1 = layers.Dense(num_branches, use_bias=True) # projection
        self.poly    = PolyActivation(num_branches) # activation polynomiale
        self.linear2 = layers.Dense(units, use_bias=True) # retour à la dimension d'origine

    def call(self, x):
        h = self.linear1(x)
        h = self.poly(h)
        return self.linear2(h)


class TKAN(layers.Layer):
    """TrueTKAN : MultiHeadAttention + KANFeedForward + résidus + LayerNorm"""
    def __init__(self,units,num_heads: int = 4,dropout: float = 0.1,return_sequences: bool = True,num_branches: int = 32,**kwargs):
                         
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.input_proj  = layers.Dense(units) # projection linéaire 
       
        self.attn = layers.MultiHeadAttention(num_heads=num_heads,key_dim=units) #attention multi-têtes
        self.kan_ffn       = KANFeedForward(units, num_branches)# sous-couche KAN-feed-forward
        self.dropout_layer = layers.Dropout(dropout)# dropout sur la sortie du feed-forward

        # deux normalisations pour les connexions résiduelles
        self.layernorm1    = layers.LayerNormalization()
        self.layernorm2    = layers.LayerNormalization()

    def call(self, inputs, training=None, **kwargs):
        
        z = self.input_proj(inputs)  # projection initiale              
        a = self.attn(z, z, training=training) # multi-head self-attention    
        r1 = self.layernorm1(z + a) # premier résidu + normalisation

        f  = self.kan_ffn(r1)# KAN feed-forward 
        f  = self.dropout_layer(f, training=training) #dropout
        
        r2 = self.layernorm2(r1 + f) # second résidu et normalisation

        if self.return_sequences:
            return r2                             
        return r2[:, -1, :]                 




from keras import layers, models, optimizers

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


def create_rnn_model(
    input_shape,
    nb_assets,
    rnn_layer,
    use_conv=False,
    conv_filters=32,
    conv_kernel_size=3,
    conv_activation="relu"
):
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
    return model

