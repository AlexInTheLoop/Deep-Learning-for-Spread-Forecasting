from keras import Model, layers

class MLP(Model):
    """
    MLP classique entièrement connecté, sans normalisation ni skip connections.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], **kwargs):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.hidden_layers = [layers.Dense(h, activation='relu') for h in hidden_dims]
        self.out = layers.Dense(1, activation='softplus')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out(x)

class ResidualMLPBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.units = units
        self.dense1 = layers.Dense(units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units, activation=None)
        self.norm = layers.LayerNormalization()
        self.proj = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.proj = layers.Dense(self.units)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        shortcut = self.proj(inputs) if self.proj else inputs
        x = self.norm(shortcut + x)
        return layers.Activation('relu')(x)

class MLPResidualRegressor(Model):
    """
    MLP profond utilisant des blocs résiduels normalisés pour améliorer l'apprentissage
    dans des tâches complexes ou à forte dimensionnalité.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.input_norm = layers.LayerNormalization()
        self.blocks = [ResidualMLPBlock(h, dropout_rate) for h in hidden_dims]
        self.out = layers.Dense(1, activation='softplus')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x, training=training)
        return self.out(x)

def create_mlp_model(input_shape, model_type="simple", **kwargs):
    """
    Crée un modèle MLP.

    Inputs:
        input_shape (tuple): (sequence_length, input_dim)
        model_type (str): "simple" ou "residual"
        **kwargs: paramètres supplémentaires passés aux classes internes

    Ouput:
        keras.Model: instance du modèle MLP
    """
    seq_len, input_dim = input_shape
    total_input_dim = seq_len * input_dim

    if model_type == "simple":
        model = MLP(input_dim=total_input_dim, **kwargs)
        model.compile(optimizer='adam', loss='mse')
        return model
    elif model_type == "residual":
        model = MLPResidualRegressor(input_dim=total_input_dim, **kwargs)
        model.compile(optimizer='adam', loss='mse')
        return model
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'simple' or 'residual'.")

