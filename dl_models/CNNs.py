from keras import Model, layers
import tensorflow as tf

class CNN(Model):
    def __init__(self, input_shape,output_activation='softplus', **kwargs):
        super().__init__(**kwargs)
        seq_len, in_channels = input_shape

        if seq_len < 8:
            print(f"Séquence trop courte (seq_len={seq_len}) pour un CNN avec 3 poolings. Désactivation des pools.")
            self.use_pooling = False
        else:
            self.use_pooling = True

        self.conv1 = layers.Conv1D(filters=16, kernel_size=5, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool1D(pool_size=2)

        self.conv2 = layers.Conv1D(filters=32, kernel_size=5, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool1D(pool_size=2)

        self.conv3 = layers.Conv1D(filters=64, kernel_size=3, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPool1D(pool_size=2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.fc2 = layers.Dense(1, activation=output_activation)  # sortie scalaire, toujours positive

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        if self.use_pooling:
            x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        if self.use_pooling:
            x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        if self.use_pooling:
            x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.fc2(x)


class BGR(Model):
    """
    CNN 1D profond et robuste pour séries multivariées.
    """
    def __init__(self,
                 input_shape,
                 filtres=[16, 32, 64],
                 kernel_sizes=[5, 5, 3],
                 hidden_dim=128,
                 dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        seq_len, in_channels = input_shape
        self.num_blocks = len(filtres)

        # Évaluer si pooling possible
        min_required_len = 2 ** self.num_blocks
        if seq_len < min_required_len:
            print(f"Séquence trop courte (seq_len={seq_len}) pour {self.num_blocks} blocs avec pooling.")
            print(f"Pooling désactivé")
            self.use_pooling = False
        else:
            self.use_pooling = True

        # Construction dynamique des blocs convolutionnels
        self.conv_blocks = []
        for i in range(self.num_blocks):
            conv = layers.Conv1D(filters=filtres[i],
                                 kernel_size=kernel_sizes[i],
                                 padding='same',
                                 activation=None)
            bn = layers.BatchNormalization()
            pool = layers.MaxPool1D(pool_size=2)
            self.conv_blocks.append((conv, bn, pool))

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.fc2 = layers.Dense(1, activation='softplus')  # sortie scalaire positive

    def call(self, inputs, training=False):
        x = inputs
        for conv, bn, pool in self.conv_blocks:
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
            if self.use_pooling:
                x = pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.fc2(x)
    

def create_cnn_model(input_shape, model_type="simple",output_activation='softplus', **kwargs):
    """
    Crée un modèle CNN compatible avec une entrée 3D (batch, seq_len, input_dim).
    """
    if model_type == "simple":
        model = CNN(input_shape=input_shape,output_activation=output_activation,**kwargs)
        model.compile(optimizer='adam', loss='mse')
        return model
    elif model_type == "bgr":
        model = BGR(input_shape=input_shape, **kwargs)
        model.compile(optimizer='adam', loss='mse')
        return model
    else:
        raise ValueError(f"Modèle inconnu '{model_type}'. Utiliser 'simple' ou 'bgr'.")