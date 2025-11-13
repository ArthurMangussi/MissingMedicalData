import tensorflow as tf
from tensorflow.keras import Model, layers, backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import numpy as np
import keras.ops as ops

class VAELossLayer(layers.Layer):
    """
    Camada wrapper para injetar a perda VAE (Reconstrução + KL) no grafo.
    """
    def __init__(self, vae_loss_fn, input_shape, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.vae_loss_fn = vae_loss_fn
        # Guardamos as dimensões da imagem para usar no cálculo da perda
        self.width = input_shape[0]
        self.height = input_shape[1]

    def call(self, inputs):
        # inputs: [m_input, m_output, masks, z_mean, z_log_var]
        m_input, m_output, masks, z_mean, z_log_var = inputs
        
        # O cálculo da perda agora é executado DENTRO do grafo simbólico
        vae_loss = self.vae_loss_fn(m_input, m_output, masks, z_mean, z_log_var, self.width, self.height)
        
        # Adiciona a perda ao modelo. self.add_loss é a forma simbólica.
        self.add_loss(vae_loss) 
        
        # Retorna a saída principal (m_output) para manter o fluxo do grafo.
        return m_output
    
@dataclass
class ConfigVAE:
    """
    Data class with the configuration for the Variational Autoencoder architecture.
    All attributes are self-explanatory. However, there are a few important notes:
        - The ``validation_split`` value is ignored if validation data is supplied to the ``fit()`` method.
        - Each value of the ``filters`` list specifies the number of output filters of each convolutional layer.
            Therefore, for multi-layer architectures, this list would contain two or more values.
        - Each value of the ``kernels`` list specifies the kernel size of each convolutional layer.
            Therefore, for multi-layer architectures, this list would contain two or more values.
        - Each value of the ``neurons`` list specifies the number of neurons in each hidden layer.
            Therefore, for multi-layer architectures, this list would contain two or more values.
        - The ``dropout`` rates are optional. However, if they are supplied, an independent rate for each
            convolutional layer (``dropout_conv``) and dense hidden layer (``dropout_fc``) must be defined.
        - The ``missing_values_weight`` can be used to assign a higher or lower weight to the missing values
            on the reconstruction error component of the loss function.
        - The ``kullback_leibler_weight`` can be used to assign a higher or lower weight to the Kulback-Leibler
            divergence component of the loss function.
        - The only mandatory attribute is the input shape.
    """
    loss: Any = MeanSquaredError()
    metrics: List = field(default_factory=lambda: [])
    epochs: int = 200
    batch_size: int = 32
    validation_split: float = 0.0
    verbose: int = 1
    filters: List[int] = field(default_factory=lambda: [])
    kernels: List[float] = field(default_factory=lambda: [])
    activation: str = "relu"
    output_activation: str = "sigmoid"
    neurons: List[int] = field(default_factory=lambda: [10])
    dropout: List[float] = None
    latent_dimension: int = 1
    weights_initializer: str = "glorot_normal"
    l2_lambda: float = 0.01
    learning_rate: float = 0.001
    reduce_learning_rate_factor: float = 0.2
    reduce_learning_rate_patience: int = 10
    early_stopping_patience: int = 10
    input_shape: Tuple = None
    missing_values_weight: int = 1
    kullback_leibler_weight: int = 1


class Sampling(layers.Layer):
    """
    Custom layer which implements the reparameterization trick of the Variational Autoencoder.
    """
    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder:
    """
    Implementation of the Variational Autoencoder.

    Attributes:
        _config (ConfigVAE): Data class with the configuration for the Variational Autoencoder architecture.
        _model: Complete Keras model (encoding and decoding), obtained after the fitting process.
        _encoder: Keras model of the encoding side, obtained after the fitting process.
        _decoder: Keras model of the decoding side, obtained after the fitting process.
        _fitted (bool): Boolean flag used to indicate if the ``fit()`` method was already invoked.
    """
    def __init__(self, config: ConfigVAE):
        self._config = config
        self._model = None
        self._encoder = None
        self._decoder = None
        self._fitted = False

    def _create_auto_encoder(self, input_shape):
        """
        Creates the Variational Autoencoder Keras models with the architecture details provided in ``_config``.

        Args:
            input_shape: Input shape of the Variational Autoencoder.

        Returns: Tuple with three Keras models (complete model, encoding and decoding sides) and the layers
            needed by the custom loss function.

        """
        x = enc_input = tf.keras.Input(shape=input_shape)
        masks = tf.keras.Input(shape=input_shape)

        x = layers.Concatenate(axis=-1)([enc_input, masks])

        for i, f in enumerate(self._config.filters):
            k = self._config.kernels[i] if isinstance(self._config.kernels, list) else self._config.kernels
            x = layers.Conv2D(f, kernel_size=k, padding='same', strides=2, activation=self._config.activation,
                              kernel_regularizer=l2(self._config.l2_lambda),
                              kernel_initializer=self._config.weights_initializer)(x)

        shape_before_flat = None
        if len(self._config.filters) > 0:
            shape_before_flat = list(filter(None, x.shape))
            x = layers.Flatten()(x)

        for i, n in enumerate(self._config.neurons):
            x = layers.Dense(n, activation=self._config.activation,
                             kernel_regularizer=l2(self._config.l2_lambda),
                             kernel_initializer=self._config.weights_initializer)(x)
            if self._config.dropout is not None:
                x = layers.Dropout(rate=self._config.dropout[i])(x)

        z_mean_layer = layers.Dense(self._config.latent_dimension,
                                    kernel_initializer=self._config.weights_initializer, name="z_mean")
        z_log_var_layer = layers.Dense(self._config.latent_dimension,
                                       kernel_initializer=self._config.weights_initializer, name="z_log_var")
        z_mean = z_mean_layer(x)
        z_log_var = z_log_var_layer(x)
        x = [z_mean, z_log_var, Sampling()([z_mean, z_log_var])]

        m_encoder = Model([enc_input, masks], x, name='encoder')
        dec_input = tf.keras.Input(shape=(self._config.latent_dimension,))
        x = dec_input

        for i, n in reversed(list(enumerate(self._config.neurons))):
            x = layers.Dense(n, activation=self._config.activation,
                             kernel_regularizer=l2(self._config.l2_lambda),
                             kernel_initializer=self._config.weights_initializer)(x)
            if self._config.dropout is not None:
                x = layers.Dropout(rate=self._config.dropout[i])(x)

        if len(self._config.filters) > 0:
            x = layers.Dense(units=np.prod(shape_before_flat), activation=self._config.activation,
                             kernel_regularizer=l2(self._config.l2_lambda),
                             kernel_initializer=self._config.weights_initializer)(x)
            x = layers.Reshape(target_shape=shape_before_flat)(x)

            for i, f in reversed(list(enumerate(self._config.filters))):
                k = self._config.kernels[i] if isinstance(self._config.kernels, list) else self._config.kernels
                x = layers.Conv2DTranspose(f, kernel_size=k, strides=2, padding='same',
                                           activation=self._config.activation,
                                           kernel_regularizer=l2(self._config.l2_lambda),
                                           kernel_initializer=self._config.weights_initializer)(x)

            x = layers.Conv2DTranspose(filters=list(filter(None, enc_input.shape))[2],
                                       kernel_size=self._config.kernels, strides=1, padding='same',
                                       activation=self._config.output_activation,
                                       kernel_regularizer=l2(self._config.l2_lambda),
                                       kernel_initializer=self._config.weights_initializer)(x)
        else:
            x = layers.Dense(units=list(filter(None, enc_input.get_shape().as_list()))[0],
                             activation=self._config.output_activation,
                             kernel_regularizer=l2(self._config.l2_lambda),
                             kernel_initializer=self._config.weights_initializer)(x)

        m_decoder = Model(dec_input, x, name='decoder')
        enc_output = m_decoder(m_encoder([enc_input, masks])[2])
        
        input_size = self._config.input_shape
        loss_layer = VAELossLayer(
            vae_loss_fn=self._vae_wl_loss, 
            input_shape=input_size, 
            name='vae_loss_injector'
        )
        _ = loss_layer([enc_input, enc_output, masks, z_mean, z_log_var])

        m_global = Model([enc_input, masks], enc_output, name='vae')

        return m_global, m_encoder, m_decoder

    def _vae_wl_loss(self, y_true, y_pred, masks, z_mean, z_log_var, width, height):
        """
        Custom loss function of the Variational Autoencoder.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            masks: Missing values masks.
            z_mean: Mean being learned by the VAE.
            z_log_var: Log transform of the variance being learned by the VAE.
            width: Width of the training data.
            height: Height of the training data.

        Returns: Loss value.

        """
        flatten_layer = layers.Flatten() 
        y_true_flat = flatten_layer(y_true * masks)
        y_pred_flat = flatten_layer(y_pred * masks)
        
        bce_loss_mv = self._config.loss(y_true_flat, y_pred_flat) * width * height
        
        # Repita para bce_loss_ov
        y_true_ov_flat = flatten_layer(y_true * (masks * -1 + 1))
        y_pred_ov_flat = flatten_layer(y_pred * (masks * -1 + 1))
        
        bce_loss_ov = self._config.loss(y_true_ov_flat, y_pred_ov_flat) * width * height
        
        bce_loss = bce_loss_ov + self._config.missing_values_weight * bce_loss_mv
        
        # As operações KL usam K.sum, que é mais tolerante e deve estar OK
        kl_loss = - 0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
        vae_loss = ops.mean(bce_loss + self._config.kullback_leibler_weight * kl_loss)
        return vae_loss

    def fit(self, x_train, x_mask, y_train, x_val=None, x_val_mask=None, y_val=None):
        """
        Fits the Variational Autoencoder model.

        Args:
            x_train: Training data.
            x_mask: Missing values mask of the training data.
            y_train: Target data.
            x_val (optional): Validation training data.
            x_val_mask (optional): Missing values mask of the validation training data.
            y_val (optional): Validation target data.

        """
        self._model, self._encoder, self._decoder = self._create_auto_encoder(self._config.input_shape)
        # m_input, m_output, masks, z_mean, z_log_var = loss_params
        # self._model.add_loss(self._vae_wl_loss(m_input, m_output, masks, z_mean,
                                            #    z_log_var, x_train.shape[1], x_train.shape[2]))
        self._model.compile(optimizer=Adam(self._config.learning_rate), loss=None, metrics=self._config.metrics)

        callbacks = []
        if self._config.reduce_learning_rate_patience > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=self._config.reduce_learning_rate_factor,
                                               patience=self._config.reduce_learning_rate_patience, min_lr=0))

        if self._config.early_stopping_patience > 0:
            callbacks.append(EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                                           patience=self._config.early_stopping_patience))

        fit_args = {
            "epochs": self._config.epochs,
            "batch_size": self._config.batch_size,
            "callbacks": callbacks,
            "validation_split": self._config.validation_split,
            "verbose": self._config.verbose
        }

        if x_val is not None and y_val is not None:
            fit_args["validation_data"] = ([x_val, x_val_mask], y_val)

        self._model.fit([x_train, x_mask], y_train, **fit_args)
        self._fitted = True

    def encode(self, x, x_mask):
        """
        Encodes new data points with the Variational Autoencoder.

        Args:
            x: Data to be encoded.
            x_mask: Missing values mask of the data to be encoded.

        Returns: The encoded representation of ``x``.

        """
        if not self._fitted:
            raise RuntimeError("The fit method must be called before encode.")
        return self._encoder.predict([x, x_mask])

    def decode(self, x):
        """
        Decodes encoded representations with the Variational Autoencoder.

        Args:
            x: Data to be decoded.

        Returns: The decoded data from the encoded representations supplied in ``x``.

        """
        if not self._fitted:
            raise RuntimeError("The fit method must be called before decode.")
        return self._decoder.predict(x)

    def encode_and_decode(self, x, x_mask):
        """
        Encodes and decodes new data points with the Variational Autoencoder.

        Args:
            x: Data to be encoded.
            x_mask: Missing values mask of the data to be encoded.

        Returns: The decoded data from the new data points supplied in ``x``.

        """
        if not self._fitted:
            raise RuntimeError("The fit method must be called before encode and/or decode.")
        return self._model.predict([x, x_mask])
