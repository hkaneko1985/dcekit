# -*- coding: utf-8 -*-

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, LeakyReLU,
    Flatten, Dense, Lambda
)
from tensorflow.keras import backend as K


class EncoderPattern:
    """
    Encoder builder class with selectable CNN architecture patterns.
    cnn_pattern:
        "0" -> Baseline spec (Conv1D x3)
        "1" -> Modified kernel size and stride (Conv1D x3)
        "2" -> One additional Conv1D layer (Conv1D x4)
        "3" -> Fine-grained receptive field expansion (Conv1D x3)
    """

    def __init__(self, network_pattern="1", cnn_pattern="0",
                 max_filters=128, filter_size=[5, 3, 3], strides=[2, 2, 1]):
        self.network_pattern = str(network_pattern)
        self.cnn_pattern     = str(cnn_pattern)
        self.max_filters     = max_filters
        self.filter_size     = filter_size
        self.strides         = strides

    def build(self, input_shape):
        """
        Build encoder model.

        Parameters
        ----------
        input_shape : tuple
            (input_dim, channel_dim)

        Returns
        -------
        encoder   : tf.keras.Model
        latent_dim : int
        map_size   : int
            Sequence length after Conv1D blocks (used by decoder).
        """

        encoder_inputs = Input(shape=input_shape, name="encoder_input")
        x = encoder_inputs

        # ==================== CNN architecture branch ====================
        if self.cnn_pattern == "0":
            # Baseline pattern
            conv_params = [
                (self.max_filters // 4, 5, 2),
                (self.max_filters // 2, 3, 2),
                (self.max_filters,      3, 1),
            ]
        elif self.cnn_pattern == "1":
            # Modified kernel size and stride
            conv_params = [
                (self.max_filters // 4, 3, 1),
                (self.max_filters // 2, 3, 2),
                (self.max_filters,      3, 2),
            ]
        elif self.cnn_pattern == "2":
            # Additional Conv1D layer
            conv_params = [
                (self.max_filters // 8, 3, 1),
                (self.max_filters // 4, 3, 2),
                (self.max_filters // 2, 3, 2),
                (self.max_filters,      3, 1),
            ]
        elif self.cnn_pattern == "3":
            # Fine-grained receptive field expansion (no extra layers)
            conv_params = [
                (self.max_filters // 4, 5, 2),
                (self.max_filters // 2, 3, 2),
                (self.max_filters,      3, 1),
            ]
        else:
            raise ValueError(f"Invalid cnn_pattern: {self.cnn_pattern}")

        # ==================== Conv1D block construction ====================
        for i, (f, k, s) in enumerate(conv_params, start=1):
            x = Conv1D(f, k, strides=s, padding='same', name=f"enc_conv{i}")(x)
            x = BatchNormalization(name=f"enc_bn{i}")(x)
            x = LeakyReLU(0.2, name=f"enc_act{i}")(x)

        map_size = int(x.shape[1])
        x = Flatten(name="enc_flatten")(x)

        # ==================== Fully connected layers (selected by network_pattern) ====================
        if self.network_pattern == "0":
            x = Dense(1024, activation="sigmoid")(x)
            latent_dim = 256

        elif self.network_pattern == "1":
            latent_dim = 256

        elif self.network_pattern == "2":
            latent_dim = 512

        elif self.network_pattern == "3":
            x = Dense(1024, activation="sigmoid")(x)
            latent_dim = 256

        elif self.network_pattern == "4":
            x = Dense(2048, activation="sigmoid")(x)
            latent_dim = 1024

        elif self.network_pattern == "5":
            x = Dense(1024, activation="sigmoid")(x)
            x = Dense(512,  activation="sigmoid")(x)
            latent_dim = 256

        elif self.network_pattern == "6":
            x = Dense(2048, activation="sigmoid")(x)
            x = Dense(1024, activation="sigmoid")(x)
            latent_dim = 512

        else:
            raise ValueError(f"Invalid network_pattern: {self.network_pattern}")

        # ==================== Latent space output ====================
        z_mean    = Dense(latent_dim, activation="linear", name="z_mean")(x)
        z_log_var = Dense(latent_dim, activation="linear", name="z_log_var")(x)

        def sampling(args):
            mu, log_var = args
            eps = K.random_normal(shape=(K.shape(mu)[0], latent_dim))
            return mu + K.exp(0.5 * log_var) * eps

        z = Lambda(sampling, name="z")([z_mean, z_log_var])
        encoder = Model(encoder_inputs, [z, z_mean, z_log_var],
                        name=f"encoder_cnn{self.cnn_pattern}_pat{self.network_pattern}")

        return encoder, latent_dim, map_size