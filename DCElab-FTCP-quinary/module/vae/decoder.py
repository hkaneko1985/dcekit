<<<<<<< HEAD
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, BatchNormalization, Activation, Conv2DTranspose,
    Reshape, Lambda
)


class DecoderPattern:
    """
    Decoder builder class with selectable CNN architecture patterns.
    cnn_pattern:
        "0" -> Baseline spec (Conv2DTranspose x3)
        "1" -> Modified kernel size and stride (Conv2DTranspose x3)
        "2" -> One additional Conv2DTranspose layer (Conv2DTranspose x4)
        "3" -> Fine-grained upsampling counterpart of CNN encoder pattern 3
    """

    def __init__(self, network_pattern="1", cnn_pattern="0",
                 max_filters=128, filter_size=[5, 3, 3], strides=[2, 2, 1],
                 channel_dim=1, map_size=None):
        self.network_pattern = str(network_pattern)
        self.cnn_pattern     = str(cnn_pattern)
        self.max_filters     = max_filters
        self.filter_size     = filter_size
        self.strides         = strides
        self.channel_dim     = channel_dim
        self.map_size        = map_size

    def get_latent_dim(self):
        """Return latent_dim corresponding to the selected network_pattern."""
        if self.network_pattern in ["0", "1", "3", "5"]:
            return 256
        elif self.network_pattern in ["2", "6"]:
            return 512
        elif self.network_pattern == "4":
            return 1024
        else:
            raise ValueError(f"Invalid network_pattern: {self.network_pattern}")

    def build(self):
        """
        Build decoder model.

        Returns
        -------
        decoder : tf.keras.Model
        """
        latent_dim    = self.get_latent_dim()
        latent_inputs = Input(shape=(latent_dim,), name="z_input")
        xd = latent_inputs

        # ==================== Fully connected layers (selected by network_pattern) ====================
        if self.network_pattern == "0":
            xd = Dense(1024, activation="relu")(xd)

        elif self.network_pattern in ["1", "2"]:
            pass # no FC layer between z and Conv2DTranspose

        elif self.network_pattern == "3":
            xd = Dense(1024, activation="relu")(xd)

        elif self.network_pattern == "4":
            xd = Dense(2048, activation="relu")(xd)

        elif self.network_pattern == "5":
            xd = Dense(512,  activation="relu")(xd)
            xd = Dense(1024, activation="relu")(xd)

        elif self.network_pattern == "6":
            xd = Dense(1024, activation="relu")(xd)
            xd = Dense(2048, activation="relu")(xd)

        else:
            raise ValueError(f"Invalid network_pattern: {self.network_pattern}")

        # ==================== Conv2DTranspose architecture branch ====================
        xd = Dense(self.max_filters * self.map_size, activation='relu')(xd)
        xd = Reshape((self.map_size, 1, self.max_filters))(xd)
        xd = BatchNormalization()(xd)

        if self.cnn_pattern == "0":
            # Baseline spec (3 layers)
            conv_params = [
                (self.max_filters // 2, (3, 1), (1, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.channel_dim,      (5, 1), (2, 1)),
            ]

        elif self.cnn_pattern == "1":
            # Modified kernel size and stride
            conv_params = [
                (self.max_filters // 2, (3, 1), (2, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.channel_dim,      (3, 1), (1, 1)),
            ]

        elif self.cnn_pattern == "2":
            # Additional Conv2DTranspose layer (4 layers)
            conv_params = [
                (self.max_filters // 2, (3, 1), (2, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.max_filters // 8, (3, 1), (1, 1)),
                (self.channel_dim,      (3, 1), (1, 1)),
            ]

        elif self.cnn_pattern == "3":
            # Fine-grained upsampling counterpart of CNN3
            conv_params = [
                (self.max_filters // 2, (3, 1), (1, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.channel_dim,      (5, 1), (2, 1)),
            ]

        else:
            raise ValueError(f"Invalid cnn_pattern: {self.cnn_pattern}")

        # ==================== Conv2DTranspose block construction ====================
        for i, (f, k, s) in enumerate(conv_params[:-1], start=1):
            xd = Conv2DTranspose(f, k, strides=s, padding='same', name=f"dec_convT{i}")(xd)
            xd = BatchNormalization(name=f"dec_bn{i}")(xd)
            xd = Activation('relu', name=f"dec_act{i}")(xd)

        # Output layer (sigmoid activation, no batch norm)
        f, k, s = conv_params[-1]
        xd = Conv2DTranspose(f, k, strides=s, padding='same', name="dec_convT_last")(xd)
        xd = Activation('sigmoid', name="dec_output_act")(xd)

        decoder_outputs = Lambda(lambda t: tf.squeeze(t, axis=2), name="dec_squeeze")(xd)
        decoder = Model(latent_inputs, decoder_outputs,
                        name=f"decoder_cnn{self.cnn_pattern}_pat{self.network_pattern}")

=======
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, BatchNormalization, Activation, Conv2DTranspose,
    Reshape, Lambda
)


class DecoderPattern:
    """
    Decoder builder class with selectable CNN architecture patterns.
    cnn_pattern:
        "0" -> Baseline spec (Conv2DTranspose x3)
        "1" -> Modified kernel size and stride (Conv2DTranspose x3)
        "2" -> One additional Conv2DTranspose layer (Conv2DTranspose x4)
        "3" -> Fine-grained upsampling counterpart of CNN encoder pattern 3
    """

    def __init__(self, network_pattern="1", cnn_pattern="0",
                 max_filters=128, filter_size=[5, 3, 3], strides=[2, 2, 1],
                 channel_dim=1, map_size=None):
        self.network_pattern = str(network_pattern)
        self.cnn_pattern     = str(cnn_pattern)
        self.max_filters     = max_filters
        self.filter_size     = filter_size
        self.strides         = strides
        self.channel_dim     = channel_dim
        self.map_size        = map_size

    def get_latent_dim(self):
        """Return latent_dim corresponding to the selected network_pattern."""
        if self.network_pattern in ["0", "1", "3", "5"]:
            return 256
        elif self.network_pattern in ["2", "6"]:
            return 512
        elif self.network_pattern == "4":
            return 1024
        else:
            raise ValueError(f"Invalid network_pattern: {self.network_pattern}")

    def build(self):
        """
        Build decoder model.

        Returns
        -------
        decoder : tf.keras.Model
        """
        latent_dim    = self.get_latent_dim()
        latent_inputs = Input(shape=(latent_dim,), name="z_input")
        xd = latent_inputs

        # ==================== Fully connected layers (selected by network_pattern) ====================
        if self.network_pattern == "0":
            xd = Dense(1024, activation="relu")(xd)

        elif self.network_pattern in ["1", "2"]:
            pass # no FC layer between z and Conv2DTranspose

        elif self.network_pattern == "3":
            xd = Dense(1024, activation="relu")(xd)

        elif self.network_pattern == "4":
            xd = Dense(2048, activation="relu")(xd)

        elif self.network_pattern == "5":
            xd = Dense(512,  activation="relu")(xd)
            xd = Dense(1024, activation="relu")(xd)

        elif self.network_pattern == "6":
            xd = Dense(1024, activation="relu")(xd)
            xd = Dense(2048, activation="relu")(xd)

        else:
            raise ValueError(f"Invalid network_pattern: {self.network_pattern}")

        # ==================== Conv2DTranspose architecture branch ====================
        xd = Dense(self.max_filters * self.map_size, activation='relu')(xd)
        xd = Reshape((self.map_size, 1, self.max_filters))(xd)
        xd = BatchNormalization()(xd)

        if self.cnn_pattern == "0":
            # Baseline spec (3 layers)
            conv_params = [
                (self.max_filters // 2, (3, 1), (1, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.channel_dim,      (5, 1), (2, 1)),
            ]

        elif self.cnn_pattern == "1":
            # Modified kernel size and stride
            conv_params = [
                (self.max_filters // 2, (3, 1), (2, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.channel_dim,      (3, 1), (1, 1)),
            ]

        elif self.cnn_pattern == "2":
            # Additional Conv2DTranspose layer (4 layers)
            conv_params = [
                (self.max_filters // 2, (3, 1), (2, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.max_filters // 8, (3, 1), (1, 1)),
                (self.channel_dim,      (3, 1), (1, 1)),
            ]

        elif self.cnn_pattern == "3":
            # Fine-grained upsampling counterpart of CNN3
            conv_params = [
                (self.max_filters // 2, (3, 1), (1, 1)),
                (self.max_filters // 4, (3, 1), (2, 1)),
                (self.channel_dim,      (5, 1), (2, 1)),
            ]

        else:
            raise ValueError(f"Invalid cnn_pattern: {self.cnn_pattern}")

        # ==================== Conv2DTranspose block construction ====================
        for i, (f, k, s) in enumerate(conv_params[:-1], start=1):
            xd = Conv2DTranspose(f, k, strides=s, padding='same', name=f"dec_convT{i}")(xd)
            xd = BatchNormalization(name=f"dec_bn{i}")(xd)
            xd = Activation('relu', name=f"dec_act{i}")(xd)

        # Output layer (sigmoid activation, no batch norm)
        f, k, s = conv_params[-1]
        xd = Conv2DTranspose(f, k, strides=s, padding='same', name="dec_convT_last")(xd)
        xd = Activation('sigmoid', name="dec_output_act")(xd)

        decoder_outputs = Lambda(lambda t: tf.squeeze(t, axis=2), name="dec_squeeze")(xd)
        decoder = Model(latent_inputs, decoder_outputs,
                        name=f"decoder_cnn{self.cnn_pattern}_pat{self.network_pattern}")

>>>>>>> 2.15.X
        return decoder