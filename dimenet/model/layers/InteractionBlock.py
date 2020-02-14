import tensorflow as tf
from tensorflow.keras import layers

from .ResidualLayer import ResidualLayer
from ..initializers import GlorotOrthogonal


class InteractionBlock(layers.Layer):
    def __init__(self, num_features, num_bilinear, num_before_skip, num_after_skip,
                 activation=None, seed=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.num_bilinear = num_bilinear
        weight_init = GlorotOrthogonal(seed=seed)
        self.bilin_initializer = tf.initializers.RandomNormal(
            mean=0.0, stddev=2/num_features, seed=seed)

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = layers.Dense(num_features, use_bias=False, kernel_initializer=weight_init)
        self.dense_sbf = layers.Dense(num_bilinear, use_bias=False, kernel_initializer=weight_init)

        # Dense transformations of input messages
        self.dense_ji = layers.Dense(num_features, activation=activation, use_bias=True,
                                     kernel_initializer=weight_init)
        self.dense_kj = layers.Dense(num_features, activation=activation, use_bias=True,
                                     kernel_initializer=weight_init)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(num_features, activation=activation, use_bias=True,
                              kernel_initializer=weight_init))
        self.final_before_skip = layers.Dense(num_features, activation=activation, use_bias=True,
                                              kernel_initializer=weight_init)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(num_features, activation=activation, use_bias=True,
                              kernel_initializer=weight_init))

    def build(self, input_shape):
        self.W_bilin = tf.Variable(self.bilin_initializer(
            (self.num_features, self.num_features, self.num_bilinear)))

    def call(self, inputs):
        x, rbf, sbf, id_expand_kj, id_reduce_ji = inputs
        num_interactions = tf.shape(x)[0]

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis
        g = self.dense_rbf(rbf)
        x_kj = x_kj * g

        # Transform via spherical basis
        sbf = self.dense_sbf(sbf)
        x_kj = tf.gather(x_kj, id_expand_kj)
        # Apply bilinear layer to interactions and basis function activation
        x_kj = tf.einsum("wi,wl,ijl->wj", x_kj, sbf, self.W_bilin)
        x_kj = tf.math.unsorted_segment_sum(
            x_kj, id_reduce_ji, num_interactions)  # sum over messages

        # Transformations before skip connection
        x2 = x_ji + x_kj
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)

        # Skip connection
        x = x + x2

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)
        return x
