import tensorflow as tf
from tensorflow.keras import layers

from .residual_layer import ResidualLayer
from ..initializers import GlorotOrthogonal


class InteractionPPBlock(layers.Layer):
    def __init__(self, emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip,
                 activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        weight_init = GlorotOrthogonal()

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = layers.Dense(basis_emb_size, use_bias=False, kernel_initializer=weight_init)
        self.dense_rbf2 = layers.Dense(emb_size, use_bias=False, kernel_initializer=weight_init)
        self.dense_sbf1 = layers.Dense(basis_emb_size, use_bias=False, kernel_initializer=weight_init)
        self.dense_sbf2 = layers.Dense(int_emb_size, use_bias=False, kernel_initializer=weight_init)

        # Dense transformations of input messages
        self.dense_ji = layers.Dense(emb_size, activation=activation, use_bias=True,
                                     kernel_initializer=weight_init)
        self.dense_kj = layers.Dense(emb_size, activation=activation, use_bias=True,
                                     kernel_initializer=weight_init)

        # Embedding projections for interaction triplets
        self.down_projection = layers.Dense(int_emb_size, activation=activation, use_bias=False,
                                            kernel_initializer=weight_init)
        self.up_projection = layers.Dense(emb_size, activation=activation, use_bias=False,
                                          kernel_initializer=weight_init)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=weight_init))
        self.final_before_skip = layers.Dense(emb_size, activation=activation, use_bias=True,
                                              kernel_initializer=weight_init)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=weight_init))

    def call(self, inputs):
        x, rbf, sbf, id_expand_kj, id_reduce_ji = inputs
        num_interactions = tf.shape(x)[0]

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis
        rbf = self.dense_rbf1(rbf)
        rbf = self.dense_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj)
        x_kj = tf.gather(x_kj, id_expand_kj)

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf)
        sbf = self.dense_sbf2(sbf)
        x_kj = x_kj * sbf

        # Aggregate interactions and up-project embeddings
        x_kj = tf.math.unsorted_segment_sum(x_kj, id_reduce_ji, num_interactions)
        x_kj = self.up_projection(x_kj)

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
