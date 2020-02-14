import tensorflow as tf
from tensorflow.keras import layers

from ..initializers import GlorotOrthogonal


class OutputBlock(layers.Layer):
    def __init__(self, num_features, num_dense, num_targets=12,
                 activation=None, seed=None, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        weight_init = GlorotOrthogonal(seed=seed)

        self.dense_rbf = layers.Dense(num_features, activation=activation, use_bias=False,
                                      kernel_initializer=weight_init)
        self.dense_layers = []
        for i in range(num_dense):
            self.dense_layers.append(
                layers.Dense(num_features, activation=activation, use_bias=True,
                             kernel_initializer=weight_init))
        self.dense_final = layers.Dense(num_targets, activation=activation, use_bias=False,
                                        kernel_initializer='zeros')

    def call(self, inputs):
        x, rbf, idnb_i, n_atoms = inputs

        g = self.dense_rbf(rbf)
        x = g * x
        x = tf.math.unsorted_segment_sum(x, idnb_i, n_atoms)

        for layer in self.dense_layers:
            x = layer(x)
        x = self.dense_final(x)
        return x
