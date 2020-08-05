import tensorflow as tf
from tensorflow.keras import layers

from ..initializers import GlorotOrthogonal


class OutputBlock(layers.Layer):
    def __init__(self, emb_size, num_dense, num_targets=12,
                 activation=None, output_init='zeros', name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        weight_init = GlorotOrthogonal()
        if output_init == 'GlorotOrthogonal':
            output_init = GlorotOrthogonal()

        self.dense_rbf = layers.Dense(emb_size, use_bias=False,
                                      kernel_initializer=weight_init)
        self.dense_layers = []
        for i in range(num_dense):
            self.dense_layers.append(
                layers.Dense(emb_size, activation=activation, use_bias=True,
                             kernel_initializer=weight_init))
        self.dense_final = layers.Dense(num_targets, use_bias=False,
                                        kernel_initializer=output_init)

    def call(self, inputs):
        x, rbf, idnb_i, n_atoms = inputs

        g = self.dense_rbf(rbf)
        x = g * x
        x = tf.math.unsorted_segment_sum(x, idnb_i, n_atoms)

        for layer in self.dense_layers:
            x = layer(x)
        x = self.dense_final(x)
        return x
