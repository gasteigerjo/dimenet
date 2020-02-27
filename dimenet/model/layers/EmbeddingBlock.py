import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..initializers import GlorotOrthogonal


class EmbeddingBlock(layers.Layer):
    def __init__(self, num_features, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.weight_init = GlorotOrthogonal()
        self.emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))

        self.dense_rbf = layers.Dense(self.num_features, activation=activation, use_bias=True,
                                      kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.num_features, activation=activation, use_bias=True,
                                  kernel_initializer=self.weight_init)

    def build(self, input_shape):
        # Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        self.embeddings = tf.Variable(self.emb_init((95, self.num_features)), name="embeddings")

    def call(self, inputs):
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)
        x = tf.gather(self.embeddings, Z)

        x1 = tf.gather(x, idnb_i)
        x2 = tf.gather(x, idnb_j)

        x = tf.concat([x1, x2, rbf], axis=-1)
        x = self.dense(x)
        return x
