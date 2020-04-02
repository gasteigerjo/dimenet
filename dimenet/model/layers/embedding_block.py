import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..initializers import GlorotOrthogonal


class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_init = GlorotOrthogonal()

        # Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        self.embeddings = self.add_weight(name="embeddings", shape=(95, self.emb_size),
                                          dtype=tf.float32, initializer=emb_init, trainable=True)

        self.dense_rbf = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                      kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                  kernel_initializer=self.weight_init)

    def call(self, inputs):
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)

        Z_i = tf.gather(Z, idnb_i)
        Z_j = tf.gather(Z, idnb_j)

        x_i = tf.gather(self.embeddings, Z_i)
        x_j = tf.gather(self.embeddings, Z_j)

        x = tf.concat([x_i, x_j, rbf], axis=-1)
        x = self.dense(x)
        return x
