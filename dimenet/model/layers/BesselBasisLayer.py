import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .envelope import Envelope


class BesselBasisLayer(layers.Layer):
    def __init__(self, num_radial, cutoff, envelope_exponent=5,
                 name='bessel_basis', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_radial = num_radial
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope = Envelope(envelope_exponent)

    def build(self, input_shape):
        # Initialize centers
        frequencies = np.pi * np.arange(1, self.num_radial + 1)
        self.frequencies = tf.Variable(
            frequencies, name="frequencies", dtype=tf.float32)

    def call(self, inputs):
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = tf.expand_dims(d_scaled, -1)

        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * tf.sin(self.frequencies * d_scaled)
