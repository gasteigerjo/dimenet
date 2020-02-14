import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class BesselBasisLayer(layers.Layer):
    def __init__(self, num_radial, cutoff, envelope_exponent=5,
                 name='bessel_basis', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_radial = num_radial
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope_exponent = envelope_exponent

    def build(self, input_shape):
        # Initialize centers
        frequencies = np.pi * np.arange(1, self.num_radial + 1)
        self.frequencies = tf.Variable(
            frequencies, name="frequencies", dtype=tf.float32)

    def get_envelope(self, p):
        """
        Create formula for envelope function
        """
        p += 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2

        def envelope(r):
            """Cutoff function divided by r"""
            return 1/r + a * r**(p - 1) + b * r**p + c * r**(p + 1)
        return envelope

    def cutoff(self, x):
        """
        Envelope function that ensures a smooth cutoff
        """
        return tf.where(x < 1, self.get_envelope(self.envelope_exponent)(x), tf.zeros_like(x))

    def call(self, inputs):
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = tf.expand_dims(d_scaled, -1)

        d_cutoff = self.cutoff(d_scaled)
        return d_cutoff * tf.sin(self.frequencies * d_scaled)
