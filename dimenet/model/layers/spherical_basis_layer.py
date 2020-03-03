import sympy as sym
import tensorflow as tf
from tensorflow.keras import layers

from .basis_utils import bessel_basis, real_sph_harm
from .envelope import Envelope


class SphericalBasisLayer(layers.Layer):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5,
                 name='spherical_basis', **kwargs):
        super().__init__(name=name, **kwargs)

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical

        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to tensorflow functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow')(0)
                self.sph_funcs.append(lambda tensor: tf.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow'))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], 'tensorflow'))

    def call(self, inputs):
        d, Angles, id_expand_kj = inputs

        d_scaled = d * self.inv_cutoff
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = tf.stack(rbf, axis=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf
        rbf_env = tf.gather(rbf_env, id_expand_kj)

        cbf = [f(Angles) for f in self.sph_funcs]
        cbf = tf.stack(cbf, axis=1)
        cbf = tf.repeat(cbf, self.num_radial, axis=1)

        return rbf_env * cbf
