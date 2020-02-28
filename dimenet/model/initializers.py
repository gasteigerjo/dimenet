import tensorflow as tf


class GlorotOrthogonal(tf.initializers.Initializer):
    """
    Generate a weight matrix with variance according to Glorot initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """

    def __init__(self, scale=2.0, seed=None):
        super().__init__()
        self.orth_init = tf.initializers.Orthogonal(seed=seed)
        self.scale = scale

    def __call__(self, shape, dtype=tf.float32):
        assert len(shape) == 2
        W = self.orth_init(shape, dtype)
        W *= tf.sqrt(self.scale / ((shape[0] + shape[1]) * tf.math.reduce_variance(W)))
        return W
