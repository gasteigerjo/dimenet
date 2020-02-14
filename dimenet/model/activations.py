import tensorflow as tf


def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x*tf.sigmoid(x)


def shifted_softplus(x):
    return tf.nn.softplus(x) - tf.log(2.0)
