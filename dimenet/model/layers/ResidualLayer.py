from tensorflow.keras import layers


class ResidualLayer(layers.Layer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None,
                 name='residual', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_1 = layers.Dense(units, activation=activation, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer)
        self.dense_2 = layers.Dense(units, activation=activation, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer)

    def call(self, inputs):
        x = inputs + self.dense_2(self.dense_1(inputs))
        return x
