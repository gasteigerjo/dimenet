import tensorflow as tf
from .schedules import LinearWarmupExponentialDecay


class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)

        self.ema = tf.train.ExponentialMovingAverage(self.ema_decay)

        # Make backup variables
        self.backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                            initializer=var.initialized_value())
                            for var in self.model.trainable_weights]

    def update_weights(self, loss, gradient_tape):
        grads = gradient_tape.gradient(loss, self.model.trainable_weights)

        global_norm = tf.linalg.global_norm(grads)
        tf.summary.scalar("global_gradient_norm", global_norm)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.ema.apply(self.model.trainable_weights)

    def load_averaged_variables(self):
        for var in self.model.trainable_weights:
            var.assign(self.ema.average(var))

    def save_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)
