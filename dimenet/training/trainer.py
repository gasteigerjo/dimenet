import tensorflow as tf
import tensorflow_addons as tfa
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

        opt = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)

        # Initialize backup variables
        if model.built:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            self.backup_vars = None

    def update_weights(self, loss, gradient_tape):
        grads = gradient_tape.gradient(loss, self.model.trainable_weights)

        global_norm = tf.linalg.global_norm(grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def load_averaged_variables(self):
        self.optimizer.assign_average_vars(self.model.trainable_weights)

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
            mean_mae = tf.reduce_mean(mae)
            loss = mean_mae
        self.update_weights(loss, tape)

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def predict_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)

        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae
        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return preds
