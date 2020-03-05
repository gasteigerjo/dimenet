import numpy as np
import tensorflow as tf


class Metrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex

        self.loss_metric = tf.keras.metrics.Mean()
        self.mean_mae_metric = tf.keras.metrics.Mean()
        self.mae_metric = tf.keras.metrics.MeanTensor()
        self.mae_metric.update_state([0] * len(targets), sample_weight=[0] * len(targets))

    def update_state(self, loss, mean_mae, mae, nsamples):
        self.loss_metric.update_state(loss, sample_weight=nsamples)
        self.mean_mae_metric.update_state(mean_mae, sample_weight=nsamples)
        self.mae_metric.update_state(mae, sample_weight=nsamples)

    def write(self):
        """Write metrics to tf.summary and the Sacred experiment."""
        for key, val in self.result().items():
            tf.summary.scalar(key, val)
            if self.ex is not None:
                if key not in self.ex.current_run.info:
                    self.ex.current_run.info[key] = []
                self.ex.current_run.info[key].append(val)

        if self.ex is not None:
            if f'step_{self.tag}' not in self.ex.current_run.info:
                self.ex.current_run.info[f'step_{self.tag}'] = []
            self.ex.current_run.info[f'step_{self.tag}'].append(tf.summary.experimental.get_step())

    def reset_states(self):
        self.loss_metric.reset_states()
        self.mean_mae_metric.reset_states()
        self.mae_metric.reset_states()

    def result(self):
        result_dict = {}
        result_dict[f'loss_{self.tag}'] = self.loss
        result_dict[f'mean_mae_{self.tag}'] = self.loss
        result_dict[f'mean_log_mae_{self.tag}'] = self.loss
        for i, key in enumerate(self.targets):
            result_dict[key + '_' + self.tag] = self.mae[i]
        return result_dict

    @property
    def loss(self):
        return self.loss_metric.result()

    @property
    def mae(self):
        return self.mae_metric.result()

    @property
    def mean_mae(self):
        return self.mean_mae_metric.result()

    @property
    def mean_log_mae(self):
        return np.mean(np.log(self.mae_metric.result()))
