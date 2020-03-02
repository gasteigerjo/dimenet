import tensorflow as tf
import threading
from collections import OrderedDict
from .DataContainer import index_keys


class DataQueue:
    def __init__(self, get_batch, ntargets=1, capacity=5000, dtype=tf.float32):
        self.get_batch = get_batch
        self._is_running = False

        self.dtypes_dict = OrderedDict()
        self.dtypes_dict['id'] = tf.int32
        self.dtypes_dict['N'] = tf.int32
        self.dtypes_dict['Z'] = tf.int32
        self.dtypes_dict['R'] = tf.float32
        self.dtypes_dict['targets'] = tf.float32
        for key in index_keys:
            self.dtypes_dict[key] = tf.int32

        shapes_dict = OrderedDict()
        shapes_dict['id'] = [None]
        shapes_dict['N'] = [None]
        shapes_dict['Z'] = [None]
        shapes_dict['R'] = [None, 3]
        shapes_dict['targets'] = [None, ntargets]
        for key in index_keys:
            shapes_dict[key] = [None]

        self._queue = tf.queue.PaddingFIFOQueue(capacity=capacity,
                                                dtypes=list(self.dtypes_dict.values()),
                                                shapes=list(shapes_dict.values()),
                                                names=list(self.dtypes_dict.keys()))

    def create_thread(self, coord=None, daemon=False):
        if coord is None:
            coord = tf.train.Coordinator()

        if self._is_running:
            return []

        thread = threading.Thread(target=self._run, args=(coord,))
        thread.daemon = daemon
        thread.start()
        self._is_running = True
        return [thread]

    def _run(self, coord):
        while not coord.should_stop():
            batch = self.get_batch()
            for key, val in batch.items():
                batch[key] = tf.constant(val, dtype=self.dtypes_dict[key])
            try:
                self.enqueue(batch)
            except Exception as e:
                coord.request_stop(e)

    def enqueue(self, vals):
        return self._queue.enqueue(vals)

    def dequeue(self):
        return self._queue.dequeue()
