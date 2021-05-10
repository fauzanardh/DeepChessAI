from pathlib import Path

import numpy as np
import tensorflow as tf


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class DatasetWrapper(object):
    def __init__(self, tfr_folder):
        self.tfr_files = [list(Path(tfr_folder).glob("*.tfrecords"))]

    @staticmethod
    def parse_tfrecord(record):
        features = {
            # shape=chess board size
            "state": tf.io.FixedLenSequenceFeature([18, 8, 8], dtype=tf.float32, allow_missing=True, default_value=0.0),
            # shape=uci labels
            "policy": tf.io.FixedLenSequenceFeature([1968], dtype=tf.float32, allow_missing=True, default_value=0.0),
            "value": tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True, default_value=0.0)
        }
        sample = tf.io.parse_single_example(record, features)
        return sample["state"], sample["policy"], sample["value"]

    def get_dataset(self, batch_size, train_size, is_training=True):
        dataset = tf.data.TFRecordDataset(
            self.tfr_files, compression_type="GZIP"
        )
        dataset = dataset.map(
            self.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if is_training:
            dataset.repeat()
            dataset = dataset.take(train_size)
            dataset = dataset.shuffle(2048)
        else:
            dataset = dataset.skip(train_size)
        dataset.batch(batch_size)
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        return dataset
