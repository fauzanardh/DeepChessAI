from pathlib import Path

import tensorflow as tf

from config import Config


def _float_feature(value):
    """
    Convert the float value to Tensorflow Feature

    :param value:
        The float value to convert
    :return:
        Converted float value
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class DatasetWrapper(object):
    """
    Wrapper for the dataset

    :ivar config:
        Config to use
    :ivar tfr_files:
        A list of TFRecord paths
    """

    def __init__(self, config: Config, tfr_files=None):
        self.config = config
        if tfr_files is None:
            files = [str(path) for path in list(Path(self.config.tfr_path).glob("*.tfrecords"))]
        else:
            files = [str(path) for path in tfr_files]
        self.tfr_files = files

    @staticmethod
    def parse_tfrecord(record):
        """
        Parse the record to be used in the Policy and Value network

        :param record:
            Record to be parsed
        :return:
            Parsed data
        """
        features = {
            # shape=chess board size
            "state": tf.io.FixedLenSequenceFeature([18, 8, 8], dtype=tf.float32, allow_missing=True, default_value=0.0),
            # shape=uci labels
            "policy": tf.io.FixedLenSequenceFeature([1968], dtype=tf.float32, allow_missing=True, default_value=0.0),
            "value": tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True, default_value=0.0)
        }
        sample = tf.io.parse_single_example(record, features)
        return sample["state"], sample["policy"], sample["value"]

    def filter_data(self, state, policy, value):
        """
        Filter the data to have more than `min_resign_turn` moves
        """
        return tf.convert_to_tensor(
            tf.py_function(
                self.filter_data_py,
                [state, policy, value, self.config.play.min_resign_turn],
                [bool]
            )
        )

    @staticmethod
    def filter_data_py(state, policy, value, min_resign_turn):
        """
        PyFunction for filtering data used in Tensorflow graph
        """
        return state.shape[0] >= min_resign_turn and \
               policy.shape[0] >= min_resign_turn and \
               value.shape[0] >= min_resign_turn

    @staticmethod
    def convert(state, policy, value):
        """
        Convert the data to match the Policy and Value network input
        """
        return state, {"policy_out": policy, "value_out": value}

    def get_dataset(self, train_size, is_training=True):
        """
        Make a TFRecordDataset from the current TFRecords

        :param train_size:
            Train total data
        :param is_training:
            Whether it is training or validating the data
        """
        dataset = tf.data.TFRecordDataset(
            self.tfr_files, compression_type="GZIP"
        )
        dataset = dataset.map(
            self.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if is_training:
            dataset = dataset.take(train_size)
            dataset = dataset.shuffle(
                train_size,  # train_size // 40 if train_size > 8192 else train_size,
                reshuffle_each_iteration=True
            )
            dataset = dataset.filter(self.filter_data)
            dataset = dataset.map(
                self.convert, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        else:
            dataset = dataset.skip(train_size)
            dataset = dataset.filter(self.filter_data)
            dataset = dataset.map(
                self.convert, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        dataset = dataset.unbatch()
        dataset = dataset.batch(self.config.training.batch_size)
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        return dataset
