from pathlib import Path

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, Add, BatchNormalization
from tensorflow.keras.regularizers import l2

from config import Config


class ChessModel(object):
    def __init__(self, config: Config):
        self.config = config
        self.model = self.build()

    def build(self):
        _mc = self.config.model

        # Network Input
        in_x = x = Input((18, 8, 8))
        x = Conv2D(filters=_mc.cnn_filter_num, kernel_size=_mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(_mc.l2_reg),
                   name=f"input_conv-{_mc.cnn_filter_size}-{_mc.cnn_filter_num}")(x)
        x = BatchNormalization(axis=1, name=f"input_batchnorm")(x)
        x = Activation("relu", name=f"input_relu")(x)

        # ResNet
        for _i in range(_mc.res_layer_num):
            x = self._build_residual_block(x, _i + 1)
        res_out = x

        # Policy Network output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(_mc.l2_reg), name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(_mc.l2_reg), activation="softmax",
                           name="policy_out")(x)

        # Value Network output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(_mc.l2_reg), name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(_mc.value_fc_size, kernel_regularizer=l2(_mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(_mc.l2_reg), activation="tanh", name="value_out")(x)

        return Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, i):
        _mc = self.config.model
        in_x = x
        res_name = f"residual_{i}"
        x = Conv2D(filters=_mc.cnn_filter_num, kernel_size=_mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(_mc.l2_reg),
                   name=f"{res_name}_conv1-{_mc.cnn_filter_size}-{_mc.cnn_filter_num}")(x)
        x = BatchNormalization(axis=1, name=f"{res_name}_batchnorm1")(x)
        x = Activation("relu", name=f"{res_name}_relu1")(x)
        x = Conv2D(filters=_mc.cnn_filter_num, kernel_size=_mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(_mc.l2_reg),
                   name=f"{res_name}_conv2-{_mc.cnn_filter_size}-{_mc.cnn_filter_num}")(x)
        x = BatchNormalization(axis=1, name=f"{res_name}_batchnorm2")(x)
        x = Add(name=f"{res_name}_add")([in_x, x])
        x = Activation("relu", name=f"{res_name}_relu2")(x)
        return x

    def load(self, config: Config, n: int):
        path = Path(config.model_path) / f"model_{n:05}.h5"
        assert path.exists(), "Invalid model!"
        print(f"Loading weights from {path}")
        self.model.load_weights(str(path))

    def load_latest(self, config: Config):
        path = Path(config.model_path)
        if path.exists():
            weights_path = list(path.glob("model_*.h5"))
            if len(weights_path) != 0:
                latest_weight = str(max(list(weights_path), key=lambda x: int(str(x).split("_")[1].split(".")[0])))
                print(f"Loading weights from {latest_weight}")
                self.model.load_weights(latest_weight)
            else:
                print("No weight(s) found.")
        else:
            print("New run detected.")
            path.mkdir()

    def save(self, config: Config):
        path = Path(config.model_path)
        weights_path = list(path.glob("model_*.h5"))
        if len(weights_path) != 0:
            latest_weight = str(max(list(weights_path), key=lambda x: int(str(x).split("_")[1].split(".")[0])))
            latest_num = int(str(latest_weight).split("_")[1].split(".")[0]) + 1
        else:
            latest_num = 1

        print(f"Saving the model to {path}")
        self.model.save_weights(path / f"model_{latest_num:05}.h5")
