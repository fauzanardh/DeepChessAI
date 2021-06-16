import shutil
from multiprocessing import Pipe
from pathlib import Path
from typing import List

from tensorflow.keras import Input, Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, Add, BatchNormalization, Layer
from tensorflow.keras.regularizers import l2

from agent.api import ChessModelAPI
from config import Config


class ChessModel(object):
    """
    This class holds the keras model used for the chess AI

    :ivar config:
        Config object used for creating and saving the keras model
    :ivar model:
        Fully build keras model
    :ivar api:
        Handle to the API for running the inference
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = self.build()
        self.api = None

    def reset(self) -> None:
        """
        Reset the model entirely,
        closing the pipes and clearing the tensorflow session
        """
        if self.api is not None:
            for pipe in self.api.pipes:
                pipe.close()
            del self.api
            self.api = None
        clear_session()

    def get_pipes(self, num=1) -> List[Pipe]:
        """
        Creates a list of pipes on which
        the game state observation will be listened.

        :param num:
            The number of pipes that will be created

        :return:
            A list of newly created pipes
        """
        if self.api is None:
            self.api = ChessModelAPI(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]

    def build(self) -> Model:
        """
        Builds the full keras model

        :return:
            full keras model
        """
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
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(_mc.l2_reg), name="value_conv-1-1")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(_mc.value_fc_size, kernel_regularizer=l2(_mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(_mc.l2_reg), activation="tanh", name="value_out")(x)

        return Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x: Layer, i: int) -> Layer:
        """
        Create a residual network on the model

        :param x:
            The layer that will be used on
        :param i:
            The index of the residual network
        :return:
            Layer with a new residual network on top
        """
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

    def get_weights_path(self) -> List[Path]:
        """
        Get all the saved weights and sort it

        :return:
            Sorted weights path
        """
        path = Path(self.config.model_path)
        weights_path = list(sorted(path.glob("model_*.h5")))
        return weights_path

    def load_path(self, path: Path) -> None:
        """
        Load the model weight from a given path

        :param path:
            Path of the model
        :return:
            Model with the weights loaded from the path
        """
        assert path.exists(), "Invalid model!"
        print(f"Loading weights from {path}")
        self.model.load_weights(str(path))

    def load_best(self) -> None:
        """
        Load the best model weight

        :return:
            Model with the weights loaded from the path
        """
        path = Path(self.config.model_path) / "best_model"
        best_model_path = path / "model.h5"
        if path.exists():
            if not best_model_path.exists():
                all_weights = list(Path(self.config.model_path).glob("model_*.h5"))
                if len(all_weights) == 0:
                    print("No weight(s) found, creating a new model.")
                    self.save()
                all_weights = list(Path(self.config.model_path).glob("model_*.h5"))
                latest_weight = str(max(all_weights, key=lambda x: int(str(x).split("_")[1].split(".")[0])))
                shutil.copy(latest_weight, best_model_path)
        else:
            path.mkdir()
            all_weights = list(Path(self.config.model_path).glob("model_*.h5"))
            if len(all_weights) == 0:
                print("No weight(s) found, creating a new model.")
                self.save()
            all_weights = list(Path(self.config.model_path).glob("model_*.h5"))
            latest_weight = str(max(all_weights, key=lambda x: int(str(x).split("_")[1].split(".")[0])))
            shutil.copy(latest_weight, best_model_path)
        self.model.load_weights(str(best_model_path))

    def load_n(self, n: int) -> None:
        """
            Load the model weight from the selected index
        :param n:
            The model index
        :return:
            Model with the weights loaded from the path
        """
        path = Path(self.config.model_path) / f"model_{n:05}.h5"
        assert path.exists(), "Invalid model!"
        print(f"Loading weights from {path}")
        self.model.load_weights(str(path))

    def load_latest(self) -> None:
        """
        Load the latest model weight

        :return:
            Model with the weights loaded from the path
        """
        path = Path(self.config.model_path)
        if path.exists():
            weights_path = list(path.glob("model_*.h5"))
            if len(weights_path) != 0:
                latest_weight = str(max(weights_path, key=lambda x: int(str(x).split("_")[1].split(".")[0])))
                print(f"Loading weights from {latest_weight}")
                self.model.load_weights(latest_weight)
            else:
                print("No weight(s) found, creating a new model.")
                self.save()
        else:
            print("New run detected.")
            path.mkdir()
            self.save()

    # Save the latest model weight
    def save(self) -> None:
        """
        Save the latest model weight
        """
        path = Path(self.config.model_path)
        weights_path = list(path.glob("model_*.h5"))
        if len(weights_path) != 0:
            latest_weight = str(max(list(weights_path), key=lambda x: int(str(x).split("_")[1].split(".")[0])))
            latest_num = int(str(latest_weight).split("_")[1].split(".")[0]) + 1
        else:
            latest_num = 1

        print(f"Saving the model to {path}")
        self.model.save_weights(path / f"model_{latest_num:05}.h5")
