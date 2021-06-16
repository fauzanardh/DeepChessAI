from tensorflow.keras.optimizers import Adam

from agent.model import ChessModel
from config import Config
from util.data_helper import get_tfr_filenames
from util.dataset_wrapper import DatasetWrapper


def start(config: Config):
    """
    Helper method to start the optimization

    :param config:
        Config to use
    """
    return Optimizer(config).start()


class Optimizer(object):
    """
    Class which optimize the model

    :ivar config:
        Config to use
    :ivar agent:
        The model to train
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.agent = None

    def load_model(self) -> ChessModel:
        """
        Load the latest weight

        :return:
            Model with the latest weight loaded
        """
        agent = ChessModel(self.config)
        agent.load_latest()
        agent.model.summary()
        return agent

    def start(self) -> None:
        """
        Starts the training
        """
        self.agent = self.load_model()
        self.training()

    def training(self) -> None:
        """
        Compile, train, and save the model.
        """
        self.compile_model()
        self.train_epoch()
        self.agent.save()

    def _get_total_games(self) -> int:
        """
        Get the total games in the TFRecords
        :return:
            The total games
        """
        files = get_tfr_filenames(self.config)
        total_games = 0
        for file in files:
            total_games += int(str(file).split('-')[1].split('.')[0])
        return total_games

    def train_epoch(self) -> None:
        """
        Train the model for x epochs,
        where the x is the number configured in the config
        """
        ct = self.config.training
        total_games = self._get_total_games()
        print(f"Total Games: {total_games}")
        train_size = int(0.9 * total_games)
        dataset_wrapper = DatasetWrapper(self.config)
        self.agent.model.fit(
            dataset_wrapper.get_dataset(train_size),
            epochs=ct.epoch_to_checkpoint,
            validation_data=dataset_wrapper.get_dataset(train_size, is_training=False),
        )

    def compile_model(self) -> None:
        """
        Compile the Model
        """
        opt = Adam(learning_rate=3e-3)
        losses = ["categorical_crossentropy", "mean_squared_error"]
        self.agent.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.training.loss_weight)
