from concurrent.futures import ProcessPoolExecutor

from tensorflow.keras.optimizers import Adam

from agent.model import ChessModel
from config import Config
from util.data_helper import get_tfr_filenames
from util.dataset_wrapper import DatasetWrapper


def start(config: Config):
    return Optimizer(config).start()


class Optimizer(object):
    def __init__(self, config: Config):
        self.config = config
        self.agent = None

    def load_model(self):
        agent = ChessModel(self.config)
        agent.load_latest()
        agent.model.summary()
        return agent

    def start(self):
        self.agent = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        self.train_epoch()
        self.agent.save()

    def get_epoch_size(self):
        files = get_tfr_filenames(self.config)
        epoch_size = 0
        for file in files:
            epoch_size += int(str(file).split('-')[1].split('.')[0])
        return epoch_size

    def train_epoch(self):
        ct = self.config.training
        epoch_size = self.get_epoch_size()
        print(f"Epoch Size: {epoch_size}")
        train_size = int(0.9 * epoch_size)
        dataset_wrapper = DatasetWrapper(self.config)
        self.agent.model.fit(
            dataset_wrapper.get_dataset(train_size),
            epochs=ct.epoch_to_checkpoint,
            validation_data=dataset_wrapper.get_dataset(train_size, is_training=False),
        )

    def compile_model(self):
        opt = Adam(learning_rate=3e-4)
        losses = ["categorical_crossentropy", "mean_squared_error"]
        self.agent.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.training.loss_weight)
