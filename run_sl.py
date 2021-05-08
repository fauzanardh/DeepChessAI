import sys
import multiprocessing as mp

from config import Config
from ai.sl import start as SupervisedLearningStart


def start():
    conf = Config("config-default.json")
    SupervisedLearningStart(conf)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.setrecursionlimit(10000)
    start()
