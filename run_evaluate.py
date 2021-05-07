import sys
import multiprocessing as mp

from config import Config
from ai.evaluate import start as EvaluateStart


def start():
    conf = Config("config-default.json")
    EvaluateStart(conf)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.setrecursionlimit(10000)
    start()
