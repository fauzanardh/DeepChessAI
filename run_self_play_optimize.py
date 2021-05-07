import sys
import multiprocessing as mp

from config import Config
from ai.self_play_optimize import start as StartBoth


def start():
    conf = Config("config-default.json")
    StartBoth(conf)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.setrecursionlimit(10000)
    start()
