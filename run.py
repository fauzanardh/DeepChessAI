import sys
import multiprocessing as mp

from config import Config
from ai.self_play import start as SelfPlayStart


def start():
    conf = Config("config-default.json")
    SelfPlayStart(conf)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.setrecursionlimit(10000)
    start()
