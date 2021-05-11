import sys
import multiprocessing as mp

from config import Config
from ai.do_all import start as DoAll


def start():
    conf = Config("config-default.json")
    DoAll(conf)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.setrecursionlimit(10000)
    start()
