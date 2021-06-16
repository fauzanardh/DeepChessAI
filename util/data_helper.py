import json
from pathlib import Path
from typing import List

from config import Config


def get_pgn_filenames(config: Config) -> List[Path]:
    """
    Get the PGNs filenames

    :param config:
        Config to use
    :return:
        Sorted PGNs path
    """
    path = Path(config.pgn_path)
    return sorted(path.glob("*.pgn"))


def get_tfr_filenames(config: Config) -> List[Path]:
    """
    Get the TFRs filenames

    :param config:
        Config to use
    :return:
        Sorted TFRs path
    """
    path = Path(config.tfr_path)
    return sorted(path.glob("*.tfrecords"))
