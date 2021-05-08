import json
from pathlib import Path

from config import Config


def get_game_data_filenames(config: Config):
    path = Path(config.play_path)
    return list(sorted(path.glob("*.json")))


def get_pgn_filenames(config: Config):
    path = Path(config.pgn_path)
    return list(sorted(path.glob("*.pgn")))


def write_data(path, data):
    try:
        with open(path, "w+") as fp:
            fp.write(json.dumps(data))
    except Exception as e:
        print(e)


def load_data(path):
    try:
        with open(path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        print(e)
