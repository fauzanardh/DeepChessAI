from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import cpu_count

import chess.pgn

from config import Config
from util.data_helper import get_pgn_filenames
from util.dataset_tool import TFRecordExporter


def add_to_tfr(file, config: Config):
    dataset_name = str(file.name).split('.')[0]
    games = get_games_from_pgn(file)
    exporter = TFRecordExporter(f"{dataset_name}-{len(games)}", config)
    for game in games:
        exporter.add_data(game)
    exporter.close()
    return exporter.game_idx


def get_games_from_pgn(filename):
    pgn = open(filename, "r", errors='ignore')
    offsets = []
    while True:
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break
        offsets.append(offset)
    offsets = offsets
    n = len(offsets)
    print(f"Found {n} games!")
    games = []
    for offset in offsets:
        pgn.seek(offset)
        try:
            games.append(chess.pgn.read_game(pgn))
        except ValueError:
            continue
    return games


_config = Config("config-default.json")
files = get_pgn_filenames(_config)
total_games = 0
with ProcessPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(add_to_tfr, file, _config) for file in files]
    for future in as_completed(futures):
        total_games += future.result()
        print(f"Current total games: {total_games}")
print(f"Added {total_games} games!")
