from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np


class ChessModelAPI(object):
    def __init__(self, agent):
        self.agent = agent
        self.pipes = []

    def start(self):
        worker = Thread(target=self._predict_batch_worker, name="prediction_worker")
        worker.daemon = True
        worker.start()

    def create_pipe(self):
        me, you = Pipe()
        self.pipes.append(me)
        return you

    def _predict_batch_worker(self):
        while True:
            ready = connection.wait(self.pipes, timeout=0.1)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            policy_arr, value_arr = self.agent.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_arr, value_arr):
                pipe.send((p, float(v)))
