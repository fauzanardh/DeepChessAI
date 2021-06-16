from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np


class ChessModelAPI(object):
    """
    Class used to make an API between processes

    :ivar agent:
        Chess model that will be used for inference
    :ivar pipes:
        List of pipes for communicating between processes
    """
    def __init__(self, agent) -> None:
        self.agent = agent
        self.pipes = []

    def start(self) -> None:
        """
        Starts the prediction worker thread
        """
        worker = Thread(target=self._predict_batch_worker, name="prediction_worker")
        worker.daemon = True
        worker.start()

    def create_pipe(self) -> Pipe:
        """
        Create new pipes for communicating between processes

        :return:
            returns a pipe
        """
        me, you = Pipe()
        self.pipes.append(me)
        return you

    def _predict_batch_worker(self) -> None:
        """
        Used for gathering data and from the pipes and
        run inference from the loaded model
        """
        while True:
            # Does polling until one or more pipes have data
            try:
                ready = connection.wait(self.pipes, timeout=0.1)
            except OSError:
                return
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                try:
                    while pipe.poll():
                        data.append(pipe.recv())
                        result_pipes.append(pipe)
                except BrokenPipeError:
                    print("Pipe closed!")
                    return

            # Cast the data to numpy array with float32 data type
            data = np.asarray(data, dtype=np.float32)
            # Starts prediction on the policy and value network
            policy_arr, value_arr = self.agent.model.predict_on_batch(data)
            # Send back the data to the other process that gave it
            for pipe, p, v in zip(result_pipes, policy_arr, value_arr):
                pipe.send((p, float(v)))
