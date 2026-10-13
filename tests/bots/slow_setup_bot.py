import time


class Policy:
    def __init__(self, seed=0):
        end = time.process_time() + 1.5
        while time.process_time() < end:
            pass

    def act(self, obs):
        return []
