import time


class Policy:
    """Uses 20 ms CPU every tick: under the 50 ms hard limit, over the 5 ms average."""

    def __init__(self, seed=0):
        pass

    def act(self, obs):
        end = time.process_time() + 0.02
        while time.process_time() < end:
            pass
        return [
            {"move": (0, 0), "aim": a["facing"], "fire": False, "swim": False} for a in obs["team"]
        ]
