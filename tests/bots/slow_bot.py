import time


class Policy:
    """Burns 80 ms of CPU on tick 5, otherwise a no-op."""

    def __init__(self, seed=0):
        pass

    def act(self, obs):
        if obs["tick"] == 5:
            end = time.process_time() + 0.08
            while time.process_time() < end:
                pass
        return [
            {"move": (0, 0), "aim": a["facing"], "fire": False, "swim": False} for a in obs["team"]
        ]
