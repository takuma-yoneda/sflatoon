"""A bot that stands still and looks where it already looks."""


class Policy:
    def __init__(self, seed: int = 0):
        pass

    def act(self, obs):
        return [
            {"move": (0.0, 0.0), "aim": a["facing"], "fire": False, "swim": False}
            for a in obs["team"]
        ]
