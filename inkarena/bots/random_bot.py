"""Uniformly random actions, for sanity checks and stress tests."""

import math
import random


class Policy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs):
        r = self.rng
        out = []
        for _ in obs["team"]:
            angle = r.uniform(0, 2 * math.pi)
            out.append(
                {
                    "move": (math.cos(angle), math.sin(angle)),
                    "aim": r.uniform(0, 2 * math.pi),
                    "fire": r.random() < 0.5,
                    "swim": r.random() < 0.3,
                }
            )
        return out
