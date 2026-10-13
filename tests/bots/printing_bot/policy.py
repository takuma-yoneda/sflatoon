from helper_module import ACTION  # a module next to policy.py


class Policy:
    def __init__(self, seed=0):
        print("hello from setup")

    def act(self, obs):
        print("lots of output " * 1000)
        return [dict(ACTION, aim=a["facing"]) for a in obs["team"]]
