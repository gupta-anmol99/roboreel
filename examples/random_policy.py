"""The smallest possible policy. Copy this and replace act() with your algorithm."""

from benchmark import Policy


class RandomPolicy(Policy):
    def reset(self, episode_id, env, human_demo):
        self.action_space = env.action_space

    def act(self, obs):
        return self.action_space.sample()
