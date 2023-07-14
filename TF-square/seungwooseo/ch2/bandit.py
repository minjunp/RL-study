import dm_env
import matplotlib.pyplot as plt
from dm_env import TimeStep, specs
import numpy as np


class Bandit(dm_env.Environment):
    """A Bandit environment built on the 'dm_env.Environment' class.

    TODO
    <desciption for Bandit>
    """
    def __init__(self, n_arms: int = 10, seed: int = 2022):
        """
        Initializes a new Bandit environment.

        :param n_arms: number of arms of bandit
        :param seed: random seed for RNG
        """
        # Set arms and q-values
        np.random.seed(seed)
        self._means = np.random.normal(0, 1, n_arms)    # selected from Sutton & Barto book p.29
        self._reset_next_step = True

    def step(self, action: int) -> TimeStep:

        assert action <= len(self._means)-1
        assert action >= 0

        # take action and get the reward
        reward = np.random.normal(self._means[action], 1)

        # terminate
        return dm_env.termination(reward=reward, observation=0)

    def reset(self) -> TimeStep:
        """Return the first 'TimeStep' of a new episode.
        reset() is abstract method of dm_env.Environment but will not be used
        In Bandit, first 'TimeStep' without reward is not mandatory."""
        return dm_env.restart(observation=0)

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=len(self._means), name="action")

    def observation_spec(self):
        """Returns the observation spec.
        Bandit has no states
        """
        return None


if __name__ == '__main__':
    bandit = Bandit(n_arms=10, seed=2022)
    record = dict()
    for i in range(10):
        record.update({f'{i}': list()})

    for i in range(20000):
        action = np.random.randint(0, 10)
        # Bandit has no states and reward is given immediately.
        # Step will be reset
        first_step = bandit.step(action)
        record[f'{action}'].append(first_step.reward)

    final_record = list()
    for i in range(10):
        final_record.append(record[f'{i}'])
    plt.violinplot(dataset=final_record)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.show()
