#!/usr/bin/env python3

# A general purpose EXP3 implementation

import dataclasses
import random
import math
from typing import List

@dataclasses.dataclass
class EXP3:
    weights: List[float] = None
    gamma: float = 0
    no_actions: int = 0
    round: int = 0
    r: random.Random = random
    _waiting_reward: bool = False
    _history: dict = None

    def __init__(self, no_actions: int, gamma: float, r = random, max_weight: float = 10_000, enable_history: bool = True) -> None:
        assert 0 < gamma <= 1.0, f"Gamma {gamma} is not comprised in ]0, 1]"
        assert max_weight > 1.0 , f"max_weight must be greater than 1"
        self.no_actions = no_actions
        self.weights = [1.0] * no_actions
        self.gamma = gamma
        if r is None:
            r = random
        self.r = r
        self.weight_ranges = 0
        self.max_weight = max_weight
        self._history = {}
        self.enable_history = enable_history

    @property
    def probabilities(self) -> List[float]:
        """ Returns the current probabilities for each actions. """
        assert type(self.weights) is list
        weights_sum = sum(self.weights)
        return [((1.0 - self.gamma) * (w / weights_sum)) + (self.gamma / len(self.weights)) for w in self.weights]

    def take_action(self) -> int:
        """ Returns the action that should be taken. """
        assert not self._waiting_reward, "EXP3 is waiting for reward to complete the round"
        action = self.r.choices(list(range(self.no_actions)), weights=self.probabilities, k=1)[0]
        self._waiting_reward = True
        return action

    def give_reward(self, action: int, reward: float):
        """ Gives the reward corresponding to the action taken. """
        assert 0 <= action < self.no_actions, f"Action {action} is unknown"
        assert 0 <= reward <= 1.0, f"Reward {reward} is not comprised in [0, 1]"
        assert self._waiting_reward, "EXP3 has not taken an action yet"
        estimated_reward = reward / self.probabilities[action]
        self.weights[action] *= math.exp(estimated_reward * (self.gamma * 1) / self.no_actions)
        weights_sum = sum(self.weights)
        for idx, w in enumerate(self.weights):
            self.weights[idx] *= self.max_weight / weights_sum
            self.weights[idx] = max(1, self.weights[idx])
        if self.enable_history:
            self._history[self.round] = {'weights': self.weights.copy(), 'probabilities': self.probabilities.copy(), 'action': action, 'reward': reward}
        self.round += 1
        self._waiting_reward = False

if __name__ == "__main__":
    random.seed("exp3_test")
    N_ROUNDS = 100
    exp3 = EXP3(2, gamma=1, enable_history=True)
    total_actions = 0
    for _ in range(N_ROUNDS):
        action = exp3.take_action()
        exp3.give_reward(action, 1 - action)
        total_actions += action
    assert 40 <= total_actions <= 60
    
    exp3 = EXP3(2, gamma=0.1, max_weight=100, enable_history=True)
    total_actions = 0
    for _ in range(N_ROUNDS):
        action = exp3.take_action()
        exp3.give_reward(action, 1 - action)
        total_actions += action
    assert total_actions <= 20

    convergences = []
    for _ in range(100):
        exp3 = EXP3(2, gamma=0.2, max_weight=100, enable_history=True)
        for _ in range(N_ROUNDS):
            action = exp3.take_action()
            exp3.give_reward(action, 1 - action)
            if exp3.probabilities[0] > 0.85:
                convergences.append(exp3.round)
                break
    assert sorted(convergences)[len(convergences) // 2] <= 30

    convergences = []
    for _ in range(100):
        exp3 = EXP3(2, gamma=0.1, max_weight=100, enable_history=True)
        exp3.weights = [1, 100]
        for _ in range(N_ROUNDS*10):
            action = exp3.take_action()
            exp3.give_reward(action, 1 - action)
            if exp3.probabilities[0] > 0.9:
                convergences.append(exp3.round)
                break
    assert sorted(convergences)[len(convergences) // 2] <= 150