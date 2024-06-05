from typing import Dict

import numpy as np


class Evaluator:
    def evaluate(self, agent, env, n_episodes: int = 10) -> Dict[str, float]:
        rewards, lengths = [], []
        was_training = agent._training
        agent.set_training(False)
        for _ in range(n_episodes):
            state, _ = env.reset()
            ep_reward, ep_len = 0.0, 0
            done = False
            while not done:
                action = agent.choose_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_len += 1
                done = terminated or truncated
            rewards.append(ep_reward)
            lengths.append(ep_len)
        agent.set_training(was_training)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards) if len(rewards) > 1 else 0.0),
            "mean_steps": float(np.mean(lengths)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }

    def evaluate_tabular(self, agent, env, n_episodes: int = 10) -> Dict[str, float]:
        return self.evaluate(agent, env, n_episodes)
