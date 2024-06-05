from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.agents.base_agent import BaseAgent
from src.networks.actor_critic import ActorCritic


class PPOAgent(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims=(64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor_critic = ActorCritic(obs_dim, n_actions, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self._loss: float = 0.0
        self._n_updates: int = 0

    def choose_action(self, state) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.actor_critic.get_action(state_t)
        return action

    def collect_rollout(self, env, rollout_steps: int = 128) -> Dict:
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state, _ = env.reset()
        for _ in range(rollout_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, value = self.actor_critic(state_t)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state if not done else env.reset()[0]

        # bootstrap value for last state
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.actor_critic(state_t)
        values.append(last_value.item())

        return {
            "states": states, "actions": actions, "rewards": rewards,
            "dones": dones, "log_probs": log_probs, "values": values,
        }

    def compute_gae(self, rewards, values, dones) -> Tuple[List[float], List[float]]:
        advantages, returns = [], []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns

    def update(self, rollout: Dict) -> float:
        states = torch.FloatTensor(np.array(rollout["states"])).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)

        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        n = len(rollout["states"])
        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_lp = old_log_probs[mb]
                mb_adv = advantages_t[mb]
                mb_ret = returns_t[mb]

                action_probs, values = self.actor_critic(mb_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (log_probs - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values.squeeze(-1), mb_ret)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self._loss = total_loss
        self._n_updates += 1
        return total_loss

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path: str) -> None:
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def n_actions(self) -> int:
        return self.actor_critic.policy_head.out_features

    @property
    def last_loss(self) -> float:
        return self._loss
