"""Main training entrypoint.

Usage:
    python -m src.training.trainer --config-name q_learning_gridworld
    python -m src.training.trainer --config-name dqn_cartpole
    python -m src.training.trainer --config-name ppo_cartpole
"""

import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.utils.seeding import set_seed
from src.utils.logger import TrainingLogger
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer


def _make_gridworld():
    from src.environments.gridworld import GridWorld
    return GridWorld()


def _make_minigame():
    from src.environments.minigame import MiniGame
    return MiniGame()


def _make_cartpole():
    import gymnasium as gym
    return gym.make("CartPole-v1")


def train_tabular(cfg: DictConfig) -> None:
    set_seed(cfg.get("seed", 42))
    env = _make_gridworld() if cfg.training.env == "gridworld" else _make_minigame()

    if cfg.agent.type == "q_learning":
        from src.agents.q_learning import QLearningAgent
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.action_space.n,
            alpha=cfg.agent.alpha,
            gamma=cfg.agent.gamma,
            epsilon=cfg.agent.epsilon,
            epsilon_decay=cfg.agent.epsilon_decay,
            epsilon_min=cfg.agent.epsilon_min,
        )
    else:
        from src.agents.sarsa import SARSAAgent
        agent = SARSAAgent(
            n_states=env.n_states,
            n_actions=env.action_space.n,
            alpha=cfg.agent.alpha,
            gamma=cfg.agent.gamma,
            epsilon=cfg.agent.epsilon,
            epsilon_decay=cfg.agent.epsilon_decay,
            epsilon_min=cfg.agent.epsilon_min,
        )

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(
        str(ckpt_dir),
        use_tensorboard=cfg.logging.tensorboard,
    )
    evaluator = Evaluator()
    visualizer = Visualizer()
    rewards_history = []

    for ep in range(1, cfg.training.n_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(cfg.training.max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            if done:
                break

        rewards_history.append(ep_reward)
        logger.log(ep, {"reward": ep_reward, "epsilon": agent.get_epsilon()})

        if ep % cfg.training.eval_every == 0:
            stats = evaluator.evaluate(agent, env)
            rolling = float(np.mean(rewards_history[-20:]))
            print(f"  Ep {ep:4d} | reward {ep_reward:7.2f} | rolling-20 {rolling:7.2f} "
                  f"| eval {stats['mean_reward']:7.2f} | eps {agent.get_epsilon():.3f}")

    agent.save(str(ckpt_dir / "q_table.npy"))
    visualizer.plot_training_curves(
        rewards_history,
        title=f"{cfg.agent.type.upper()} on {cfg.training.env}",
        save_path=str(ckpt_dir / "training_curve.png"),
    )
    visualizer.plot_heatmap(
        agent.get_q_table(), env.size, save_path=str(ckpt_dir / "heatmap.png")
    )
    visualizer.plot_policy_arrows(
        agent.get_q_table(), env.size, save_path=str(ckpt_dir / "policy.png")
    )
    logger.close()
    print(f"\nDone. Artifacts saved to {ckpt_dir}")


def train_dqn(cfg: DictConfig) -> None:
    import gymnasium as gym
    set_seed(cfg.get("seed", 42))
    env = gym.make(cfg.training.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    from src.agents.dqn import DQNAgent
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=tuple(cfg.agent.hidden_dims),
        gamma=cfg.agent.gamma,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        buffer_size=cfg.agent.buffer_size,
        target_update_freq=cfg.agent.target_update_freq,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_end=cfg.agent.epsilon_end,
        epsilon_decay_steps=cfg.agent.epsilon_decay_steps,
    )

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(str(ckpt_dir), use_tensorboard=cfg.logging.tensorboard)
    evaluator = Evaluator()
    visualizer = Visualizer()
    rewards_history = []

    for ep in range(1, cfg.training.n_episodes + 1):
        state, _ = env.reset()
        ep_reward, ep_loss = 0.0, 0.0
        for _ in range(cfg.training.max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            loss = agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            ep_loss += loss
            state = next_state
            if done:
                break

        rewards_history.append(ep_reward)
        logger.log(ep, {"reward": ep_reward, "loss": ep_loss, "epsilon": agent.get_epsilon()})

        if ep % cfg.training.eval_every == 0:
            stats = evaluator.evaluate(agent, env)
            rolling = float(np.mean(rewards_history[-20:]))
            print(f"  Ep {ep:4d} | reward {ep_reward:7.2f} | rolling-20 {rolling:7.2f} "
                  f"| eval {stats['mean_reward']:7.2f} | eps {agent.get_epsilon():.3f}")
            agent.save(str(ckpt_dir / f"dqn_ep{ep}.pt"))

    visualizer.plot_training_curves(
        rewards_history,
        title="DQN on CartPole-v1",
        save_path=str(ckpt_dir / "training_curve.png"),
    )
    agent.save(str(ckpt_dir / "dqn_final.pt"))
    logger.close()
    best = max(rewards_history)
    print(f"\nDone. Best episode reward: {best:.1f}. Artifacts saved to {ckpt_dir}")


def train_ppo(cfg: DictConfig) -> None:
    import gymnasium as gym
    set_seed(cfg.get("seed", 42))
    env = gym.make(cfg.training.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    from src.agents.ppo import PPOAgent
    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_eps=cfg.agent.clip_eps,
        value_coef=cfg.agent.value_coef,
        entropy_coef=cfg.agent.entropy_coef,
        n_epochs=cfg.agent.n_epochs,
        batch_size=cfg.agent.batch_size,
    )

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(str(ckpt_dir), use_tensorboard=cfg.logging.tensorboard)
    visualizer = Visualizer()
    rewards_history = []
    rollout_steps = cfg.agent.rollout_steps

    for update in range(1, cfg.training.n_updates + 1):
        rollout = agent.collect_rollout(env, rollout_steps)
        loss = agent.update(rollout)

        ep_reward = sum(rollout["rewards"])
        rewards_history.append(ep_reward)
        logger.log(update, {"reward": ep_reward, "loss": loss})

        if update % cfg.training.eval_every == 0:
            rolling = float(np.mean(rewards_history[-10:]))
            print(f"  Update {update:4d} | rollout_reward {ep_reward:7.2f} | rolling-10 {rolling:7.2f}")

    agent.save(str(ckpt_dir / "ppo_final.pt"))
    visualizer.plot_training_curves(
        rewards_history,
        title="PPO on CartPole-v1",
        save_path=str(ckpt_dir / "training_curve.png"),
    )
    logger.close()
    print(f"\nDone. Artifacts saved to {ckpt_dir}")


@hydra.main(config_path="../../configs", config_name=None, version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Starting training: agent={cfg.agent.type}, seed={cfg.get('seed', 42)}")
    agent_type = cfg.agent.type
    if agent_type in ("q_learning", "sarsa"):
        train_tabular(cfg)
    elif agent_type == "dqn":
        train_dqn(cfg)
    elif agent_type == "ppo":
        train_ppo(cfg)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    main()
