# reinforcement-learning

From-scratch implementations of the core RL algorithms — Q-Learning, SARSA, DQN, and PPO — with training dashboards, policy visualizations, and Hydra configs. If you've ever wanted to understand how these algorithms actually work under the hood, this is where I work through them one piece at a time.

---

## What's in here

| Algorithm | Type | Environment | Key idea |
|-----------|------|-------------|----------|
| Q-Learning | Off-policy TD | GridWorld | Tabular, greedy target |
| SARSA | On-policy TD | GridWorld | Tabular, on-policy target |
| DQN | Off-policy | CartPole-v1 | Neural Q-network + replay buffer + target net |
| PPO | On-policy | CartPole-v1 | Actor-critic, GAE, clipped surrogate |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Train Q-Learning on the custom GridWorld
python -m src.training.trainer --config-name q_learning_gridworld

# Short DQN run on CartPole
python -m src.training.trainer --config-name dqn_cartpole

# PPO on CartPole
python -m src.training.trainer --config-name ppo_cartpole

# Run the tests
pytest tests/ -v
```

Results — training curves, Q-table heatmaps, checkpoints — land in `experiments/results/`.

---

## How it works

The core loop is always the same:

```
Agent ──── choose_action(state) ────►  Environment
  ▲                                         │
  │                                    step(action)
  │                                         │
  └──── update(s, a, r, s', done) ◄── (next_state, reward, done)
```

What varies between algorithms is what goes inside `update()`.

### Tabular methods (Q-Learning, SARSA)

Both maintain a `Q[n_states × n_actions]` numpy array and learn from single transitions.

**Q-Learning** (off-policy):
```
Q[s, a] ← Q[s, a] + α (r + γ · max_a' Q[s', a'] − Q[s, a])
```

**SARSA** (on-policy): the target uses the *actually chosen* next action instead of the greedy max, so the agent learns the value of the policy it's currently following — including its own exploration mistakes.

### DQN

```
  State ──► Q-Network ──► Q(s, a) for all a
                │
                └── Target Network (lagged copy, updated every C steps)
                        └── r + γ · max Q_target(s', ·)

  Replay Buffer: [(s, a, r, s', done), ...]  ← random minibatch
```

The network is `input → 128 → ReLU → 128 → ReLU → n_actions`. Loss is MSE between the predicted Q and the bootstrapped target. Adam lr=1e-3, gradient clipped at 10.

### PPO

```
  ActorCritic
  ┌──────────────────┐
  │  Shared body     │  (64 → 64)
  ├──────────────────┤
  │  Policy head     │  → action probabilities
  │  Value head      │  → V(s)
  └──────────────────┘

  Per update:
  1. Collect T-step rollout
  2. Compute GAE advantages and returns
  3. Run K=4 epochs of minibatch updates with clipped surrogate loss
```

GAE (λ=0.95) smoothly interpolates between 1-step TD and full Monte Carlo returns. The clipped objective `min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)` with ε=0.2 prevents any single update from moving the policy too far.

---

## Project structure

```
reinforcement-learning/
├── src/
│   ├── agents/          # QLearningAgent, SARSAAgent, DQNAgent, PPOAgent
│   ├── networks/        # MLP, DQNNetwork, ActorCritic
│   ├── environments/    # GridWorld, MiniGame, wrappers
│   ├── training/        # Trainer (Hydra entrypoint), ReplayBuffer, Scheduler
│   ├── evaluation/      # Evaluator, Visualizer
│   └── utils/           # seeding, logger, config
├── configs/             # Hydra YAML configs for each experiment
├── experiments/results/ # Training outputs (gitignored)
├── notebooks/           # Three Jupyter notebooks walking through each algorithm
├── tests/               # pytest suite
├── requirements.txt
└── LICENSE
```

---

## Training outputs

After a Q-Learning run you'll find:

```
experiments/results/q_learning_gridworld/
├── rewards.csv          # episode, reward, epsilon
├── training_curve.png   # reward over time with rolling mean
├── heatmap.png          # max Q per grid cell
└── policy.png           # arrows showing the greedy policy
```

---

## Testing

```bash
pytest tests/ -v
```

The suite covers GridWorld mechanics (walls, holes, goal rewards, truncation), Q-Learning update correctness and epsilon decay, and ReplayBuffer shapes, types, and capacity wrapping.

---

## Known limitations

- **CPU only** — the torch install here is CPU-only. DQN and PPO will train fine but slower than on a CUDA machine; GridWorld converges in seconds regardless.
- **Discrete actions only** — PPO uses a `Categorical` distribution. Continuous control (Pendulum, MuJoCo) would need a `Normal` policy head and a re-shaped actor network.
- **No vectorised environments** — a single env instance per training run. Wrapping with `gymnasium.vector` or something like `stable-baselines3`'s VecEnv would speed up PPO rollouts by a lot.
- **Hyperparameters are tuned for fast completion** — configs finish in under a minute on CPU. Real CartPole convergence with DQN typically needs 300–500 episodes at the full buffer size.
- **No early stopping** — training always runs the full `n_episodes`/`n_updates` even if the agent has solved the task.
