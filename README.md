# tensorflow-machine-learning

Machine learning implementations across supervised learning and reinforcement learning — from clinical disease prediction models to from-scratch RL algorithm implementations.

---

## What's inside

### Supervised learning (TensorFlow / Keras)
End-to-end ML pipelines for disease prediction: heart disease classification and liver disease prediction. Covers the full workflow — data collection, preprocessing, feature engineering, model development, and evaluation.

```
data collection and preprocessing/
model development/
project planning/
final documentation and demonstration/
```

### `reinforcement-learning/`
From-scratch implementations of the core RL algorithms with training dashboards, policy visualizations, and Hydra configs.

| Algorithm | Type | Environment | Notes |
|---|---|---|---|
| Q-Learning | Off-policy TD | GridWorld | Tabular, greedy target |
| SARSA | On-policy TD | GridWorld | Tabular, on-policy target |
| DQN | Off-policy | CartPole-v1 | Neural Q-network, replay buffer, target net |
| PPO | On-policy | CartPole-v1 | Actor-critic, GAE, clipped surrogate |

```bash
cd reinforcement-learning
pip install -r requirements.txt
python src/train.py algorithm=dqn
```

---

## Stack
Python · TensorFlow · Keras · PyTorch · Gymnasium · Hydra · Weights & Biases
