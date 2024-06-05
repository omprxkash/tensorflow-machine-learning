from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Visualizer:
    def plot_training_curves(
        self,
        rewards: List[float],
        title: str = "Training Curve",
        save_path: Optional[str] = None,
        window: int = 20,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        episodes = range(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")
        if len(rewards) >= window:
            rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(rewards) + 1), rolling, color="steelblue", linewidth=2, label=f"{window}-ep mean")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def plot_heatmap(
        self,
        Q_table: np.ndarray,
        grid_size: int,
        save_path: Optional[str] = None,
        title: str = "Value heatmap (max Q per cell)",
        dpi: int = 120,
    ) -> None:
        V = np.max(Q_table, axis=1).reshape(grid_size, grid_size)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(V, cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi)
        plt.close(fig)

    def plot_policy_arrows(
        self,
        Q_table: np.ndarray,
        grid_size: int,
        save_path: Optional[str] = None,
        title: str = "Greedy policy",
    ) -> None:
        _ARROWS = {0: (0, -0.4), 1: (0, 0.4), 2: (-0.4, 0), 3: (0.4, 0)}  # UP/DOWN/LEFT/RIGHT
        fig, ax = plt.subplots(figsize=(6, 6))
        V = np.max(Q_table, axis=1).reshape(grid_size, grid_size)
        ax.imshow(V, cmap="coolwarm", origin="upper", alpha=0.5)
        policy = np.argmax(Q_table, axis=1).reshape(grid_size, grid_size)
        for r in range(grid_size):
            for c in range(grid_size):
                dx, dy = _ARROWS[policy[r, c]]
                ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.set_title(title)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
        plt.close(fig)
