import csv
import os
from pathlib import Path
from typing import Dict, Optional


class TrainingLogger:
    def __init__(self, log_dir: str, use_tensorboard: bool = False, csv_filename: str = "rewards.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / csv_filename
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = None

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                pass

    def log(self, episode: int, metrics: Dict[str, float]) -> None:
        if self._writer is None:
            fieldnames = ["episode"] + list(metrics.keys())
            self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._writer.writeheader()
        row = {"episode": episode, **metrics}
        self._writer.writerow(row)
        self._csv_file.flush()

        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, episode)

    def close(self) -> None:
        self._csv_file.flush()
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
