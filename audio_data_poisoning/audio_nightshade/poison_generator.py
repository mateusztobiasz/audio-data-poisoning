import os
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F

from audio_data_poisoning.common.utils import create_dataset, save_audio
from audio_data_poisoning.models.base_model import BaseModel


class PoisonGenerator:
    OUTPUT_DIR: str = "./audio_data_poisoning/data/poison_samples"
    DATASET_DIR: str = "./audio_data_poisoning/data"

    def __init__(
        self,
        model: BaseModel,
        alpha: float = 10.0,  # penalty weight
        p: float = 0.05,  # perturbation budget
        steps: int = 100,  # optimization steps
        lr: float = 0.01,  # learning rate
    ):
        self.model = model
        self.alpha = alpha
        self.p = p
        self.steps = steps
        self.lr = lr
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def poison_dataset(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        poison_pairs = []

        for i, ((_, source_row), (_, target_row)) in enumerate(
            zip(source_dataset.iterrows(), target_dataset.iterrows())
        ):

            poisoned_mel = self._poison_audio(
                source_row["audio_path"], target_row["audio_path"]
            )

            poison_path = f"{self.OUTPUT_DIR}/poison_{i}.wav"
            if save:
                wav = self.model.mel_to_audio(poisoned_mel, save=False)
                save_audio(wav, poison_path, sr=self.model.SR)

            poison_pairs.append(
                {
                    "audio_path": poison_path,
                    "prompt": source_row["prompt"],
                }
            )

        create_dataset(
            poison_pairs,
            f"{self.DATASET_DIR}/final_samples.csv",
            additional_columns=["caption"],
        )
        return pd.DataFrame(poison_pairs)

    def _poison_audio(
        self,
        source_path: str,
        target_path: str,
    ) -> torch.Tensor:

        device = self.model.model.device
        source_mel = self.model.audio_to_mel(source_path).to(device).detach()

        with torch.no_grad():
            target_latent = self.model.get_features(target_path)

        delta = torch.zeros_like(source_mel, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.steps):
            optimizer.zero_grad()

            poisoned_mel = source_mel + delta

            poisoned_latent = self.model.model.vae.encode(
                poisoned_mel
            ).latent_dist.sample()

            feature_loss = F.mse_loss(poisoned_latent, target_latent)
            delta_norm = torch.norm(delta)
            penalty = self.alpha * torch.clamp(delta_norm - self.p, min=0)

            loss = feature_loss + penalty
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"  step {step}: feature_loss={feature_loss.item():.4f}, "
                    f"penalty={penalty.item():.4f}, "
                    f"||delta||={delta_norm.item():.4f}"
                )

        return (source_mel + delta).detach()
