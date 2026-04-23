import random
from typing import Callable, List

import torch

from audio_data_poisoning.models.base_model import BaseModel


class DatasetFilter:
    BATCH_SIZE: int = 100

    def __init__(
        self,
        dataset: List[str],
        model: BaseModel,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.dataset = dataset
        self.model = model
        self.metric = metric

    def filter(
        self,
        target_subject: str = "dog",
        target_phrase: str = "dog is barking",
        top_k: int = 5000,
        n_sample: int = 100,
    ) -> List[str]:
        self.dataset = self._initial_filter(target_subject)
        similarities = self._get_similarities(target_phrase)

        top_k = min(top_k, len(self.dataset))
        top_indices = torch.topk(similarities, top_k).indices.cpu().tolist()

        top_samples = [self.dataset[i] for i in top_indices]
        top_samples = random.sample(top_samples, min(n_sample, len(top_samples)))

        return top_samples

    def _initial_filter(self, target: str) -> List[str]:
        return [data for data in self.dataset if target in data.lower()]

    def _get_similarities(self, target: str) -> torch.Tensor:
        target_feature = self.model.get_features([target]).squeeze(0)
        features = []

        for i in range(0, len(self.dataset), self.BATCH_SIZE):
            batch = self.dataset[i : i + self.BATCH_SIZE]
            batch_features = self.model.get_features(batch)
            features.append(batch_features)

        features = torch.cat(features, dim=0)
        return self.metric(target_feature, features)
