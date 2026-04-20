import random
from typing import Callable, List

import torch

from audio_data_poisoning.models.base_model import BaseModel


class DatasetFilter:
    BATCH_SIZE: int = 64

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
        self, target: str = "dog", top_k: int = 5000, n_sample: int = 100
    ) -> List[str]:
        similarities = self._get_similarities(target)

        top_k = min(top_k, len(self.dataset))
        top_indices = torch.topk(similarities, top_k).indices.cpu().tolist()

        top_samples = [self.dataset[i] for i in top_indices]
        top_samples = random.sample(top_samples, min(n_sample, len(top_samples)))

        return top_samples

    def _get_similarities(self, target: str) -> torch.Tensor:
        target_feature = self.model.get_features([target])
        similarities = []

        for i in range(0, len(self.dataset), self.BATCH_SIZE):
            batch = self.dataset[i : i + self.BATCH_SIZE]
            batch_features = self.model.get_features(batch)
            batch_similarities = self.metric(target_feature, batch_features)
            similarities.extend(batch_similarities)

        return torch.tensor(similarities)
