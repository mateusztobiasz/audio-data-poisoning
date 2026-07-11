import pandas as pd

from audio_data_poisoning.models.base_model import BaseModel


class PoisonGenerator:
    def __init__(
        self,
        model: BaseModel,
        alpha: float = 10.0,
        p: float = 0.05,
        steps: int = 500,
        lr: float = 0.01,
    ):
        self.model = model
        self.alpha = alpha
        self.p = p
        self.steps = steps
        self.lr = lr

    def poison_dataset(
        self, source_dataset: pd.Dataframe, target_dataset: pd.Dataframe
    ) -> pd.Dataframe:
        for i, ((_, source_row), (_, target_row)) in enumerate(
            (source_dataset.iterrows(), target_dataset.iterrows())
        ):
            source_mel = self.model.audio_to_mel(source_row["audio_path"])
            target_mel = self.model.audio_to_mel(target_row["audio_path"])

            poison_mel = self._poison_audio(source_mel, target_mel)

    def _poison_audio(self, source_mel, target_mel):
        # Implement the logic to poison the audio based on the source and target mel spectrograms
        pass
