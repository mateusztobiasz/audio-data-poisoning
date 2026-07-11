import os
from typing import List

from audio_data_poisoning.common.utils import create_dataset, save_audio
from audio_data_poisoning.models.base_model import BaseModel


class AudioGenerator:
    OUTPUT_DIR: str = "./audio_data_poisoning/data/gen_samples"
    DATASET_DIR: str = "./audio_data_poisoning/data"

    def __init__(self, model: BaseModel):
        self.model = model

    def generate(
        self,
        n: int = 100,
        target_phrase: str = "a clear and high-quality recording of a cat meowing loud",
        audio_length: float = 10.0,
    ) -> List[str]:
        audios = []
        for i in range(n):
            output = self.model.model(
                prompt=target_phrase,
                num_inference_steps=50,
                audio_length_in_s=audio_length,
            )
            audio = output.audios[0]
            audio_path = f"{self.OUTPUT_DIR}/{target_phrase.replace(' ', '_')}_{i}.wav"

            save_audio(
                audio,
                audio_path,
                transform=False,
            )

            audios.append(audio_path)

        create_dataset(audios, f"{self.DATASET_DIR}/gen_samples.csv")
        return audios
