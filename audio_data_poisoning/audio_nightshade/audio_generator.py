import os

from audio_data_poisoning.common.utils import save_audio
from audio_data_poisoning.models.base_model import BaseModel


class AudioGenerator:
    OUTPUT_DIR: str = "./audio_data_poisoning/data/gen_samples"

    def __init__(self, model: BaseModel):
        self.model = model
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def generate(
        self,
        n: int = 100,
        target_phrase: str = "a clear and high-quality recording of a cat meowing loud",
        audio_length: float = 10.0,
        save: bool = True,
    ) -> list:
        audios = []
        for i in range(n):
            output = self.model.model(
                prompt=target_phrase,
                num_inference_steps=50,
                audio_length_in_s=audio_length,
            )
            audio = output.audios[0]
            audios.append(audio)

            if save:
                save_audio(
                    audio,
                    f"{self.OUTPUT_DIR}/{target_phrase.replace(' ', '_')}_{i}.wav",
                    transform=False,
                )

        return audios
