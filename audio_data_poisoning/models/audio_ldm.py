import librosa
import numpy as np
import torch

from diffusers import AudioLDMPipeline

from audio_data_poisoning.common.utils import save_audio
from audio_data_poisoning.models.base_model import BaseModel


class AudioLDM(BaseModel):
    SR: int = 16000
    N_FFT: int = 1024
    HOP_LENGTH: int = 160
    N_MELS: int = 64

    def __init__(
        self,
        model_type: str = "cvssp/audioldm",
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            model=AudioLDMPipeline,
            model_type=model_type,
            torch_dtype=torch_dtype,
        )

    def get_features(self, audio_path: str, *args, **kwargs) -> torch.Tensor:
        audio = self.audio_to_mel(audio_path).to(device=self.model.device)
        with torch.no_grad():
            return self.model.vae.encode(audio).latent_dist.sample()

    def mel_to_audio(self, mel: torch.Tensor, save: bool = True) -> torch.Tensor:
        mel = mel.to(device=self.model.device)
        mel = mel.squeeze().squeeze()
        mel = mel.T

        with torch.no_grad():
            wav = self.model.vocoder(mel)

        if save:
            save_audio(wav, "output/test_1.wav", sr=self.SR)

        return wav

    def audio_to_mel(self, audio_path: str) -> torch.Tensor:
        audio, _ = librosa.load(audio_path, sr=self.SR)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        return self._normalize_mel(mel_db)

    def _normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        mel = (mel - mel.min()) / (mel.max() - mel.min()) * 2 - 1

        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    model = AudioLDM()
    mel = model.audio_to_mel("./output/test.wav")
    model.mel_to_audio(mel)
