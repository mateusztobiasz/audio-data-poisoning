import librosa
import soundfile as sf
import torch
from waveglow_vocoder import WaveGlowVocoder


class Vocoder:
    def __init__(self):
        self.model = WaveGlowVocoder()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        y, _ = librosa.load(audio_path)

        return torch.from_numpy(y).to(device="cuda", dtype=torch.float32)

    def gen_mel(self, tensor: torch.Tensor) -> torch.Tensor:
        mel = self.model.wav2mel(tensor)

        return mel

    def gen_wav(self, tensor: torch.Tensor) -> torch.Tensor:
        wav = self.model.mel2wav(tensor)

        return wav

    def save_audio(self, tensor: torch.Tensor, audio_path: str) -> None:
        sf.write(audio_path, tensor[0].cpu(), samplerate=22050)
