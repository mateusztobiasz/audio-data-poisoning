from abc import ABCMeta
from email.mime import audio
from threading import Lock

import scipy
import scipy
import soundfile as sf
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_audio(
    waveform: torch.Tensor, path: str, sr: int = 16000, transform: bool = True
):
    if transform:
        waveform = waveform.detach().cpu().numpy()

    scipy.io.wavfile.write(path, rate=sr, data=waveform)


class SingletonMeta(ABCMeta):

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):

        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]
