from abc import ABCMeta
from ast import List
from email.mime import audio
from threading import Lock

import pandas as pd
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


def create_dataset(
    data: list,
    file_path: str,
    audio_column: str = "audio_path",
    prompt_column: str | None = None,
):
    if prompt_column:
        df = pd.DataFrame(data, columns=[audio_column, prompt_column])
    else:
        df = pd.DataFrame(data, columns=[audio_column])

    df.to_csv(file_path, index=False)


class SingletonMeta(ABCMeta):

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):

        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]
