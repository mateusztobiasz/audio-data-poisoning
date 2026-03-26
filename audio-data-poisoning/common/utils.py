import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F

AUDIOLDM_DATASET_DIR = "./finetuning/audioldm/data/dataset"
AUDIOS_DIR = f"{AUDIOLDM_DATASET_DIR}/audioset"
CONCEPT_A = "dog"
CONCEPT_C = "cat"
CONCEPT_A_ACTION = "bark"
CONCEPT_C_ACTION = "meow"
DATA_DIR = "./data"
CSV_DATASET_FILE = f"{DATA_DIR}/audiocaps_train.csv"
AUDIOS_SAMPLES_DIR = f"{DATA_DIR}/audios"
CSV_CONCEPT_C_FILE = f"{DATA_DIR}/audiocaps_{CONCEPT_C}.csv"
CSV_NS_SAMPLES_FILE = f"{AUDIOLDM_DATASET_DIR}/{CONCEPT_C}_samples.csv"
CSV_MISMATCHED_FILE = f"{AUDIOLDM_DATASET_DIR}/audioset_{CONCEPT_C}_{CONCEPT_A}.csv"
ROWS_NUMBER = 3000
THREADS_NUMBER = 20
SAMPLES_NUMBER = 450


def read_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, skipinitialspace=True)

    return df


def check_audio_file(dir_relative_path: str, audio_file: str) -> bool:
    file_path = os.path.join(os.getcwd(), dir_relative_path, audio_file)

    return os.path.exists(file_path)


def pad_waveforms(
    w_1: torch.Tensor, w_2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    l_w_1 = len(w_1)
    l_w_2 = len(w_2)

    if l_w_1 > l_w_2:
        return w_1, F.pad(w_2, (0, l_w_1 - l_w_2))
    else:
        return F.pad(w_1, (0, l_w_2 - l_w_1)), w_2


def normalize_tensor(
    tensor: torch.Tensor,
    reverse: bool = False,
    original_max: float = 0,
    original_min: float = 0,
) -> torch.Tensor:
    if not reverse:
        return (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 2 - 1
    else:
        return (tensor + 1) / 2 * (original_max - original_min) + original_min
