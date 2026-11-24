import os

import pandas as pd
import torch.nn.functional as F

from audio_data_poisoning.audio_nightshade.audio_generator import AudioGenerator
from audio_data_poisoning.audio_nightshade.dataset_filter import DatasetFilter
from audio_data_poisoning.common.audio_loader import AudioLoader
from audio_data_poisoning.common.dataset_validator import DatasetValidator
from audio_data_poisoning.models.audio_ldm import AudioLDM
from audio_data_poisoning.models.clap import CLAP

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
DATASET_PATH = "./audio_data_poisoning/data/audiocaps_train.csv"
FILTERED_DATASET_PATH = "./audio_data_poisoning/data/filtered_samples.csv"
DATASET_YT_PATH = "./audio_data_poisoning/data/yt_audios"
DATASET_GEN_PATH = "./audio_data_poisoning/data/gen_samples"
SOURCE_SUBJECT = "dog"  # concept C to attack
SOURCE_PHRASE = "a dog is barking"
TARGET_PHRASE = "a clear and high-quality recording of a cat meowing loudly"  # anchor A
TOP_K = 1000  # top K samples to consider for filtering
N_POISON = 50  # N_p poison samples


def main():
    # ─────────────────────────────────────────────────────
    # STAGE 1: Filter AudioCaps captions related to "dog"
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Filtering AudioCaps dataset")
    print("=" * 60)

    audiocaps = pd.read_csv(DATASET_PATH)[
        ["caption", "youtube_id", "start_time"]
    ].values.tolist()

    clap = CLAP()
    print(f"CLAP model loaded successfully, device: {clap.model.device}")
    dataset_filter = DatasetFilter(
        dataset=audiocaps,
        model=clap,
        metric=F.cosine_similarity,
    )

    if os.path.exists(FILTERED_DATASET_PATH):
        print(
            f"Filtered dataset already exists at {FILTERED_DATASET_PATH}. Skipping filtering."
        )
        filtered_dataset_path = FILTERED_DATASET_PATH
    else:
        filtered_dataset_path = dataset_filter.filter(
            target_subject=SOURCE_SUBJECT,
            target_phrase=SOURCE_PHRASE,
            top_k=TOP_K,
            n_sample=N_POISON,
        )

    # ───────────────────────────────────────────────────────────────
    # STAGE 2: Download source audios from YouTube (needed only once)
    # ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 2: Downloading source audios from YouTube")
    print("=" * 60)

    if not os.path.exists(DATASET_YT_PATH):
        downloader = AudioLoader(dataset_path=filtered_dataset_path, threads_number=5)
        downloader.download_audios_parallel()
    else:
        print(f"Dataset already exists at {DATASET_YT_PATH}. Skipping download.")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 3: Validating filtered dataset in order to remove missing audio files
    # ───────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 3: Validating filtered dataset in order to remove missing audio files")
    print("=" * 60)

    validator = DatasetValidator(dataset_path=filtered_dataset_path)
    updated_len = validator.validate()

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 4: Generating clean samples using AudioLDM
    # ───────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 4: Generating clean samples using AudioLDM")
    print("=" * 60)

    audio_ldm = AudioLDM()
    print(f"AudioLDM model loaded successfully, device: {audio_ldm.model.device}")

    if not os.path.exists(DATASET_GEN_PATH):
        audio_generator = AudioGenerator(model=audio_ldm)
        audio_generator.generate(n=updated_len)
    else:
        print(
            f"Generated dataset already exists at {DATASET_GEN_PATH}. Skipping generation."
        )

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 5: Generating poison samples from filtered dataset and clean samples
    # ───────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
