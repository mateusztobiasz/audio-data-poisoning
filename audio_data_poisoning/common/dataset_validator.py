import os

import pandas as pd


class DatasetValidator:
    YT_AUDIOS_DIR: str = "./audio_data_poisoning/data/yt_audios"

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)

    def validate(self, audio_column: str = "audio_path"):
        paths = self.dataset[audio_column] + ".wav"

        exists = paths.apply(os.path.exists)
        missing = self.dataset[~exists]

        for path in missing[audio_column]:
            print(f"Audio file not found: {path}")

        print(f"Removed {len(missing)} of {len(self.dataset)} rows")

        updated_len = len(self.dataset) - len(missing)
        self.dataset = self.dataset[exists].reset_index(drop=True)
        self.dataset.to_csv(self.dataset_path, index=False)

        return updated_len
