import pandas as pd
import torch.nn.functional as F

from audio_data_poisoning.audio_nightshade.dataset_filter import DatasetFilter
from audio_data_poisoning.models.clap import CLAP


if __name__ == "__main__":
    dataset = pd.read_csv("./audio_data_poisoning/data/audiocaps_train.csv")
    dataset = dataset["caption"].to_list()

    bm = CLAP()
    ds = DatasetFilter(dataset, bm, F.cosine_similarity)
    samples = ds.filter()
    print(samples)
