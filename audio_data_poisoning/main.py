import pandas as pd
import torch.nn.functional as F

from audio_data_poisoning.audio_nightshade.audio_generator import AudioGenerator
from audio_data_poisoning.audio_nightshade.dataset_filter import DatasetFilter
from audio_data_poisoning.models.audio_ldm import AudioLDM
from audio_data_poisoning.models.clap import CLAP

if __name__ == "__main__":
    dataset = pd.read_csv("./audio_data_poisoning/data/audiocaps_train.csv")
    dataset = dataset["caption"].to_list()

    clap = CLAP()
    audio_ldm = AudioLDM()

    ds = DatasetFilter(dataset, clap, F.cosine_similarity)
    ag = AudioGenerator(audio_ldm)

    samples = ds.filter()
    print(samples)
    audios = ag.generate()
    print(audios)
