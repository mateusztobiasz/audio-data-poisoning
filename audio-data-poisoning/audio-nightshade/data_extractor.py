import os

import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

from wimudp.data_poisoning.nightshade.clap import CLAP
from wimudp.data_poisoning.utils import (
    AUDIOS_SAMPLES_DIR,
    CONCEPT_C,
    CONCEPT_C_ACTION,
    CSV_CONCEPT_C_FILE,
    CSV_NS_SAMPLES_FILE,
    SAMPLES_NUMBER,
    check_audio_file,
    read_csv,
)


def get_samples(
    concept_c: str = None, concept_c_action: str = None, samples_number: int = None
) -> pd.DataFrame:
    # Use default values if arguments are None
    concept_c = concept_c if concept_c is not None else CONCEPT_C
    concept_c_action = (
        concept_c_action if concept_c_action is not None else CONCEPT_C_ACTION
    )
    samples_number = samples_number if samples_number is not None else SAMPLES_NUMBER

    df = read_csv(CSV_CONCEPT_C_FILE)
    similarities = calculate_similarities(df, concept_c, concept_c_action)
    candidates = get_top_candidates(df, similarities, samples_number)

    return candidates


def calculate_similarities(
    df: pd.DataFrame, concept_c: str, concept_c_action: str
) -> torch.Tensor:
    target_caption = [f"{concept_c.capitalize()} is {concept_c_action}ing"]
    captions = df["caption"].to_list()
    clap = CLAP()

    target_caption_emb = clap.get_text_features(target_caption)
    captions_emb = clap.get_text_features(captions)

    return cosine_similarity(target_caption_emb, captions_emb)


def get_top_candidates(
    df: pd.DataFrame, similarities: torch.Tensor, samples_number: int
):
    candidates_indices = torch.argsort(similarities, descending=True)[:samples_number]
    candidates_df = pd.DataFrame(columns=["audio", "caption"])

    for i in candidates_indices:
        index = i.item()
        if not check_audio_file(
            AUDIOS_SAMPLES_DIR, f"{df.iloc[index]['youtube_id']}.wav"
        ):
            continue

        candidates_df.loc[index] = [
            f"{df.iloc[index]['youtube_id']}.wav",
            df.iloc[index]["caption"],
        ]

    return candidates_df


if __name__ == "__main__":
    samples = get_samples()

    samples.to_csv(CSV_NS_SAMPLES_FILE, index=False)
