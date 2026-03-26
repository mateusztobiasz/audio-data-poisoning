import pandas as pd
import torch

from wimudp.data_poisoning.nightshade.pipeline import Pipeline
from wimudp.data_poisoning.nightshade.vocoder import Vocoder
from wimudp.data_poisoning.utils import (
    AUDIOS_DIR,
    AUDIOS_SAMPLES_DIR,
    CSV_NS_SAMPLES_FILE,
    check_audio_file,
    normalize_tensor,
    pad_waveforms,
    read_csv,
)

MAX_EPOCHS = 500
EPS = 0.05


def generate_poison(
    row: pd.Series, vocoder: Vocoder, pipeline: Pipeline
) -> torch.Tensor:
    w_1 = vocoder.load_audio(f"{AUDIOS_SAMPLES_DIR}/{row['audio']}")
    w_2 = vocoder.load_audio(f"{AUDIOS_SAMPLES_DIR}/big.wav")

    w_1, w_2 = pad_waveforms(w_1, w_2)
    w_1_mel = vocoder.gen_mel(w_1)
    w_2_mel = vocoder.gen_mel(w_2)
    w_1_mel_norm = normalize_tensor(w_1_mel)
    w_2_mel_norm = normalize_tensor(w_2_mel)

    target_latent = pipeline.get_latent(w_2_mel_norm)
    target_latent = target_latent.detach()

    delta = torch.clone(w_1_mel_norm) * 0.0
    best_delta = torch.clone(delta)
    max_change = EPS * 2
    step_size = max_change
    min_loss = float("inf")

    for i in range(MAX_EPOCHS):
        actual_step_size = step_size - (step_size - step_size / 100) / MAX_EPOCHS * i
        delta.requires_grad_()

        pert_mel = torch.clamp(delta + w_1_mel_norm, -1, 1)
        per_latent = pipeline.get_latent(pert_mel)
        diff_latent = per_latent - target_latent

        loss = diff_latent.norm()
        grad = torch.autograd.grad(loss, delta)[0]

        if min_loss > loss:
            min_loss = loss
            best_delta = torch.clone(delta)

        delta = delta - torch.sign(grad) * actual_step_size
        delta = torch.clamp(delta, -max_change, max_change)
        delta = delta.detach()

        if i % 20 == 0:
            print(f"[{row['audio']}] in {i}. epoch - loss: {loss}")
    print(f"[{row['audio']}] min loss: {min_loss}")
    final_mel_norm = torch.clamp(best_delta + w_1_mel_norm, -1, 1)
    return normalize_tensor(final_mel_norm, True, w_1_mel.max(), w_1_mel.min())


def generate_all(df: pd.DataFrame):
    vocoder = Vocoder()
    pipeline = Pipeline()

    for _, row in df.iterrows():
        if not check_audio_file(AUDIOS_DIR, row["audio"]):
            final_mel = generate_poison(row, vocoder, pipeline)
            final_wav = vocoder.gen_wav(final_mel)
            vocoder.save_audio(final_wav, f"{AUDIOS_DIR}/{row['audio']}")
        else:
            print(f"Poison sample already present for: {row['audio']}")


if __name__ == "__main__":
    df = read_csv(CSV_NS_SAMPLES_FILE)
    generate_all(df)
