from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import pandas as pd
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, download_range_func

from wimudp.data_poisoning.utils import (
    AUDIOS_SAMPLES_DIR,
    CSV_CONCEPT_C_FILE,
    THREADS_NUMBER,
    read_csv,
)


def build_urls_and_ranges(df: pd.DataFrame) -> Tuple[List[str], List[Tuple[int]]]:
    yt_ids = df["youtube_id"].to_list()
    start_times = df["start_time"].to_list()

    yt_urls = list(map(lambda id: f"https://www.youtube.com/watch?v={id}", yt_ids))
    ranges = list(map(lambda st: (st, st + 10), start_times))

    return yt_urls, ranges


def download_audios_in_batch(urls_batch: List[str], ranges_batch: List[Tuple[int]]):
    for url, range in zip(urls_batch, ranges_batch):
        with YoutubeDL(setup_yt_dlp(range)) as ydl:
            try:
                ydl.download([url])
            except DownloadError as de:
                print(f"Cannot download audio with url: {url}. {de}")


def download_audios_parallel(
    urls: List[str], ranges: List[Tuple[int]], num_threads: int
):
    batch_size = len(urls) // num_threads
    urls_batches = [urls[i : i + batch_size] for i in range(0, len(urls), batch_size)]
    ranges_batches = [
        ranges[i : i + batch_size] for i in range(0, len(ranges), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(
            lambda data_batch: download_audios_in_batch(data_batch[0], data_batch[1]),
            zip(urls_batches, ranges_batches),
        )


def setup_yt_dlp(range: Tuple[int]) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": f"{AUDIOS_SAMPLES_DIR}/%(id)s.%(ext)s",
        "download_ranges": download_range_func(None, [range]),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "force_keyframes_at_cuts": True,
        "quiet": True,
    }


if __name__ == "__main__":
    df = read_csv(CSV_CONCEPT_C_FILE)
    yt_urls, ranges = build_urls_and_ranges(df)
    download_audios_parallel(yt_urls, ranges, THREADS_NUMBER)
