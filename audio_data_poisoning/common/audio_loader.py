from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import pandas as pd
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, download_range_func


class AudioLoader:
    AUDIO_LENGTH: int = 10

    def __init__(self, dataset_path: str, threads_number: int = 10):
        self.df = pd.read_csv(dataset_path)
        self.threads_number = threads_number

    def download_audios_parallel(self):
        urls, ranges = self._build_urls_and_ranges()

        def thread_task(i):
            start, end = self._get_batch_range(len(urls), i)
            self._download_audios_in_batch(urls[start:end], ranges[start:end])

        with ThreadPoolExecutor(max_workers=self.threads_number) as executor:
            executor.map(thread_task, range(self.threads_number))

    def _build_urls_and_ranges(self) -> Tuple[List[str], List[Tuple[int]]]:
        yt_ids = self.df["youtube_id"].to_list()
        start_times = self.df["start_time"].to_list()

        yt_urls = [f"https://www.youtube.com/watch?v={id}" for id in yt_ids]
        ranges = [(st, st + self.AUDIO_LENGTH) for st in start_times]

        return yt_urls, ranges

    def _get_batch_range(self, data_length: int, index: int) -> Tuple[int, int]:
        batch_size, remainder = divmod(data_length, self.threads_number)
        start = index * batch_size + min(index, remainder)
        end = start + batch_size + (1 if index < remainder else 0)
        return start, end

    def _download_audios_in_batch(
        self, urls_batch: List[str], ranges_batch: List[Tuple[int, int]]
    ):
        for url, time_range in zip(urls_batch, ranges_batch):
            with YoutubeDL(self._get_yt_dlp_params(time_range)) as ydl:
                try:
                    ydl.download([url])
                except DownloadError as de:
                    print(f"Cannot download audio with url: {url}. {de}")

    def _get_yt_dlp_params(self, time_range: Tuple[int, int]) -> dict:
        return {
            "format": "bestaudio/best",
            "download_ranges": download_range_func(None, [time_range]),
            "force_keyframes_at_cuts": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "outtmpl": "output/%(title)s.%(ext)s",
        }


if __name__ == "__main__":
    loader = AudioLoader(
        "./audio_data_poisoning/data/audiocaps_train.csv", threads_number=5
    )
    loader.download_audios_parallel()
