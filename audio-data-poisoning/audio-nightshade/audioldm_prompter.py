import subprocess

import pandas as pd

from wimudp.data_poisoning.utils import AUDIOS_SAMPLES_DIR, CONCEPT_A, CONCEPT_A_ACTION


def query_audioldm():
    caption = f"{CONCEPT_A.capitalize()} is {CONCEPT_A_ACTION}ing"
    subprocess.run(
        [
            "poetry",
            "run",
            "audioldm",
            "--model_name",
            "audioldm-s-full",
            "-t",
            f"'{caption}'",
            "-s",
            AUDIOS_SAMPLES_DIR,
        ]
    )


if __name__ == "__main__":
    query_audioldm()
