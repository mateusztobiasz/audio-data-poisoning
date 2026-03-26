import pandas as pd

from wimudp.data_poisoning.utils import (
    CONCEPT_A,
    CONCEPT_C_ACTION,
    CSV_CONCEPT_C_FILE,
    CSV_DATASET_FILE,
    ROWS_NUMBER,
    read_csv,
)


def process_csv_file(
    concept_a: str = None, concept_c_action: str = None, rows_number: int = None
) -> pd.DataFrame:
    concept_a = concept_a if concept_a is not None else CONCEPT_A
    concept_c_action = (
        concept_c_action if concept_c_action is not None else CONCEPT_C_ACTION
    )
    rows_number = rows_number if rows_number is not None else ROWS_NUMBER

    df = read_csv(CSV_DATASET_FILE)
    filtered_indexes = df.apply(
        lambda row: filter_caption_len(row, concept_a, concept_c_action), axis=1
    )
    filtered_df = df[filtered_indexes]

    return filtered_df.head(rows_number)


def filter_caption_len(row: pd.Series, concept_a: str, concept_c_action: str) -> bool:
    return concept_c_action in row["caption"] and concept_a not in row["caption"]


if __name__ == "__main__":
    df = process_csv_file()
    df.head(ROWS_NUMBER).to_csv(CSV_CONCEPT_C_FILE)
