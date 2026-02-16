# * -  file utilities for the project - *
import yaml
import wfdb 
from pathlib import Path
import os



def load_config():
    with open(Path(__file__).resolve().parents[2] / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config


def get_participant_ids(raw_data_path: str | Path) -> list[str]:
    raw_data_path = Path(raw_data_path)

    ids = sorted(
        f.stem
        for f in raw_data_path.glob("*.dat")
    )

    if not ids:
        raise ValueError(f"No .dat files found in {raw_data_path}")

    return ids
    
    
def load_raw_participant_data(raw_data_path: str, participant_id: str):
    record_path = Path(raw_data_path) / participant_id

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    signal = record.p_signal
    fs = record.fs
    channels = record.sig_name

    r_peaks = annotation.sample
    labels = annotation.symbol

    return signal, fs, channels, r_peaks, labels

