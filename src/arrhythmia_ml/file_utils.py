# * -  file utilities for the project - *
from __future__ import annotations
import yaml
import wfdb 
import numpy as np
from pathlib import Path



def load_config():
    with open(Path(__file__).resolve().parents[2] / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config


def get_participant_ids(raw_data_path: str | Path) -> list[str]:
    """_summary_

    Args:
        raw_data_path (str | Path): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[str]: _description_
    """
    raw_data_path = Path(raw_data_path)

    ids = sorted(
        f.stem
        for f in raw_data_path.glob("*.dat")
    )

    if not ids:
        raise ValueError(f"No .dat files found in {raw_data_path}")

    return ids
    
    
def load_raw_participant_data(raw_data_path: str, participant_id: str):
    """_summary_

    Args:
        raw_data_path (str): _description_
        participant_id (str): _description_

    Returns:
        _type_: _description_
    """
    record_path = Path(raw_data_path) / participant_id

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    signal = np.asarray(record.p_signal, dtype=np.float64)  # type: ignore[attr-defined]
    r_peaks = np.asarray(annotation.sample, dtype=np.int64)  # type: ignore[attr-defined]
    fs = int(record.fs)                                  # type: ignore[attr-defined]
    channels = list(record.sig_name)                       # type: ignore[attr-defined]
    labels = list(annotation.symbol)                       # type: ignore[attr-defined]

    return signal, fs, channels, r_peaks, labels

