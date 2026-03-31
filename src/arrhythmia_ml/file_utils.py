# * -  file utilities for the project - *
from __future__ import annotations
import wfdb 
import numpy as np
from pathlib import Path



def get_participant_ids(raw_data_path: str | Path) -> list[str]:
    """
    Get all participant ids using raw data path.

    Args:
        raw_data_path (str | Path): Path for raw data.


    Returns:
        list[str]: list of participant ids
    """
    raw_data_path = Path(raw_data_path)

    ids = sorted(
        f.stem
        for f in raw_data_path.glob("*.dat")
    )

    if not ids:
        raise ValueError(f"No .dat files found in {raw_data_path}")

    return ids
    
    
def load_raw_participant_data(raw_data_path: str, participant_id: str)-> tuple[np.ndarray, ...]:
    """
    Load ecg signal for a participant using the wfdb library.

    Args:
        raw_data_path (str): -
        participant_id (str): -

    Returns:
        tuple[np.ndarray, int,np.ndarray, list, list]: raw ecg signal, sampling freq, channel names, r_peaks and labels for beats.
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