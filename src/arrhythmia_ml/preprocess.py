# * -  preprocessing script for the project - *
import numpy as np
import neurokit2 as nk


# --------------------------------------------------------------------------------  helpers ----------------------------------------------------------------------------------------------------------------------------------------------------
def bandpass_1d(
    signal: np.ndarray, fs: int, low: float = 0.5, high: float = 30.0
) -> np.ndarray:
    """
    Bandpass filter for 1d signal. Takes in a low pass and highpass limit.

    Returns:
    band pass filtered signal
    """
    assert signal.ndim == 1, "Expected 1D signal array"

    y = nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=low,
        highcut=high,
    )
    return np.asarray(y, dtype=np.float32)


def notch_filter_1d(signal: np.ndarray, fs: int, freq: int = 50) -> np.ndarray:
    """1d notch filter. Defaults at 50Hz
    """
    assert signal.ndim == 1, "Expected 1D signal array"

    y = nk.signal_filter(
        signal,
        sampling_rate=fs,
        method="powerline",
        powerline=freq,
    )
    return np.asarray(y, dtype=np.float32)


def detrend_baseline_correct_1d(
    signal: np.ndarray, duration: float, fs: int
) -> np.ndarray:
    """
    Detrend signal and correct basline.

    Args:
        signal (np.ndarray): ECG signal in this case.
        duration (float): Duration for baseline correction
        fs (int): Sampling rate of the signal

    Returns:
        np.ndarray: corrected signal
    """

    signal = np.asarray(signal)
    assert signal.ndim == 1, f"Expected 1D signal array, got {signal.ndim}D"

    signal_detrend = nk.signal_detrend(signal, sampling_rate=fs)

    baseline_samples = int(duration * fs)
    baseline_samples = min(baseline_samples, signal_detrend.shape[0])

    signal_corrected = signal_detrend - np.mean(signal_detrend[:baseline_samples])

    return signal_corrected.astype(np.float32)


# -------------------------------------------------------  use cases --------------------------------------------------------------------------
def get_preprocessed_signal(raw_sig: np.ndarray, fs: int, bandpass_window: list):
    """
    Return a signal that is band pass filtered, detrended and baseline corrected.

    Args:
        raw_sig (np.ndarray): -
        fs (int):-
        bandpass_window (list): -

    """
    processed_sig = notch_filter_1d(
        bandpass_1d(raw_sig, fs=fs, low=bandpass_window[0], high=bandpass_window[1]),
        fs=fs,
        freq=50,
    )
    return processed_sig
