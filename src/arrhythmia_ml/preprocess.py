# preprocessing script for the project
import numpy as np
import neurokit2 as nk


def bandpass_1d(signal: np.ndarray,fs: int,low: float = 0.5,high: float = 30.0) -> np.ndarray:
    assert signal.ndim == 1, "Expected 1D signal array"

    y = nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=low,
        highcut=high,
    )
    return np.asarray(y, dtype=np.float32)


def bandpass_2d(signal: np.ndarray,fs: int,low: float = 0.5,high: float = 30.0) -> np.ndarray:

    signal = np.asarray(signal)
    assert signal.ndim == 2, f"Expected 2D signal array, got {signal.ndim}D"

    out = np.empty(signal.shape, dtype=np.float32)

    for ch in range(signal.shape[1]):
        out[:, ch] = bandpass_1d(signal[:, ch], fs, low, high)

    return out


def notch_filter_1d(signal: np.ndarray, fs: int, freq: float = 30.0) -> np.ndarray:
    assert signal.ndim == 1, "Expected 1D signal array"

    y = nk.signal_filter(
        signal,
        sampling_rate=fs,
        method="powerline",
        powerline=freq,
    )
    return np.asarray(y, dtype=np.float32)

def notch_filter_2d(signal: np.ndarray, fs: int, freq: float = 30.0) -> np.ndarray:

    signal = np.asarray(signal)
    assert signal.ndim == 2, f"Expected 2D signal array, got {signal.ndim}D"

    out = np.empty(signal.shape, dtype=np.float32)

    for ch in range(signal.shape[1]):
        out[:, ch] = notch_filter_1d(signal[:, ch], fs, freq)

    return out


def detrend_baseline_correct_1d(signal: np.ndarray, duration: float, fs: int) -> np.ndarray:

    signal = np.asarray(signal)
    assert signal.ndim == 1, f"Expected 1D signal array, got {signal.ndim}D"

    signal_detrend = nk.signal_detrend(signal, sampling_rate=fs)

    baseline_samples = int(duration * fs)
    baseline_samples = min(baseline_samples, signal_detrend.shape[0])

    signal_corrected = signal_detrend - np.mean(signal_detrend[:baseline_samples])

    return signal_corrected.astype(np.float32)

def detrend_baseline_correct_2d(signal: np.ndarray, duration: float, fs: int) -> np.ndarray:

    signal = np.asarray(signal)
    assert signal.ndim == 2, f"Expected 2D signal array, got {signal.ndim}D"

    out = np.empty(signal.shape, dtype=np.float32)

    for ch in range(signal.shape[1]):
        out[:, ch] = detrend_baseline_correct_1d(signal[:, ch], duration, fs)

    return out

