# * - extract featiures from ECG signals - *
import numpy as np
from arrhythmia_ml import preprocess
from scipy.signal import find_peaks


def derivative_1d(signal: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        signal (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    assert signal.ndim == 1, "Expected 1D signal array"

    derivative = np.diff(signal, prepend=signal[0])
    return derivative


def square_1d(signal: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        signal (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    assert signal.ndim == 1, "Expected 1D signal array"

    squared = np.power(signal, 2)
    return squared


def moving_average_1d(signal: np.ndarray, window_size_s: float, fs: int) -> np.ndarray:
    """_summary_

    Args:
        signal (np.ndarray): _description_
        window_size_s (float): _description_
        fs (int): _description_

    Returns:
        np.ndarray: _description_
    """
    assert signal.ndim == 1, "Expected 1D signal array"

    window_size = max(1, int(window_size_s * fs))
    smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode="same")
    return smoothed


def extract_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """_summary_

    Args:
        signal (np.ndarray): _description_
        fs (int): _description_

    Returns:
        np.ndarray: _description_
    """
    filtered_signal = preprocess.bandpass_1d(signal, fs=fs, low=5.0, high=15.0)
    derivative = derivative_1d(filtered_signal)
    squared = square_1d(derivative)
    mwi_sig = moving_average_1d(squared, window_size_s=0.150, fs=fs)
    height = np.mean(mwi_sig) + 0.5 * np.std(mwi_sig)

    peaks, _ = find_peaks(
        mwi_sig, height=height, distance=0.2 * fs
    )  # at least 200ms between peaks

    return peaks


def calibrate_r_peaks(
    signal: np.ndarray,
    candidate_peaks: np.ndarray,
    fs: int,
    search_radius_ms: float = 80.0,
) -> np.ndarray:
    """
    Refine approximate R-peak locations.

    For each candidate peak index, search within ±search_radius_ms
    and move the peak to the true local maximum of the signal.

    Parameters
    ----------
    signal : 1D ECG signal
    candidate_peaks : array of rough R-peak indices
    fs : sampling frequency (Hz)
    search_radius_ms : search window radius in milliseconds

    Returns
    -------
    refined_peaks : array of corrected R-peak indices
    """

    assert signal.ndim == 1, "Signal must be 1D"

    # Convert milliseconds to samples
    radius_samples = int((search_radius_ms / 1000) * fs)

    refined_peaks = []

    for peak_idx in candidate_peaks:
        # Define search window boundaries
        window_start = max(0, peak_idx - radius_samples)
        window_end = min(len(signal), peak_idx + radius_samples)

        # Extract local signal segment
        window = signal[window_start:window_end]

        if len(window) == 0:
            continue

        # Find index of local maximum in window
        local_max_offset = np.argmax(window)

        # Convert local index back to global signal index
        refined_peak = window_start + local_max_offset

        refined_peaks.append(refined_peak)

    return np.array(refined_peaks, dtype=int)


def validate_detected_peaks():
    pass
