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


def validate_detected_peaks(annotated:np.ndarray, detected:np.ndarray, fs:int, tol_ms:float=30) -> tuple[int,int,int]:
    """_summary_

    Args:
        annotated (np.ndarray): _description_
        detected (np.ndarray): _description_
        tol (int): tolerance for detection in samples

    Returns:
        tuple[int,int,int]: _description_
    """
    # main thing to focus on is to find the correct index from annotated to compare the detected peak to.
    # TRUE POSITIVES: Detected peaks that match with annotated peaks within tolerance
    # FALSE POSITIVES: Detected peaks that do not match annotated peaks within tolerance
    # FALSE NEGATIVES: Detected peaks that were entirely missed and had no matches with annotated peaks
   
    # sort and assure data types
    annotated = np.sort(np.asarray(annotated))
    detected = np.sort(np.asarray(detected))

    matches = [] # annotated peaks that match detected peaks within the tolerance limit
    incorrect = [] # tolerance does not match
    missed = np.zeros((len(annotated),), dtype=bool) # values from annotated peaks that do not correspond to any detected peaks (missed detections)
    
    # convert tolerance in ms to samples
    tol_samples = int(np.ceil(tol_ms * fs))
    # loop thru detected peaks
    for det_peak in detected:
        idx = np.searchsorted(annotated,det_peak) # find the insertion index of the detected peak

        # edge cases
        if idx == 0:
            # compare with the index 0 of the annotated peaks
            cand_idx = 0

        elif idx == len(annotated):
            # compare with the the last value of the annotated dataset
            cand_idx = len(annotated) - 1

        else:
            left = idx - 1
            right = idx
            cand_idx = left if abs(det_peak - annotated[left]) <= abs(det_peak - annotated[right]) else right # select cand_idx based on distance from either indices

        # fetch the annotated peak index for the candidate index and then check for tolerance
        nearest_annotated = annotated[cand_idx]

        # tolerance check + do not reuse annotated peak values
        if abs(nearest_annotated - det_peak) < tol_samples and not missed[cand_idx]:
            matches.append((det_peak,nearest_annotated))
            missed[cand_idx] = True
        else:
            incorrect.append(det_peak)

    # assign output 
    true_pos = len(matches)
    false_pos = len(incorrect)
    false_neg = len(missed[~missed])

    return true_pos, false_pos, false_neg



