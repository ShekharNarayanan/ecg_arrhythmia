# * - extract relevant features - *
import numpy as np


def extract_waveforms(
    ecg_signal: np.ndarray,
    fs: int,
    peaks: np.ndarray,
    window_start_ms: float,
    window_end_ms: float,
) -> np.ndarray:
    """_summary_

    Args:
        ecg_signal (np.ndarray): _description_
        fs (int): _description_
        peaks (np.ndarray): _description_
        window_start_ms (float): _description_
        window_end_ms (float): _description_

    Returns:
        np.ndarray: _description_
    """

    # define dimension params
    num_beats = len(peaks)
    left = int((window_start_ms / 1000) * fs)
    right = int((window_end_ms / 1000) * fs)
    num_samples = right - left
    X = np.empty([num_beats, num_samples])
    len_signal = ecg_signal.shape[0]

    for i_beat, peak in enumerate(peaks):
        # determine start and end of window in samples
        start = peak + left
        end = peak + right

        # check for padding : extra samples that could be needed if the start or the end of the window exceed bounds of ecg_signal
        # this is needed to maintain length of segment that is assigned to X[i_beat,:]
        pad_left = max(0, -start)
        pad_right = max(0, end - len_signal)

        # make start and end indices valid
        start_clipped = max(0, start)  # dont take negative value
        end_clipped = min(len_signal, end)  # dont take a value > len(signal)

        # get ecg segment with clipped start and end
        ecg_segment = ecg_signal[start_clipped:end_clipped]

        # check if padding is needed (see if we get a non zero value for either of these vars)
        if pad_left or pad_right:
            ecg_segment = np.pad(ecg_segment, (pad_left, pad_right), mode="constant")
        
        X[i_beat,:] = ecg_segment

    return X


def return_commbined_feature_matrix(
    ecg_signal: np.ndarray,
    peaks: np.ndarray,
    fs: int,
    window_start_ms: float,
    window_end_ms: float,
) -> np.ndarray:
    """Combines all extracted features and returns them as output.

    Args:
        ecg_signal (np.ndarray): _description_
        peaks (np.ndarray): _description_
        labels (list): _description_
        fs (int): _description_
        window_start_ms (float): _description_
        window_end_ms (float): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """

    # get all extracted waveforms for participant
    X_waves = extract_waveforms(
        ecg_signal=ecg_signal,
        peaks=peaks,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        fs=fs,
    )

    # possibility to add more features to X



    return X_waves
