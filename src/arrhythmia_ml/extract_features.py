# * - extract relevant features - *
import numpy as np


def extract_waveforms(
    ecg_signal: np.ndarray,
    fs: int,
    r_peaks: np.ndarray,
    window_start_ms: float,
    window_end_ms: float,
) -> np.ndarray:
    """_summary_

    Args:
        ecg_signal (np.ndarray): _description_
        fs (int): _description_
        r_peaks (np.ndarray): _description_
        window_start_ms (float): _description_
        window_end_ms (float): _description_

    Returns:
        np.ndarray: _description_
    """

    # define dimension params
    num_beats = len(r_peaks)
    left = int((window_start_ms / 1000) * fs)
    right = int((window_end_ms / 1000) * fs)
    num_samples = right - left
    X = np.empty([num_beats, num_samples])
    len_signal = ecg_signal.shape[0]

    for i_beat, peak in enumerate(r_peaks):
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

        X[i_beat, :] = ecg_segment

    return X


def compute_pre_post_delta_rr(
    r_peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Caclulate duration before and after the beat and the acceleration of rhythm.
     Ectopic beats tend to have irregular coupling intervals.

    Args:
        r_peaks (np.ndarray): _description_

    Returns:
        tuple[np.ndarray,np.ndarray]: _description_
    """
    # calculate interval difference and initialize output
    peak_intervals = np.diff(r_peaks)
    pre_rr = np.zeros((len(r_peaks),))
    post_rr = np.zeros((len(r_peaks),))

    # get acceleration of rhythm
    delta_rr = np.diff(peak_intervals)
    delta_rr = np.pad(delta_rr, pad_width=1, mode="edge") # pad it so length becomes the same as num beats

    # loop over r_peaks and store pre and post interval durations
    for p_ind in range(len(r_peaks)):
        # decide limits for indices for both arrays based on peak_intevals
        pre_rr_ind = max(0, p_ind - 1)
        post_rr_ind = min(len(peak_intervals) - 1, p_ind)

        # append appropriate values
        pre_rr[p_ind] = peak_intervals[pre_rr_ind]
        post_rr[p_ind] = peak_intervals[post_rr_ind]

    return pre_rr, post_rr, delta_rr


def pre_and_post_rr_ratio(pre_rr: np.ndarray, post_rr: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        pre_rr (np.ndarray): _description_
        post_rr (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    return pre_rr / post_rr


def compute_local_rr_mean(
    r_peaks: np.ndarray, local_rr_mean_beat_window: int
) -> np.ndarray:
    """Compute a local mean for peak intervals in a specific beat window,
    i.e. for each peak compute the mean of a number of peak intervals around it.
    This gives a gives a "what's normal for this patient right now" baseline.
    Deviations from local mean is very different for normal vs abnormal beats.

    Args:
        r_peaks (np.ndarray): _description_
        beat_window (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    # find peak intervals
    peak_intervals = np.diff(r_peaks)

    # initialize results
    rr_mean = np.zeros((len(r_peaks),))

    for p_ind in range(len(r_peaks)):
        # define left and right limits for peak_interval splicing based on window
        left = max(0, p_ind - local_rr_mean_beat_window)
        right = min(len(peak_intervals) - 1, p_ind + local_rr_mean_beat_window)

        # assign result
        rr_mean[p_ind] = np.mean(peak_intervals[left:right])

    return rr_mean


def compute_deviation_from_local_mean(
    local_rr_mean: np.ndarray, pre_rr: np.ndarray
) -> np.ndarray:
    """
    Compute how much a peak interval deviates from the local peak interval mean for each beat.
    Deviations from local mean is very different for normal vs abnormal beats.

    Args:
        local_rr_mean (np.ndarray): _description_
        pre_rr (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    return np.array([pre_i / mean_i for pre_i, mean_i in zip(pre_rr, local_rr_mean)])


def extract_all_rr_features(
    r_peaks: np.ndarray, local_rr_mean_beat_window: int
) -> dict:
    """_summary_

    Args:
        r_peaks (np.ndarray): _description_
        local_rr_mean_beat_window (int): _description_

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
    """

    # get pre and post interval durations along with the rate of acceleration of rhythms
    pre_rr, post_rr, delta_rr = compute_pre_post_delta_rr(r_peaks=r_peaks)

    # get ratio
    pre_post_rr_ratio = pre_and_post_rr_ratio(pre_rr=pre_rr, post_rr=post_rr)

    # get local interval duration mean for each beat
    local_rr_mean = compute_local_rr_mean(
        r_peaks=r_peaks, local_rr_mean_beat_window=local_rr_mean_beat_window
    )

    # get deviation from local mean for each beat
    deviation_rr = compute_deviation_from_local_mean(
        local_rr_mean=local_rr_mean, pre_rr=pre_rr
    )

    return {
    "pre_rr": pre_rr,
    "post_rr": post_rr,
    "delta_rr": delta_rr,
    "pre_post_rr_ratio": pre_post_rr_ratio,
    "local_rr_mean": local_rr_mean,
    "deviation_rr": deviation_rr,
    }


def return_commbined_feature_matrix(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    window_start_ms: float,
    window_end_ms: float,
    local_rr_beat_window: int=5,
    compute_only: list[str] | None = None
) -> tuple[np.ndarray | None, ...]:
    """Combines all extracted features and returns them as output.

    Args:
        ecg_signal (np.ndarray): _description_
        r_peaks (np.ndarray): _description_
        labels (list): _description_
        fs (int): _description_
        window_start_ms (float): _description_
        window_end_ms (float): _description_

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: _description_
    """
    if compute_only is None:
        compute_only = ["waves","interval_related"]

    
    if "waves" in compute_only:
        # get all extracted waveforms for participant
        X_waves = extract_waveforms(
            ecg_signal=ecg_signal,
            r_peaks=r_peaks,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            fs=fs,
        )
    else:
        X_waves = None

    if "interval_related" in compute_only:
        # get r_peak interval related features and corresponding feature vector
        rr_dict = extract_all_rr_features(r_peaks=r_peaks,local_rr_mean_beat_window=local_rr_beat_window)
        X_rr = np.column_stack(list(rr_dict.values()))
    else:
        X_rr = None
    

    return X_waves, X_rr

