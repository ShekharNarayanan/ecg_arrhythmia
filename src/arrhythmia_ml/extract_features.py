# * - extract relevant features - *
import numpy as np
import pywt

# --------------------------------- waveform features -----------------------------------------------------------------------------------------------
def extract_waveforms(
    ecg_signal: np.ndarray,
    fs: int,
    r_peaks: np.ndarray,
    window_start_ms: float,
    window_end_ms: float,
) -> np.ndarray:
    """
    Extract pqrs waveforms.

    Args:
        ecg_signal (np.ndarray): -
        fs (int): sampling frequency of the signal.
        r_peaks (np.ndarray): r peaks in the signal
        window_start_ms (float): left bound for waveform extraction. 
        window_end_ms (float): right bound for waveform extraction.
    Returns:
        np.ndarray: feature matrix with all the waveforms in the ecg signal.
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

# --------------------------------- RR features -----------------------------------------------------------------------------------------------
def compute_pre_post_delta_rr(
    r_peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Caclulate duration before and after the beat and the acceleration of rhythm.
     Ectopic beats tend to have irregular coupling intervals.

    Args:
        r_peaks (np.ndarray): -

    Returns:
        tuple[np.ndarray,np.ndarray]: returns pre and post rr intervals along with rate of acceleration of rhythm.
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
    """
    Compute ratio of pre and post rr interval vectors.

    Args:
        pre_rr (np.ndarray): vector of pr_rr intervals for all beats.
        post_rr (np.ndarray): vector of pr_rr intervals for all beats.

    Returns:
        np.ndarray: Element-wise ratio of pre_rr to post_rr intervals.
    """

    return pre_rr / post_rr


def compute_local_rr_mean(
    r_peaks: np.ndarray, local_rr_mean_beat_window: int
) -> np.ndarray:
    """Compute the mean peak intervals in a specific beat window,
    This gives a gives a "what's normal for this patient right now" baseline.
    Deviations from local mean is very different for normal vs abnormal beats.

    Args:
        r_peaks (np.ndarray): -
        local_rr_mean_beat_window (int):Number of beats before and after the current beat for computing the mean.


    Returns:
        np.ndarray: local mean for each beat.
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
    Compute how much a single peak interval deviates from the peak interval mean across a number of beats.
    Deviations from local mean is very different for normal vs abnormal beats.

    Args:
        local_rr_mean (np.ndarray): Vector containing local rr mean for each beat.
        pre_rr (np.ndarray): Vector containing pr_rr intervals for all beats.

    Returns:
        np.ndarray: Ratio of local rr mean to pre rr interval
    """

    return np.array([pre_i / mean_i for pre_i, mean_i in zip(pre_rr, local_rr_mean)])

def compute_rr_irregularity(r_peaks:np.ndarray,rr_irregularity_window:int)->np.ndarray:
    """
    Compute irregularity in peak intervals using a given window.

    Args:
        r_peaks (np.ndarray):-
        rr_irregularity_window (int): N beat window before and after current peak for computing irregularity

    Returns:
        np.array: Irregularity for each beat.
    """
    peak_intervals = np.diff(r_peaks)
    rr_irreg = np.zeros((len(r_peaks),))

    for p_ind in range(len(r_peaks)):
        # decide left and right side of window based on peaks
        left = max(0, p_ind  - rr_irregularity_window)
        right = min(len(peak_intervals), p_ind + rr_irregularity_window)

        # find relevant peaks in the window and take mean/std
        relevant_peaks = peak_intervals[left:right]
        mean = np.mean(relevant_peaks )
        std  = np.std(relevant_peaks )
        # append value of irregularity 
        rr_irreg[p_ind] = std / mean if mean !=0 else 0

    return np.array(rr_irreg)


def extract_all_rr_features(
    r_peaks: np.ndarray, local_rr_mean_beat_window: int, rr_irregularity_window:int
) -> dict:
    """
    Extracts all the peak interval features in one place.

    Args:
        r_peaks (np.ndarray): -
        local_rr_mean_beat_window (int): Number of beats before and after the current beat for computing the mean.
        rr_irregularity_window (int):  N beat window before and after current peak for computing irregularity.

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

    # get rr irregularity
    rr_irreg = compute_rr_irregularity(r_peaks=r_peaks, rr_irregularity_window=rr_irregularity_window)

    return {
    "pre_rr": pre_rr,
    "post_rr": post_rr,
    "delta_rr": delta_rr,
    "pre_post_rr_ratio": pre_post_rr_ratio,
    "local_rr_mean": local_rr_mean,
    "deviation_rr": deviation_rr,
    "rr_irreg": rr_irreg
    }

# --------------------------------- QRS features -----------------------------------------------------------------------------------------------
def extract_all_qrs_features(X_wfms:np.ndarray, window_start_ms:float, qrs_extraction_window:list[int], fs:int)->dict:
    """
    Extracts qrs features. Computes the following:
    1. q-r interval
    2. r-s interval
    3. q-r peaks amplitude difference
    4. r-s peaks amplitude difference
    Args:
        X_wfms (np.ndarray): Feature matrix containing waveforms of each beat.
        qrs_extraction_window (list):Bounds for extracting q and s peaks (ms)
        window_start_ms (float): left bound for waveform extraction. Used along with qrs extraction window to get Q and S peaks.

        fs (int): Sampling rate of the ecg signal

    Returns:
        dict: Dict with all the features.
    """
    
    # convert window in ms to samples
    q_window = int((qrs_extraction_window[0] * fs)/ 1000)
    s_window = int((qrs_extraction_window[1] * fs)/ 1000)
    wfm_window_start = int((window_start_ms * fs) / 1000)

    assert (s_window - q_window) < X_wfms.shape[1], 'Extraction window longer than waveform limits'
    
    # define all output vectors
    q_r_intervals = np.full((X_wfms.shape[0],),np.nan) # intervals between q and r onsets
    r_s_intervals = np.full((X_wfms.shape[0],),np.nan)  # intervals between r and s onsets

    q_r_amps = np.full((X_wfms.shape[0],),np.nan) # voltage difference between q and r onsets
    r_s_amps = np.full((X_wfms.shape[0],),np.nan)  # voltage difference between r and s onsets


    # loop through all waves
    r_peak_ind = wfm_window_start 
    for wfm_ind in range(X_wfms.shape[0]):
        
        # get waveform and peak index
        wfm = X_wfms[wfm_ind,:]

        # calculate offsets for indices as np.argmin gives a relative index, make sure they are within bounds of wfm
        q_start = max(0, r_peak_ind + q_window)
        s_end = min(len(wfm), r_peak_ind + s_window)

        q_timestamp = int(q_start + np.argmin(wfm[q_start: r_peak_ind]))
        s_timestamp = int(r_peak_ind + np.argmin(wfm[r_peak_ind: s_end]))

        # get time interval info between q r and s
        q_r_intervals[wfm_ind] = abs(r_peak_ind - q_timestamp)
        r_s_intervals[wfm_ind] =  abs(r_peak_ind - s_timestamp)  

        # get amplitude difference info for each event
        q_r_amps[wfm_ind] = abs(wfm[q_timestamp] - wfm[r_peak_ind])
        r_s_amps[wfm_ind] = abs(wfm[r_peak_ind] - wfm[s_timestamp] )
        
    return {'q_r_intervals': q_r_intervals,
            'r_s_intervals': r_s_intervals,
            'q_r_amps': q_r_amps,
            'r_s_amps': r_s_amps

    }


# --------------------------------- wavelet features -----------------------------------------------------------------------------------------------
def extract_wavelet_features(X_waveforms:np.ndarray,wavelet_decomp_level:int=4)->np.ndarray:
    """
    Extracts fast freq components using wavelet decomposition/ transform.
    Args:
        X_waveforms (np.ndarray): Feature matrix containing waveforms of all beats
        wavelet_decomp_level (int): Chosen level for wavelet transformation.
    Returns:
        np.ndarray: wavelet coefficients of waveforms.
    """
    
    wavelet_coeff = []
    for wfm_ind in range(X_waveforms.shape[0]):
        coeffs = pywt.wavedec(data=X_waveforms[wfm_ind,:], wavelet='db2',level=wavelet_decomp_level)  # get amplitudes for fast and slow freq components for each wave
        wavelet_coeff.append(np.concatenate(coeffs[1:])) # append only the fast freq component/ amps at each level

    return np.array(wavelet_coeff)

# --------------------------------- combine features -----------------------------------------------------------------------------------------------
def return_commbined_feature_matrix(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    window_start_ms: float,
    window_end_ms: float,
    local_rr_beat_window: int=5,
    qrs_extraction_window: list[int] | None = None,
    rr_irregularity_window: int = 64,
    wavelet_decomp_level:int =4,
    compute_only: list[str] | None = None
) -> tuple[np.ndarray | None, ...]:
    """Combines all extracted features and returns them as output.

    Args:
        ecg_signal (np.ndarray): -_
        r_peaks (np.ndarray): -
        fs (int): sampling freq of the ecg signal
        window_start_ms (float): Left bound for waveform extraction. Also used along qrs_extraction window.
        window_end_ms (float): Right bound for waveform extraction.
        qrs_extraction_window (list):Bounds for extracting q and s peaks (ms)
        rr_irregularity_window (int): N beat window before and after current peak for computing irregularity.
        wavelet_decomp_level (int): Chosen level for wavelet transformation.
        compute_only (list): list of features for the function to compute.



    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: Feature matrix containing features of choice
    """
    if compute_only is None:
        compute_only = ["waves","interval_related","QRS"] # set of features to be computed if no list is given

    
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
        rr_dict = extract_all_rr_features(r_peaks=r_peaks,local_rr_mean_beat_window=local_rr_beat_window,rr_irregularity_window=rr_irregularity_window)
        X_rr = np.column_stack(list(rr_dict.values()))
    else:
        X_rr = None

    if "QRS" in compute_only:
        assert X_waves is not None, 'X_waves is None so QRS features cannot be computed. Change features in config to include waves and try again'
        assert qrs_extraction_window is not None, 'qrs_extraction_window must be provided when computing QRS features'
        qrs_dict = extract_all_qrs_features(X_wfms=X_waves,window_start_ms=window_start_ms, qrs_extraction_window=qrs_extraction_window, fs=fs)
        X_qrs = np.column_stack(list(qrs_dict.values()))
    else:
        X_qrs = None

    if 'wavelet' in compute_only:
        assert X_waves is not None, 'X_waves is None so wavelet features cannot be computed. Change features in config to include waves and try again' 
        X_wavelet = extract_wavelet_features(X_waves,wavelet_decomp_level)
    else:
        X_wavelet = None
    

    return X_waves, X_rr, X_qrs, X_wavelet

if __name__ == '__main__':
    pass