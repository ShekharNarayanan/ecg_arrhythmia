from matplotlib import pyplot as plt
import numpy as np


def plot_ecg_segment(signal: np.ndarray,fs: int,
    start_s: float = 0.0,
    duration_s: float = 5.0,
    r_peaks: np.ndarray | None = None,
    show_annotations: bool = False,
    label: str = "ECG",
) :
    """_summary_

    Args:
        signal (np.ndarray): _description_
        fs (int): _description_
        start_s (float, optional): _description_. Defaults to 0.0.
        duration_s (float, optional): _description_. Defaults to 5.0.
        r_peaks (np.ndarray | None, optional): _description_. Defaults to None.
        show_annotations (bool, optional): _description_. Defaults to False.
        label (str, optional): _description_. Defaults to "ECG".
    """

    assert signal.ndim == 1, "Signal must be 1D"

    start = int(start_s * fs)
    end = int((start_s + duration_s) * fs)

    segment = signal[start:end]
    t = np.arange(segment.size) / fs

    plt.plot(t, segment, label=label)

    if show_annotations and r_peaks is not None:
        local_peaks = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
        plt.plot(t[local_peaks], segment[local_peaks], marker = 'o', linestyle = 'None', color='red', alpha=0.3)
