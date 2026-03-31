#  *- plot utils-*
from matplotlib import pyplot as plt
import numpy as np


def plot_ecg_segment(signal: np.ndarray,fs: int,
    start_s: float = 0.0,
    duration_s: float = 5.0,
    r_peaks: np.ndarray | None = None,
    show_annotations: bool = False,
    label: str = "ECG",
) :
    """Plot a segment of a 1D ECG signal.

    Args:
        signal (np.ndarray): 1D ECG signal.
        fs (int): Sampling frequency.
        start_s (float, optional): Start time in seconds. Defaults to 0.0.
        duration_s (float, optional): Duration of segment to plot in seconds. Defaults to 5.0.
        r_peaks (np.ndarray | None, optional): R-peak indices to overlay on the plot. Defaults to None.
        show_annotations (bool, optional): Whether to plot R-peak markers. Defaults to False.
        label (str, optional): Legend label for the signal. Defaults to "ECG".
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



def plot_qrs_complex(X_waveforms:np.ndarray, fs:int, pid:int=0):
    """Plot a single QRS complex waveform for a given participant index.

    Args:
        X_waveforms (np.ndarray): Feature matrix containing waveforms of all beats.
        fs (int): Sampling frequency.
        pid (int, optional): Beat index to plot. Defaults to 0.
    """

    t = np.arange(X_waveforms.shape[1]) / fs
    complex = X_waveforms[pid,:]

    plt.plot(t,complex)
    plt.title(f"QRS Complex Participant {pid}")
    plt.xlabel("time (ms)")
    plt.ylabel("mV")