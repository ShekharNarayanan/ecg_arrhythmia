from matplotlib import pyplot as plt
import numpy as np
import neurokit2 as nk

def plot_ecg_segment(signal: np.ndarray,fs: int,
    start_s: float = 0.0,
    duration_s: float = 5.0,
    r_peaks: np.ndarray | None = None,
    show_annotations: bool = False,
    label: str = "ECG",
):

    assert signal.ndim == 1, "Signal must be 1D"

    start = int(start_s * fs)
    end = int((start_s + duration_s) * fs)

    segment = signal[start:end]
    t = np.arange(segment.size) / fs

    plt.plot(t, segment, label=label)

    if show_annotations and r_peaks is not None:
        local_peaks = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
        plt.plot(t[local_peaks], segment[local_peaks], "rx")
