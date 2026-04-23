from arrhythmia_ml import preprocess, file_utils, extract_features
import numpy as np
import matplotlib
matplotlib.use("Agg") # no gui backend
from matplotlib import pyplot as plt
import yaml
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    # ----------------------------------------------------- load config params --------------------------------------------------------------------------
    with open(Path.cwd() / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    raw_data_path = config["raw_data_path"]

    # params
    bandpass_window = config["general_bandpass"]
    wave_extraction_window = config["wfm_extraction_window"]
    local_rr_mean_beat_window = config["local_rr_mean_beat_window"]
    qrs_extraction_window = config["qrs_extraction_window"]
    rr_irregularity_window = config["rr_irregularity_window"]
    keep_labels = np.array(config["keep_labels"])

    # get participant files
    participant_ids = file_utils.get_participant_ids(raw_data_path=raw_data_path)

    # select participant
    pid = 0
    participant = participant_ids[pid]


    # ----------------------------- helpers --------------------------------------------------------------------------------------------------------------------
    def annotate_counts(ax, labels, keep_labels):
        for i, beat in enumerate(keep_labels):
            count = np.sum(labels == beat)
            ax.annotate(
                f"n={count}",
                xy=(i, 0),
                xycoords=("data", "axes fraction"),
                ha="center",
                fontsize=8,
                color="gray",
            )


    # --------------------------------------------- features -----------------------------------------------------------------------------------------------------
    # get waveforms for participant
    raw_signal, fs, channels, r_peaks, beat_labels = file_utils.load_raw_participant_data(
        raw_data_path=raw_data_path, participant_id=participant
    )

    # decide chan to plot and filter labels
    chan = 0
    mask = np.isin(beat_labels, keep_labels)
    beat_labels = np.array(beat_labels)
    labels = beat_labels[mask]

    # processed waveforms
    processed_sig = preprocess.get_preprocessed_signal(
        raw_sig=raw_signal[:, chan], fs=fs, bandpass_window=bandpass_window
    )

    # wfms
    wfms = extract_features.extract_waveforms(
        ecg_signal=processed_sig,
        fs=fs,
        r_peaks=r_peaks,
        window_start_ms=wave_extraction_window[0],
        window_end_ms=wave_extraction_window[1],
    )
    rr_features_dict = extract_features.extract_all_rr_features(
        r_peaks=r_peaks,
        local_rr_mean_beat_window=local_rr_mean_beat_window,
        rr_irregularity_window=rr_irregularity_window,
    )

    wfms = wfms[mask]
    rr_features = {k: v[mask] for k, v in rr_features_dict.items()}


    colors = {"N": "steelblue", "A": "tomato", "V": "seagreen"}
    beat_colors = np.array([colors[l] for l in labels])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Participant {participant} — Feature Visuals", fontsize=14)

    # 1. mean waveform per beat type
    ax = axes[0, 0]
    x = np.arange(wfms.shape[1]) / fs * 1000
    for beat in keep_labels:
        mean_wfm = wfms[labels == beat].mean(axis=0)
        ax.plot(x, mean_wfm, label=beat, color=colors[beat])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("Mean waveform by beat type")
    annotate_counts(ax, labels, keep_labels)
    ax.legend()

    # 2. pre vs post RR colored by beat type
    ax = axes[0, 1]
    for beat in keep_labels:
        m = labels == beat
        ax.scatter(
            rr_features["pre_rr"][m] / fs * 1000,
            rr_features["post_rr"][m] / fs * 1000,
            label=f"{beat}:  count - {np.sum(m)}",
            color=colors[beat],
            alpha=0.4,
            s=10,
        )
    ax.set_xlabel("Pre RR (ms)")
    ax.set_ylabel("Post RR (ms)")
    ax.legend()

    # 3. RR irregularity distribution by beat type // violin
    ax = axes[1, 0]
    data_by_beat = [rr_features["rr_irreg"][labels == beat] for beat in keep_labels]
    ax.violinplot(data_by_beat, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{beat}\n(n={np.sum(labels == beat)})" for beat in keep_labels])
    ax.set_ylabel("RR irregularity (CV)")
    annotate_counts(ax, labels, keep_labels)
    ax.set_title("RR irregularity by beat type")

    # 4. QRS features scatter // violin
    qrs_dict = extract_features.extract_all_qrs_features(
        X_wfms=wfms,
        window_start_ms=wave_extraction_window[0],
        qrs_extraction_window=qrs_extraction_window,
        fs=fs,
    )
    ax = axes[1, 1]
    qrs_width = (qrs_dict["q_r_intervals"] + qrs_dict["r_s_intervals"]) / fs * 1000
    data_by_beat = [qrs_width[labels == beat] for beat in keep_labels]
    ax.violinplot(data_by_beat, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{beat}\n(n={np.sum(labels == beat)})" for beat in keep_labels])
    ax.set_ylabel("QRS width (ms)")
    ax.set_title("QRS width by beat type")
    plt.savefig("feature_plots/qrs_width_beat_type.png")

    # ── figure 2: local RR mean over time with beat type markers ──────────────────
    fig2, ax = plt.subplots(figsize=(18, 4))
    time_axis = time_axis = np.arange(len(labels)) / fs

    ax.plot(
        time_axis, rr_features["local_rr_mean"] / fs * 1000, color="lightgray", zorder=1
    )
    for beat in keep_labels:
        m = labels == beat
        ax.scatter(
            time_axis[m],
            rr_features["local_rr_mean"][m] / fs * 1000,
            label=beat,
            color=colors[beat],
            s=15,
            zorder=2,
            alpha=0.7,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Local RR mean (ms)")
    ax.set_title(f"Participant {participant} — Local RR mean over time")
    ax.legend()
    plt.tight_layout()
    plt.savefig("feature_plots/local_rr_mean_vs_time.png")

    # ── figure 3: feature correlation heatmap ─────────────────────────────────────
    # get wavelet features to complete the picture


    feature_matrix = np.column_stack(
        [
            rr_features["pre_rr"] / fs * 1000,
            rr_features["post_rr"] / fs * 1000,
            rr_features["delta_rr"] / fs * 1000,
            rr_features["pre_post_rr_ratio"],
            rr_features["local_rr_mean"] / fs * 1000,
            rr_features["deviation_rr"],
            rr_features["rr_irreg"],
            qrs_dict["q_r_intervals"] / fs * 1000,
            qrs_dict["r_s_intervals"] / fs * 1000,
            qrs_width,

        ]
    )

    feature_names = [
        "pre_rr",
        "post_rr",
        "delta_rr",
        "rr_ratio",
        "local_rr_mean",
        "deviation_rr",
        "rr_irreg",
        "q_r_int",
        "r_s_int",
        "qrs_width",
        
    ]
    corr = pd.DataFrame(feature_matrix, columns=feature_names).corr()

    fig3, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Participant {participant} — Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig("feature_plots/feature_correlation_heatmap.png")
