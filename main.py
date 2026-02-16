def main():
    from arrhythmia_ml import file_utils, preprocess, plots
    from matplotlib import pyplot as plt
  

    # LOAD PARTICIPANT DATA
    config = file_utils.load_config()
    raw_data_path = config["raw_data_path"]
    participant_ids = file_utils.get_participant_ids(raw_data_path=raw_data_path)
    test_id = participant_ids[0]

    # LOAD RAW ECG DATA
    signal, fs, channels, r_peaks, labels = file_utils.load_raw_participant_data(raw_data_path=raw_data_path, participant_id=test_id)

    # DISPLAY STATS
    print(f"Participant ID: {test_id}")
    print(f"Signal shape: {signal.shape}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Channels: {channels}")
    print(f"Number of beats: {len(r_peaks)}")
    print(f"Beat types: {set(labels)}")

    # DECIDE WHICH CHANNEL TO PLOT
    chan_to_plot = 0  # MLII lead    
    raw_sig = signal[:, chan_to_plot]

    # BANDPASS AND NOTCH FILTERING
    bpf_signal = preprocess.bandpass_1d(raw_sig, fs=fs, low=0.5, high=30.0)
    bpf_notch_signal = preprocess.notch_filter_1d(bpf_signal, fs=fs, freq=30.0)

    # PLOT RAW AND FILTERED SIGNALS
    plt.figure(figsize=(12, 6))
    plots.plot_ecg_segment(signal=raw_sig, fs=fs, start_s=0, duration_s=5, r_peaks=r_peaks, label=f"{channels[chan_to_plot]}_raw")
    plots.plot_ecg_segment(signal=bpf_notch_signal, fs=fs, start_s=0, duration_s=5, r_peaks=r_peaks, label=f"{channels[chan_to_plot]}_filtered")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("ECG 0–5s")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
