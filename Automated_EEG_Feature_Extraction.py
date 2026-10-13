'''
version 1 of the feature extraction pipeline. this extracts all band powers across epochs. importantly, this code identifies 
four stim types within the recordings, labelled stim1-4, and segments each recording as per these annotations. band power 
dynamics are explored within each segment type. this was done prior to information regarding what these annotations meant, 
and thus each segment, though unique, was considered stimulus agnostic. the inputs for this pipeline are the cleaned fif 
files generated through the preprocessing pipeline.

the steps for feature extraction via band power are:
1. extracting stimulus onset times and thus determining segment length
2. relabelling these to be condition blocks
3. dividing each of these condition blocks into epochs based on expected dynamic timescales
4. transform each epoch into thr fourier space and select for each band frequency 
5. transform these back to the time domain

the outputs are thus:
1. band power snapshot (png)
2. band power params (csv)
3. stim onset times (csv)
4. pipeline log (csv)

many modifications are to be made to this code to ensure that it effectively captures all band power frequencies in a 
meaningful fashion; this is simply the first pass.

code written by vaaruni, with debugging assistance from claude.
'''
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import neurokit2 as nk

from pathlib import Path

Input_dir  = Path("/home/calm-brain/vsuresh/eeg_preprocessing/output_files") # input fif files
Output_dir = Path("/home/calm-brain/vsuresh/eeg_feature_extraction/output_files")
Figures_dir = Output_dir / "figures"
Features_dir = Output_dir / "features"
Log_path = Output_dir / "pipeline_log.csv"

Figures_dir.mkdir(parents=True, exist_ok=True)
Features_dir.mkdir(parents=True, exist_ok=True)

# band values from literature, no distinctions made between high and low freqs within the same band
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# later versions of this code ignore these stim event ids following newer information from NIMHANS
stim_event_id = {
    "Cond1": 1, "Cond2": 2, "Cond3": 3, "Cond4": 4,
    "Stimulus/S  5": 5, "Stimulus/S  6": 6, "Stimulus/S  7": 7,
}
stim_id_to_label = {i: f"S{i}" for i in range(1, 8)}

def extract_subject_id(fif_path):
    match = re.search(r"(sub-[^_]+_ses-[^_]+)", fif_path.name)
    return match.group(1) if match else fif_path.stem

def extract_stimulus_onsets(raw, subject_id):
    events, event_id = mne.events_from_annotations( # these annotations seem not to contain any useful info about bad channels
        raw, event_id=stim_event_id, verbose=False
    )
    if len(events) == 0:
        return pd.DataFrame(columns=["stimulus", "onset_sec"])

    sfreq = raw.info["sfreq"]
    rows = [
        {"stimulus": stim_id_to_label[code], "onset_sec": sample / sfreq}
        for sample, _, code in events
    ]
    return pd.DataFrame(rows)

def get_condition_blocks(raw, subject_id):
    events, event_id = mne.events_from_annotations(
        raw, event_id={"Cond1": 1, "Cond2": 2, "Cond3": 3, "Cond4": 4}, verbose=False
    )

    if len(events) == 0:
        return []

    id_to_label = {v: k for k, v in event_id.items()}
    sfreq = raw.info["sfreq"]

    blocks = []
    for i, (sample, _, code) in enumerate(events):
        onset_sec = sample / sfreq
        if i + 1 < len(events):
            offset_sec = events[i + 1][0] / sfreq
        else:
            offset_sec = raw.times[-1]
        blocks.append({
            "condition": id_to_label[code],
            "tmin": onset_sec,
            "tmax": offset_sec,
        })
    return blocks
 
def epoch_by_condition(raw, subject_id, duration=2, overlap=0): # depending on the research question, both duration and overlap will change
    # these numbers have been chosen based on the timescales i'd expect for short timescale resting state activity
    blocks = get_condition_blocks(raw, subject_id)
    if not blocks:
        return None, None, None
 
    all_epochs = []
    condition_labels = []
    epoch_times = []
 
    for block in blocks:
        if block["tmax"] - block["tmin"] < duration:
            continue
 
        raw_block = raw.copy().crop(tmin=block["tmin"], tmax=block["tmax"]) # each epoch will vary in time
        block_epochs = mne.make_fixed_length_epochs(
            raw_block, duration=duration, overlap=overlap,
            preload=True, reject_by_annotation=True
        )
        if len(block_epochs) == 0:
            continue
 
        all_epochs.append(block_epochs)
        condition_labels.extend([block["condition"]] * len(block_epochs))
        epoch_times.extend([
            block["tmin"] + (samp - raw_block.first_samp) / raw.info["sfreq"]
            for samp in block_epochs.events[:, 0]
        ])
 
    if not all_epochs:
        return None, None, None
 
    epochs = mne.concatenate_epochs(all_epochs)
    return epochs, condition_labels, epoch_times
 
def process_subject(fif_path):
    subject_id = extract_subject_id(fif_path)
    band_power_csv = Features_dir / f"{subject_id}_band_power.csv"
    stim_csv = Features_dir / f"{subject_id}_stimulus_onsets.csv"

    if band_power_csv.exists() and stim_csv.exists():
        print(f"Skipping {subject_id}, already processed")
        return
 
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    sfreq = raw.info['sfreq']

    stim_df = extract_stimulus_onsets(raw, subject_id)
    stim_df.to_csv(stim_csv, index=False)

    if band_power_csv.exists():
        print(f"Band power already done for {subject_id}, stim onsets updated only")
        return {"subject_id": subject_id, "status": "stim_onsets_only"}
 
    epochs, condition_labels, epoch_times = epoch_by_condition(raw, subject_id, duration=10, overlap=0)
 
    if epochs is None:
        print(f"WARNING: no valid EO/EC epochs for {subject_id}, skipping")
        return {"subject_id": subject_id, "status": "no_condition_epochs"}
 
    reject_criteria = dict(eeg=1000e-6) # is this okay
    epochs.drop_bad(reject=reject_criteria)
 
    kept = [i for i, log in enumerate(epochs.drop_log) if len(log) == 0]
    condition_labels = [condition_labels[i] for i in kept]
    epoch_times = [epoch_times[i] for i in kept]
 
    if len(epochs) == 0:
        print(f"WARNING: all epochs dropped for {subject_id}, skipping")
        return {"subject_id": subject_id, "status": "all_epochs_dropped"}
 
    assert len(condition_labels) == len(epochs), \
        f"condition_labels length {len(condition_labels)} != epochs length {len(epochs)} for {subject_id}"
 
    e = epochs.get_data()
    fft_epochs = np.fft.rfft(e, axis=-1)
    freqs = np.fft.rfftfreq(e.shape[-1], d=1/sfreq)
    power = np.abs(fft_epochs) ** 2
 
    band_power = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power[band_name] = power[:, :, idx].mean(axis=-1)
 
    epoch_features = pd.DataFrame({
        band: band_power[band].mean(axis=1) for band in bands
    })
    epoch_features.insert(0, "condition", condition_labels)
    epoch_features.insert(1, "onset_sec", epoch_times)
 
    epoch_features.to_csv(band_power_csv)
 
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    band_means = epoch_features[list(bands.keys())].mean()
    axes[0].bar(band_means.index, band_means.values, color='steelblue')
    axes[0].set_title("Mean Band Power (all epochs)")
    axes[0].set_ylabel("Power")
    for band in bands:
        axes[1].plot(epoch_times, epoch_features[band], label=band)
    stim_colors = {f"S{i}": c for i, c in zip(range(1, 8), plt.cm.tab10.colors)}
    seen_stim = set()
    for _, row in stim_df.iterrows():
        stim_type = row["stimulus"]
        axes[1].axvline(
            row["onset_sec"], color=stim_colors.get(stim_type, 'red'),
            linestyle='--', alpha=0.5,
            label=stim_type if stim_type not in seen_stim else None
        )
        seen_stim.add(stim_type)
    axes[1].set_title("Band Power Across Epochs")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Power")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(Figures_dir / f"{subject_id}_band_power.png")
    plt.close(fig)
 
    print(f"Done: {subject_id}")
    return {"subject_id": subject_id, "status": "success"}
 
 
def main():
    fif_files = sorted(Input_dir.glob("*.fif"))
    print(f"Found {len(fif_files)} files")
 
    log_rows = []
    for fif_path in fif_files:
        try:
            result = process_subject(fif_path)
            if result:
                log_rows.append(result)
        except Exception as e:
            print(f"FAILED: {fif_path.name} -> {e}")
            log_rows.append({"subject_id": fif_path.stem, "status": f"error: {e}"})
 
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if Log_path.exists():
            log_df = pd.concat([pd.read_csv(Log_path), log_df], ignore_index=True)
        log_df.to_csv(Log_path, index=False)
 
 
if __name__ == "__main__":
    main()