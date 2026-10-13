'''
version 1 of the automated eeg preprocessing pipeline specific to the calm-brain dataset. this pipeline is designed for resting
state data, with eyes open and eyes closed being the two prinicipal conditions, other annotations do exist in these files, and 
are handled in the code but aren't retained within the analysis. 

this pipeline performs the following steps:
1. reads in raw brainvision files
2. relabels and standardizes the annotations 
3. drops non-eeg channels and sets a montage
4. plots raw signals for qc purposes
5. filters the data - bandpass 1-100 Hz initially, and notches at the powerline and first harmonic
6. detects bad channels through kurtosis and median absolute deviation (MAD) based z-score thresholding
7. interpolates bad channels and sets average reference
8. runs one roun dof ica for artefact rejections, using mne-icalabel for componenets
9. plots cleaned signals for qc purposes
10. saves the cleaned fif file, and a json file with all parameters, bad channel and ica info

having performed these steps, the outputs are saved at the specified location on the server. these outputs nclude:
1. raw trace and psd files per subject (png)
2. filtered psd plots (png)
3. ica components and sources (png)
4. cleaned traces (png)
5. params, ica labels, excluded ics and bad channels (json, csv)

code written by vaaruni, with debugging assistance from claude.
'''
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.stats import kurtosis

import json
import traceback
from pathlib import Path
from datetime import datetime

import mne
import mne_icalabel
from almirah import Dataset

# specifying output paths. for calm-brain, the data has to be queried from the almirah dataset
# this happens later in the code
Output_path   = Path(r"/home/calm-brain/vsuresh/eeg_preprocessing/output_files")
Figures_path  = Output_path / "figures"
Log_path      = Output_path / "pipeline_log.csv"
Params        = Output_path / "subject_params"

Output_path.mkdir(parents=True, exist_ok=True)
Params.mkdir(parents=True, exist_ok=True)
Figures_path.mkdir(parents=True, exist_ok=True)

powerline_freq = 50 # 50 for EU and India, 60 for the US and the Americas 
h_freq         = 40 # this pipeline is currently designed for resting state EEG, 40 is a reasonable upper limit
l_freq         = 1
resamp_freq    = 250 # sfreq = 1000, nyquist at 125 Hz

ica_methods        = "infomax"
ica_random_states  = 97 

artefact_labels    = ["eye blink", "heart beat", "muscle artifact", "channel noise", "line noise"] # standard mne ica labels
artefact_threshold = 0.7 # can be modified to suit ICA rejection needs

# specified for the calm-brain dataset 
artefact_codes = {
    "9999": "EYE_OPEN",
    "8888": "EYE_CLOSE",
    "6666": "BAD_eye_movement",
    "0000": "BAD_movement",
    "4444": "BAD_eye_blink",
    "7777": "BAD_chew",
    "3333": "BAD_drowsy",
    "0001": "BAD_muscle",
    "0002": "BAD_sweat",
}

# more recent versions of this pipeline ignores these codes. for resting state data, it is assumed that EO is at the
# beginning of the recording, and EC is at the halfway point. other codes considered irrelevant
brainvision_stim_codes = {
    "Stimulus/S  1": "Cond1",  
    "Stimulus/S  2": "Cond2",
    "Stimulus/S  3": "Cond3",
    "Stimulus/S  4": "Cond4",
}

legacy_2017_label_codes = {
    # codes from 2017, have to check if they appear in some subject files
    "EYO_": "9999",
    "eyeo": "9999",
    "EVC_": "8888",
    "eyec": "8888",
    "EYMV": "6666",
    "eyem": "6666",
    "MVT_": "0000",
    "eyeb": "4444",
}

def find_subjects():
    dataset = Dataset.get(name="calm-brain") # dataset query as per almirah specifications
    layout  = dataset.components[0]

    files    = layout.query(datatype="eeg", task="rest", extension=".vhdr")
    subjects = []

    for f in files: # raw files within dataset stored per this naming convention
        subject_id = Path(f.path).parents[2].name + "_" + Path(f.path).parents[1].name
        subjects.append({"id": subject_id, "eeg_file": Path(f.path)})
        print(f"Subject Information: {subject_id} - {f.path}")

    print(f"Discovered {len(subjects)} subjects.")
    return subjects

def relabel_annotations(raw, subject_id): # relabels the annotations in the raw data to standardize them for further processing
    original = list(raw.annotations.description)
    new_desc = []
    unmatched = set()

    for desc in original:
        matched = None

        for code, label in artefact_codes.items():
            if code in desc:
                matched = label
                break

        if matched is None:
            for token, code in legacy_2017_label_codes.items():
                if token in desc:
                    matched = artefact_codes[code]
                    break

        if matched is None:
            for code, label in brainvision_stim_codes.items():
                if code in desc:
                    matched = label
                    break

        if matched is None:
            for label in artefact_labels:
                if label in desc:
                    matched = desc
                    break

        if matched:
            new_desc.append(matched)

    raw.annotations.description = np.array(new_desc)

    if unmatched:
        print(f"  Unmapped annotation codes for {subject_id}: {unmatched}")

    return raw

# this code includes various qc steps that genrate figures for visual inspection
def plot_raw_signal(raw, subject_id):
    matplotlib.use('Agg')
    fig = raw.compute_psd(fmax=125).plot(show=False)
    fig.savefig(Figures_path / f"{subject_id}_raw_psd.png", dpi=150, bbox_inches='tight')
    plt.close(fig) # raw psd plot

    fig2 = raw.plot(n_channels=min(32, len(raw.ch_names)), show=False)
    fig2.savefig(Figures_path / f"{subject_id}_raw.png", dpi=150, bbox_inches='tight')
    plt.close(fig2) # raw signal plot

def filter_raw(raw, subject_id):
    raw.filter(l_freq=l_freq, h_freq=100, fir_design='firwin', verbose=False) # need h_freq = 100 for ICA (mne-icalabel requirement)
    raw.notch_filter(freqs=[powerline_freq, powerline_freq * 2], # notch at powerline and first harmonic
                     fir_design='firwin', verbose=False)
    
    # why is this retaining large low power signals? (low power perhaps negligible)
    fig = raw.compute_psd(fmax=125).plot(show=False) 
    fig.axes[0].set_ylim(bottom = -100) # exclude low power drift
    fig.savefig(Figures_path / f"{subject_id}_filtered_psd.png", dpi=150, bbox_inches='tight')
    plt.close(fig) # filtered psd plot (artefacts retained)

    return raw

def drop_and_montage(raw):
    import re
    non_eeg = [ch for ch in raw.ch_names if not re.match(r'^E\d+$', ch)]
    raw.drop_channels([ch for ch in non_eeg if ch in raw.ch_names])
    raw.set_montage('GSN-HydroCel-128') # default 128 channel montage, can be changed if required
    return raw

def detect_bad_channels(raw):
    bad_channels = _statistical_bad_channels(raw)
    return bad_channels

# bad channel detection through kurtosis and median absolute deviation (MAD) based z-score thresholding. 
# channels with low MAD or high kurtosis are flagged as bad
def _statistical_bad_channels(raw, mad_z_threshold=3.0, kurt_z_threshold=5.0):
    data = raw.get_data(reject_by_annotation='omit') # included rejection by annotation but not all artefacts annotated in these files

    if data.shape[1] == 0:     # fall back to all data if annotations cover the entire recording
        data = raw.get_data()

    channel_names = raw.ch_names
    median = np.median(data, axis=1, keepdims=True)
    mads = np.median(np.abs(data - median), axis=1)

    kurt_vals = np.nan_to_num(
        kurtosis(data, axis=1, fisher=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    def robust_z_score(values):
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        return (values - med) / (1.4826 * mad if mad > 0 else 1e-6)

    z_mads = robust_z_score(mads)
    z_kurt = robust_z_score(kurt_vals)

    bad_indices = np.where(
        (mads < 1e-10) |
        (np.abs(z_mads) > mad_z_threshold) |
        (z_kurt > kurt_z_threshold)
    )[0]

    bad_channels = [channel_names[i] for i in bad_indices]
    print(f"  Bad channels: {bad_channels}")
    return bad_channels

def interpolate_and_reference(raw, bad_channels):
    bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]

    raw.info['bads'] = bad_channels
    if bad_channels:
        raw.interpolate_bads(reset_bads=True)

    # can replace avergae with vref but looks like vref has zero values, throwing error
    raw.set_eeg_reference('average', projection=False, verbose=False)

    if resamp_freq and raw.info["sfreq"] > resamp_freq:
        raw.resample(resamp_freq, verbose=False)

    return raw

def _save_figs(figs, stem, dpi=150):
    if not isinstance(figs, list):
        figs = [figs]
    for i, f in enumerate(figs):
        f.savefig(Figures_path / f"{stem}_{i}.png", dpi=dpi)
        plt.close(f)

def run_ica(raw, subject_id, n_bad_channels=0):
    ica_n_components = min(40, len(raw.ch_names) - n_bad_channels - 1) # minimum 40 components; ideal: number of channels - number of bad channels - 1
    ica = mne.preprocessing.ICA(
        n_components=ica_n_components,
        method=ica_methods,
        random_state=ica_random_states,
        fit_params=dict(extended=True),
        verbose=False
    )
    ica.fit(raw, reject_by_annotation=True, verbose=False)

    _save_figs(ica.plot_components(show=False), f"{subject_id}_components.png")
    _save_figs(ica.plot_sources(raw, start=0, stop=30, show=False), f"{subject_id}_sources.png")

    ic_labels = mne_icalabel.label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    exclude = [
        idx for idx, (label, prob_vector) in enumerate(zip(labels, probs))
        if label in artefact_labels and prob_vector.max() >= artefact_threshold
    ]

    ica.exclude = exclude
    ica.apply(raw, verbose=False)
    return raw, ica, exclude, labels

def postprocess(raw, subject_id):
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False) # back to 40 for resting state
    fig2 = raw.plot(n_channels=min(32, len(raw.ch_names)), show=False)
    fig2.savefig(Figures_path / f"{subject_id}_clean.png", dpi=150)
    plt.close(fig2)
    return raw

def save_outputs(raw, subject_id, bad_channels, exclude_ica, labels):
    out_path = Output_path / f"{subject_id}_preprocessed.fif" # this throws a warning; brainvision file naming conventions specify that 
    # fif files must be saved with _raw.fif or _eeg.fif. can be changed if required.
    raw.save(out_path, overwrite=True, verbose=False)

    params = {
        "subject_id":   subject_id,
        "bad_channels": bad_channels,
        "ica_excluded": exclude_ica,
        "ica_labels":   labels,
        "preprocessing": {
            "l_freq": l_freq, "h_freq": h_freq, "h_freq_ica": 100,
            "resamp_freq": resamp_freq, "notch": powerline_freq
        }
    }
    params_path = Params / f"{subject_id}_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2) # json file with all params, bad channel and ica info

    print(f"  Saved: {out_path.name}")

def log_result(subject_id, status, bad_channels=None, exclude_ica=None, error=None):
    record = {
        "timestamp":    datetime.now().isoformat(),
        "subject_id":   subject_id,
        "status":       status,
        "bad_channels": str(bad_channels) if bad_channels else "",
        "ica_excluded": str(exclude_ica)  if exclude_ica  else "",
        "error":        error or ""
    }
    df = pd.DataFrame([record])

    if Log_path.exists():
        df.to_csv(Log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(Log_path, index=False) # csv log file with subject id, status, bad channels and ica info

def run_pipeline():
    subjects = find_subjects()

    total     = len(subjects)
    skipped   = 0
    succeeded = 0
    failed    = 0

    for i, subject in enumerate(subjects, 1):
        subject_id = subject["id"]
        print(f"\n[{i}/{total}] {subject_id}")

        try:
            raw = mne.io.read_raw_brainvision(
                subject["eeg_file"], preload=True, verbose=False
            )
            raw.resample(resamp_freq, verbose=False)

            raw = relabel_annotations(raw, subject_id)

            raw = drop_and_montage(raw)

            plot_raw_signal(raw, subject_id)

            raw = filter_raw(raw, subject_id)

            bad_channels = detect_bad_channels(raw)

            raw = interpolate_and_reference(raw, bad_channels)

            raw, ica, exclude_ica, labels = run_ica(raw, subject_id, n_bad_channels=len(bad_channels))

            raw = postprocess(raw, subject_id)

            save_outputs(raw, subject_id, bad_channels, exclude_ica, labels)

            log_result(subject_id, "success", bad_channels, exclude_ica)
            succeeded += 1

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"  ERROR: {e}")
            log_result(subject_id, "failed", error=error_msg)
            failed += 1

    print(f"\n{'='*50}")
    print(f"DONE - {total} subjects total")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Log: {Log_path}")


if __name__ == "__main__":
    run_pipeline()