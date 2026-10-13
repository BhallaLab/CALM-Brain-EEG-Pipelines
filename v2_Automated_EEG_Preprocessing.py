'''
version 2 of the automated eeg preprocessing pipeline specific to the calm-brain dataset. this pipeline is designed for resting
state data, with eyes open and eyes closed being the two prinicipal conditions, other annotations do exist in these files, and 
are handled in the code but aren't retained within the analysis. this steps of this process were provided by dr gautam from the
NIMHANS team.

this pipeline diverges from the previous version in the following ways:
1. drops 10 known bad channels that commonly have occular, ecg or muscle artefacts.
2. drops identified bad channels instead of interpolating
3. notes channel impedences and creates a csv of these values
4. applies two passes of ica, the first with a low threshold and the second with a high one

having performed these steps, the outputs are saved at the specified location on the server. these outputs nclude:
1. raw trace and psd files per subject (png)
2. filtered psd plots (png)
3. ica components and sources (png)
4. cleaned traces (png)
5. params, ica labels, excluded ics and bad channels (json, csv)

code written by vaaruni, with debugging assistance from claude.
'''
import json
import pickle
import re
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne
from almirah import Dataset
from mne.preprocessing import ICA
from mne.preprocessing import annotate_amplitude
from mne_icalabel import label_components
from scipy.stats import kurtosis

mne.set_log_level("WARNING")

Output_path = Path("/home/calm-brain/vsuresh/eeg_preprocessing/rep2")
Params = Output_path / "subject_params"
Figures_path = Output_path / "figures"
Log_path = Output_path / "pipeline_log.csv"
Impedance_csv = Output_path / "impedances.csv"

for p in (Output_path, Params, Figures_path):
    p.mkdir(parents=True, exist_ok=True)

powerline_freq = 50
h_freq = 40
h_freq_ica = 100
l_freq = 1.0
resamp_freq = 250

ica_method = "infomax"
ica_random_state = 22
ica_max_components = 40

# each round fit on the output of the previous one.
ica_passes = [
    {"max_iter": 50, "threshold": 0.60},
    {"max_iter": 50, "threshold": 0.99},
]

artefact_labels = {"eye blink", "heart beat", "muscle artifact", "channel noise", "line noise"}

IMPEDANCE_THRESHOLD_KOHM = 50.0 # threshold for impedence of each channel

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

brainvision_stim_codes = {
    "Stimulus/S 1": "Cond1",
    "Stimulus/S 2": "Cond2",
    "Stimulus/S 3": "Cond3",
    "Stimulus/S 4": "Cond4",
}

legacy_2017_label_codes = {
    "EYO_": "9999", "eyeo": "9999",
    "EVC_": "8888", "eyec": "8888",
    "EYMV": "6666", "eyem": "6666",
    "MVT_": "0000",
    "eyeb": "4444",
}

ignore_labels = {"Impedances End", "Impedances Begin", "IEND", "IBEG", "New Segment/", ""}
known_bads = {"E1", "E8", "E14", "E21", "E25", "E32", "E125", "E128"}
emg_channels = {"E127", "E126"}

# True  -> known-bad channels kept
# False -> known-bad channels dropped
interpolate_known_bads = False

_EEG_RE = re.compile(r"^E\d+$")


def find_subjects():
    dataset = Dataset.get(name="calm-brain")
    layout = dataset.components[0]

    files = layout.query(datatype="eeg", task="rest", extension=".vhdr")
    subjects = []

    for f in files:
        subject_id = Path(f.path).parents[2].name + "_" + Path(f.path).parents[1].name
        subjects.append({"id": subject_id, "eeg_file": Path(f.path)})
    return subjects


def relabel_annotations(raw, subject_id):
    new_desc, unmatched = [], set()

    for desc in raw.annotations.description:
        norm = re.sub(r"\s+", " ", desc).strip()
        matched = None

        if matched is None:
            for token, code in legacy_2017_label_codes.items():
                if token in norm:
                    matched = artefact_codes[code]
                    break

        if matched is None:
            for key, label in brainvision_stim_codes.items():
                if key in norm:
                    matched = label
                    break

        if matched is None:
            matched = desc
            if desc not in ignore_labels:
                unmatched.add(desc)
        new_desc.append(matched)

    ann = raw.annotations
    raw.set_annotations(mne.Annotations(ann.onset, ann.duration, new_desc, orig_time=ann.orig_time))

    if unmatched:
        return raw


def drop_and_montage(raw):
    drop = [ch for ch in raw.ch_names
            if not _EEG_RE.match(ch)
            or ch in emg_channels
            or (not interpolate_known_bads and ch in known_bads)]
    raw.drop_channels(drop)
    raw.set_montage("GSN-HydroCel-128", verbose=False)

    # channels absent from the 128 montage
    no_pos = [ch for ch, c in zip(raw.ch_names, raw.info["chs"])
              if np.isnan(c["loc"][:3]).any() or not np.any(c["loc"][:3])]

    raw.info["bads"] = sorted(ch for ch in known_bads if ch in raw.ch_names) if interpolate_known_bads else []
    return raw


def detect_bad_channels(raw, mad_z_threshold=3.0, kurt_z_threshold=5.0):
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    names = [raw.ch_names[i] for i in picks]

    data = raw.get_data(picks=picks, reject_by_annotation="omit")
    if data.shape[1] == 0:
        data = raw.get_data(picks=picks)

    median = np.median(data, axis=1, keepdims=True)
    mads = np.median(np.abs(data - median), axis=1)
    kurt_vals = np.nan_to_num(kurtosis(data, axis=1, fisher=True), nan=0.0, posinf=0.0, neginf=0.0)

    def robust_z(values):
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        return (values - med) / (1.4826 * mad if mad > 0 else 1e-6)

    flagged = np.where((mads < 1e-10) | (np.abs(robust_z(mads)) > mad_z_threshold)
                       | (robust_z(kurt_vals) > kurt_z_threshold))[0]
    bad = [names[i] for i in flagged]
    return bad

def drop_and_reference(raw, statistical_bads):
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(statistical_bads))
    all_bads = list(raw.info["bads"])
    if all_bads:
        raw.drop_channels(all_bads)          # drop instead of interpolate
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw, all_bads

def save_figs(figs, stem):
    if not isinstance(figs, (list, tuple)):
        figs = [figs]
    for k, f in enumerate(figs):
        suffix = "" if len(figs) == 1 else f"_{k}"
        f.savefig(Figures_path / f"{stem}{suffix}.png")
        plt.close(f)

def plot_raw_signal(raw, subject_id):
    fig = raw.compute_psd(fmax=min(125, raw.info["sfreq"] / 2)).plot(show=False)
    save_figs(fig, f"{subject_id}_raw_psd")
    fig2 = raw.plot(n_channels=min(32, len(raw.ch_names)), show=False)
    save_figs(fig2, f"{subject_id}_raw")


def plot_filtered_psd(raw, subject_id):
    fig = raw.compute_psd(fmax=min(125, raw.info["sfreq"] / 2)).plot(show=False)
    fig.axes[0].set_ylim(bottom=-100)   
    save_figs(fig, f"{subject_id}_filtered_psd")


def plot_ica_qc(ica, raw, subject_id, pass_no):
    fig = ica.plot_components(show=False, title=f"{subject_id} pass {pass_no} (excluded: {ica.exclude})")
    save_figs(fig, f"{subject_id}_pass{pass_no}_components")

    fig2 = ica.plot_sources(raw, start=0, stop=30, show=False)
    save_figs(fig2, f"{subject_id}_pass{pass_no}_sources")


def filter_raw(raw, subject_id):
    raw.filter(l_freq=l_freq, h_freq=h_freq_ica, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[powerline_freq, powerline_freq * 2], fir_design="firwin", verbose=False)
    plot_filtered_psd(raw, subject_id)
    return raw


def run_ica_passes(raw, subject_id):
    icas, results = {}, []

    annots, amp_bads = annotate_amplitude(raw, peak=dict(eeg=300e-6),
                                          bad_percent=5, min_duration=0.002)
    raw.set_annotations(raw.annotations + annots)

    n_removed = 0   # components removed by earlier rounds
    for i, cfg in enumerate(ica_passes, 1):
        n_channels = len(mne.pick_types(raw.info, eeg=True, exclude="bads"))
        n_bads = len(raw.info["bads"])
        n_comp = n_channels - n_bads - n_removed 

        ica = ICA(
            n_components=n_comp,
            method=ica_method,
            fit_params=dict(extended=True),   
            max_iter=cfg["max_iter"],
            random_state=ica_random_state,
        )
        ica.fit(raw, reject_by_annotation=True, verbose=False)

        lab = label_components(raw, ica, method="iclabel")
        labels, probs = lab["labels"], lab["y_pred_proba"]

        exclude = [idx for idx, (lbl, p) in enumerate(zip(labels, probs))
                   if lbl in artefact_labels and p >= cfg["threshold"]]
        ica.exclude = exclude

        plot_ica_qc(ica, raw, subject_id, i)

        raw = ica.apply(raw, verbose=False)
        n_removed += len(exclude)
        icas[f"pass{i}"] = ica
        results.append({"pass": i, "excluded": exclude,
                        "labels": [labels[k] for k in exclude]})

    return raw, icas, results


def postprocess(raw, subject_id):
    raw.filter(l_freq=None, h_freq=h_freq, fir_design="firwin", verbose=False)

    fig = raw.plot(n_channels=min(32, len(raw.ch_names)), show=False)
    save_figs(fig, f"{subject_id}_clean")
    return raw


def save_outputs(raw, subject_id, bad_channels, icas, ica_results):
    out_path = Output_path / f"{subject_id}_preprocessed_raw.fif"
    raw.save(out_path, overwrite=True, verbose=False)

    params = {
        "subject_id": subject_id,
        "bad_channels": bad_channels,
        "ica_passes": ica_passes,
        "ica_results": ica_results,
    }
    with open(Params / f"{subject_id}_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"  Saved: {out_path.name}, {subject_id}_icas.pkl")


def log_result(subject_id, status, bad_channels=None, ica_results=None, error=None):
    excluded = {f"pass{r['pass']}": r["excluded"] for r in ica_results} if ica_results else ""
    record = {
        "timestamp": datetime.now().isoformat(),
        "subject_id": subject_id,
        "status": status,
        "bad_channels": str(bad_channels) if bad_channels else "",
        "ica_excluded": str(excluded),
        "error": error or "",
    }
    df = pd.DataFrame([record])
    df.to_csv(Log_path, mode="a" if Log_path.exists() else "w", header=not Log_path.exists(), index=False)


def process_subject(subject):
    subject_id = subject["id"]
    raw = mne.io.read_raw_brainvision(subject["eeg_file"], preload=True, verbose=False)
    raw.resample(resamp_freq, verbose=False)

    raw = relabel_annotations(raw, subject_id)
    raw = drop_and_montage(raw)
    plot_raw_signal(raw, subject_id)

    raw = filter_raw(raw, subject_id)
    statistical_bads = detect_bad_channels(raw)
    raw, all_bads = drop_and_reference(raw, statistical_bads)

    raw, icas, ica_results = run_ica_passes(raw, subject_id)
    raw = postprocess(raw, subject_id)

    save_outputs(raw, subject_id, all_bads, icas, ica_results)
    return all_bads, ica_results


def view_subject(subject_id):
    fif = Output_path / f"{subject_id}_preprocessed_raw.fif"
    if not fif.exists():
        raise SystemExit(f"No preprocessed file for {subject_id}: {fif}")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)

    pkl = Output_path / f"{subject_id}_icas.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            for name, ica in pickle.load(f).items():
                ica.plot_components(show=True, title=f"{subject_id} {name} (excluded: {ica.exclude})")

    raw.compute_psd(fmax=min(125, raw.info["sfreq"] / 2)).plot(show=True)
    raw.plot(duration=30, n_channels=40, scalings=dict(eeg=50e-6), block=False)
    plt.show()


def run_pipeline(impedance=True, preprocess=True, force=False, only=None):
    subjects = find_subjects()
    if only:
        subjects = [s for s in subjects if s["id"] in only]

    if not preprocess:
        return

    total, skipped, succeeded, failed = len(subjects), 0, 0, 0
    for i, subject in enumerate(subjects, 1):
        subject_id = subject["id"]
        print(f"\n[{i}/{total}] {subject_id}")

        try:
            bad_channels, ica_results = process_subject(subject)
            log_result(subject_id, "success", bad_channels, ica_results)
            succeeded += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            log_result(subject_id, "failed", error=traceback.format_exc())
            failed += 1
            plt.close("all")

    print(f"\n{'=' * 50}")
    print(f"DONE - {total} subjects total")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Log: {Log_path}")
    print(f"  Figures: {Figures_path}")


if __name__ == "__main__":
    run_pipeline()