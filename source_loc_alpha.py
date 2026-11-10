'''
code for EEG-only source localiation and alpha power extraction, using MNE-Python, as a precursor to 
the full multimodal source loc approach using structural mri as a benchmark for deeper brain regions (differential weighting
of deep regions perhaps unreliable?) performs the following steps:
1. find all .fif files in the input directory (raw or epochs)
2. for each session, find the eyes-open and eyes-closed blocks (from annotations or fixed split)
3. create epochs from the blocks, reject bad epochs
4. compute the source power spectral density (PSD) for each condition and method    

tested against eLORETA, sLORETA and MNE dSPM, but only eLORETA is used in the final analysis because of best perofmance
metrics used for assessment:
1. berger effect as positive control (eyes-closed > eyes-open alpha power)
2. peak alpha source location in the parcellation (Desikan-Killiany) and MNI coordinates
3. relative alpha power per parcel (normalized to the within-condition mean)

used in complement with alpha validation code also available in this repo
'''
import json
import logging
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.minimum_norm import compute_source_psd_epochs, make_inverse_operator

INPUT_PATH = Path("/home/calm-brain/vsuresh/eeg_preprocessing/rep4")
OUTPUT_PATH = Path("/home/calm-brain/vsuresh/eeg_feature_extraction/source_localisation")
SRC_PATH = OUTPUT_PATH / "source_loc"
STC_DIR = SRC_PATH / "stc"
PLOT_DIR = SRC_PATH / "plots"
ERR_DIR = OUTPUT_PATH / "errors"
FWD_PATH = SRC_PATH / "fsaverage-oct6-eeg-fwd.fif"
OUT_CSV = OUTPUT_PATH / "source_alpha.csv"
PEAK_CSV = OUTPUT_PATH / "source_alpha_peaks.csv"
SESSIONS_CSV = OUTPUT_PATH / "source_sessions.csv"           # one row per session, failed ones included
PARCEL_TABLE_CSV = OUTPUT_PATH / "parcellation_table.csv"    # vertices per parcel in the source space
SPECTRA_CSV = OUTPUT_PATH / "source_spectra.csv"             # power per parcel x frequency, for the PSD figure

CONDITIONS = ("eyes-open", "eyes-closed")
SRC_METHODS = ("eLORETA",)
WORKERS = 2
ALPHA_BAND = (8.0, 13.0)
PSD_RANGE = (1.0, 30.0)         # spectrum saved per parcel; alpha power is the 8-13 Hz part of it
LAMBDA2 = 1.0 / 9.0
LOOSE = 0.2
DEPTH = 0.8
PSD_BANDWIDTH = "hann"          # single Hann window instead of multitaper
PARCELLATION = "aparc"          # Desikan-Killiany on fsaverage
LABEL_SKIP = ("unknown", "???", "medial_wall", "background", "corpuscallosum")
MIN_PARCEL_VERTICES = 3         # smaller parcels are flagged here and dropped in the validation

EPOCH_LEN = 2.0
TRIM = 2.0
REJECT_UV = 150.0
MIN_EPOCHS = 30
MAX_SRC_EPOCHS = 60
FIXED_SPLIT = 300

STIM1_WINDOW = (0.0, 300.0)
FIRST_STIM = "open"
MIN_WINDOW = 200.0

# GSN-HydroCel-128 outer ring (face, cheek, neck)
OUTER_RING = ("E17", "E38", "E43", "E44", "E48", "E49", "E56", "E63", "E68", "E73", "E81", "E88",
              "E94", "E99", "E107", "E113", "E114", "E119", "E120", "E121", "E125", "E126", "E127", "E128")
EXCLUDE_CHANNELS = ()           # set to OUTER_RING to test excluding the face/neck channels

#100 schizphrenia
_SCZ = """D01370_ses-101 D01394_ses-101 D01883_ses-101 D01883_ses-111 D00300_ses-101 D00360_ses-101 
D00360_ses-111 D01519_ses-101 D01668_ses-101 D01528_ses-101 D00336_ses-111 D00931_ses-101
D01405_ses-101 D01510_ses-101 D01510_ses-111 D01505_ses-101 D00932_ses-101 D00586_ses-101
D00850_ses-101 D01870_ses-101 D00322_ses-101 D00322_ses-111 D02652_ses-101 D01406_ses-101
D02002_ses-101 D02386_ses-101 D02614_ses-101 D02084_ses-101 D02653_ses-101 D00245_ses-101
D00245_ses-111 D02288_ses-101 D01713_ses-101 D00945_ses-111 D00766_ses-101 D00829_ses-101
D00829_ses-111 D02421_ses-101 D01416_ses-101 D02069_ses-101 D02069_ses-111 D01619_ses-101
D01730_ses-101 D00774_ses-101 D00157_ses-101 D02118_ses-101 D02118_ses-111 D02480_ses-101
D01231_ses-101 D01524_ses-101 D01694_ses-101 D02473_ses-101 D01386_ses-101 D00892_ses-101
D00298_ses-101 D00891_ses-101 D00937_ses-101 D00477_ses-101 D00548_ses-101 D00548_ses-111
D00400_ses-101 D00400_ses-111 D00399_ses-101 D00399_ses-111 D02214_ses-101 D02090_ses-101
D02612_ses-101 D00223_ses-101 D00466_ses-101 D01869_ses-101 D00549_ses-101 D00549_ses-111
D02587_ses-101 D02425_ses-101 D01720_ses-101 D00307_ses-111 D02508_ses-101 D02319_ses-101
D01474_ses-101 D00906_ses-101 D00906_ses-111 D00818_ses-101 D01073_ses-101 D01073_ses-111
D01044_ses-101 D02542_ses-101 D01211_ses-101 D02249_ses-101 D01974_ses-101 D00542_ses-101
D02307_ses-101 D01252_ses-101 D00886_ses-101 D00767_ses-101 D00323_ses-101 D00323_ses-111
D01840_ses-101 D01840_ses-111 D00852_ses-101 D02431_ses-101"""

#120 hcs yay
_HC = """D01852_ses-101 D01587_ses-101 D01412_ses-101 D01960_ses-101 D02152_ses-101 D02176_ses-101
D01345_ses-101 D02191_ses-101 D02123_ses-101 D01566_ses-101 D02067_ses-101 D02067_ses-111
D01530_ses-101 D02254_ses-101 D01489_ses-101 D01489_ses-111 D02171_ses-101 D02173_ses-101
D01594_ses-101 D01594_ses-111 D01551_ses-101 D01927_ses-101 D01927_ses-111 D01409_ses-101
D02172_ses-101 D02151_ses-101 D01500_ses-101 D01552_ses-101 D01606_ses-101 D02169_ses-101
D01385_ses-101 D01385_ses-111 D01636_ses-101 D01704_ses-101 D01603_ses-101 D01603_ses-111
D01515_ses-101 D02239_ses-101 D02134_ses-101 D01545_ses-101 D01473_ses-101 D01464_ses-101
D02053_ses-101 D02053_ses-111 D02109_ses-101 D01488_ses-101 D02125_ses-101 D02125_ses-111
D02331_ses-101 D02139_ses-101 D01490_ses-101 D02149_ses-101 D02126_ses-101 D01627_ses-101
D01877_ses-101 D01521_ses-101 D01875_ses-101 D01289_ses-101 D01468_ses-101 D01507_ses-101
D02332_ses-101 D01899_ses-101 D01491_ses-101 D02327_ses-101 D01767_ses-101 D02157_ses-101
D02161_ses-101 D02063_ses-101 D01815_ses-101 D01815_ses-111 D01555_ses-101 D02160_ses-101
D01746_ses-101 D01746_ses-111 D01471_ses-101 D02013_ses-101 D01526_ses-101 D01577_ses-101
D02432_ses-101 D02360_ses-101 D02162_ses-101 D01487_ses-101 D02177_ses-101 D02159_ses-101
D02121_ses-101 D01411_ses-101 D01565_ses-101 D02170_ses-101 D01614_ses-101 D01747_ses-101
D01738_ses-101 D02138_ses-101 D02022_ses-101 D01993_ses-111 D01903_ses-101 D02112_ses-101
D01467_ses-101 D02153_ses-101 D01567_ses-101 D01459_ses-101 D01354_ses-101 D02140_ses-101
D01501_ses-101 D01549_ses-101 D01661_ses-101 D01661_ses-111 D02066_ses-101 D01344_ses-101
D01344_ses-111 D01559_ses-101 D01626_ses-101 D02120_ses-101 D02244_ses-101 D02651_ses-101
D01470_ses-101 D01583_ses-101 D01554_ses-101 D02150_ses-101 D01853_ses-101 D02029_ses-101"""

SESSIONS = {f"sub-{s}": g for g, txt in (("SCZ", _SCZ), ("HC", _HC)) for s in txt.split()}


def find_files(root, sessions, glob):
    cands = [p for p in Path(root).glob(glob) if p.is_file() and OUTPUT_PATH not in p.parents]
    found, missing = [], []
    for s in sessions:
        pat = re.compile(re.escape(s) + r"(?!\d)")
        hits = sorted(p for p in cands if pat.search(p.name))
        if hits:
            found.append(hits[0])
        else:
            missing.append(s)
    return found, missing


def session_id(path):
    m = re.search(r"sub-[A-Za-z0-9]+_ses-\d+(_run-\d+)?", Path(path).name)
    if m:
        return m.group(0)
    s = re.sub(r"\.fif(\.gz)?$", "", Path(path).name)
    return re.sub(r"[-_](raw|eeg|epo|preproc\w*|clean\w*|filt\w*)$", "", s)


def is_epochs(path):
    return bool(re.search(r"epo\.fif(\.gz)?$", Path(path).name))


def build_template(subjects_dir=None):
    fs_dir = Path(mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False))
    sdir = fs_dir.parent
    if not FWD_PATH.exists():
        FWD_PATH.parent.mkdir(parents=True, exist_ok=True)
        src = mne.setup_source_space("fsaverage", spacing="oct6", subjects_dir=sdir, add_dist=False)
        bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
        info = mne.create_info([f"E{i}" for i in range(1, 129)], 250.0, "eeg")
        info.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), match_case=False)
        fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=str(bem),
                                        eeg=True, meg=False, mindist=5.0)
        mne.write_forward_solution(str(FWD_PATH), fwd, overwrite=True)
    return str(sdir)


@lru_cache(maxsize=1)
def _template(subjects_dir):
    fwd = mne.convert_forward_solution(mne.read_forward_solution(str(FWD_PATH)), surf_ori=True)
    labels = [l for l in mne.read_labels_from_annot("fsaverage", PARCELLATION, subjects_dir=subjects_dir)
              if not l.name.lower().startswith(LABEL_SKIP)]
    return fwd, labels


def label_sizes(fwd, labels):
    src = fwd["src"]
    return np.array([np.isin(l.vertices, src[0 if l.hemi == "lh" else 1]["vertno"]).sum() for l in labels])


def mont_and_bads(inst, params, extra_bads=()):
    names = [c for c in inst.ch_names if re.fullmatch(r"E\d+", c) and c != "E129"]
    inst.pick(names if names else "eeg")
    bads = set(inst.info["bads"]) | set(params.get("bads", [])) | set(extra_bads)
    inst.info["bads"] = [c for c in inst.ch_names if c in bads]
    has_avg = any("average" in p["desc"] for p in inst.info["projs"])
    if not has_avg:
        inst.set_eeg_reference("average", projection=True)   # the inverse needs an average reference
    inst.apply_proj()
    return inst


def annotation_times(raw):
    anno = raw.annotations
    shift = raw.first_time if anno.orig_time is not None else 0.0
    return [(float(o) - shift, d) for o, d in zip(anno.onset, anno.description)]


def find_stim_onsets(raw, o):
    anns = annotation_times(raw)   # MNE keeps annotations sorted by onset

    lo, hi = o["stim1_window"]
    first = [(t, d) for t, d in anns if lo <= t <= hi]
    onset1, label1 = first[0]

    second = [(t, d) for t, d in anns if t > onset1 and d != label1]
    onset2, label2 = second[0]
    return onset1, label1, onset2, label2


def _as_blocks(w):
    arr = np.atleast_2d(np.asarray(w, dtype=float))
    return [(float(a), float(b)) for a, b in arr]


def _fmt_blocks(blocks):
    return ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in blocks)


def get_windows(raw, params, o):
    end = float(raw.times[-1])
    w = params.get("windows")
    if w and all(c in w for c in CONDITIONS):
        return {c: _as_blocks(w[c]) for c in CONDITIONS}, "params", ""
    first, second = (("eyes-open", "eyes-closed") if o["first_stim"] == "open"
                     else ("eyes-closed", "eyes-open"))
    try:
        o1, l1, o2, l2 = find_stim_onsets(raw, o)
        win = {first: [(o1, o2)], second: [(o2, end)]}
        desc = f"{l1}@{o1:.1f}s -> {l2}@{o2:.1f}s"
        short = [c for c in CONDITIONS if win[c][0][1] - win[c][0][0] < o["min_window"]]
        if not short:
            return win, "markers", f"   [{desc}]"
        note = f"   [markers {desc} rejected: {', '.join(short)} < {o['min_window']:.0f}s]"
    except IndexError:                                  # no marker in stim1_window, or no second type
        note = "   [markers unusable: stimulus 1 or stimulus 2 not found]"
    if not o["allow_fixed"]:
        raise ValueError(note.strip(" []") + " (fixed-split fallback disabled)")
    return ({first: [(0.0, min(FIXED_SPLIT, end))], second: [(FIXED_SPLIT, end)]}, "fixed", note)


def epochs_from_raw(raw, win, o):
    out = {}
    for cond, blocks in win.items():
        parts = []
        for a, b in blocks:
            tmin = max(a + o["trim"], 0.0)
            tmax = min(b - o["trim"], raw.times[-1])
            if tmax - tmin < o["epoch_len"]:
                continue  # block too short after trimming
            seg = raw.copy().crop(tmin, tmax, include_tmax=False)
            ep = mne.make_fixed_length_epochs(seg, duration=o["epoch_len"], preload=True)
            if len(ep):
                parts.append(ep)
        if not parts:
            raise ValueError(f"{cond}: no block long enough after trimming ({_fmt_blocks(blocks)})")
        ep = parts[0] if len(parts) == 1 else mne.concatenate_epochs(parts)
        ep.drop_bad(reject=dict(eeg=o["reject_uv"] * 1e-6))
        out[cond] = ep
    return out


def split_epochs(epo, o):
    keys = dict(epo.event_id)
    eo = [k for k in keys if re.search(r"open|\beo\b", k, re.I)]
    ec = [k for k in keys if re.search(r"clos|\bec\b", k, re.I)]
    if eo and ec:
        return {"eyes-open": epo[eo], "eyes-closed": epo[ec]}, "event_id"
    samples = epo.events[:, 0]
    t = (samples - samples.min()) / epo.info["sfreq"]
    return {"eyes-open": epo[t < FIXED_SPLIT], "eyes-closed": epo[t >= FIXED_SPLIT]}, "fixed"


def inspect_stc(stc, src, labels, subjects_dir):
    n_vert, n_t = stc.data.shape
    n_lh, n_rh = stc.lh_data.shape[0], stc.rh_data.shape[0]
    rows_sum_ok = (n_lh + n_rh == n_vert)
    vertno_match_src = (np.array_equal(stc.lh_vertno, src[0]["vertno"]) and
                        np.array_equal(stc.rh_vertno, src[1]["vertno"]))

    pv, pt = np.unravel_index(np.argmax(stc.data), stc.data.shape)
    if pv < n_lh:
        h, i, vsurf = "lh", 0, int(stc.lh_vertno[pv])
    else:
        h, i, vsurf = "rh", 1, int(stc.rh_vertno[pv - n_lh])
    val = float(stc.data[pv, pt])

    xyz = np.ravel(mne.vertex_to_mni(vsurf, i, "fsaverage", subjects_dir=subjects_dir))
    parcel = next((l.name for l in labels if l.hemi == h and vsurf in l.vertices), "none")
    return dict(n_vertices=n_vert, n_times=n_t, n_lh=n_lh, n_rh=n_rh,
                rows_sum_ok=rows_sum_ok, vertno_match_src=vertno_match_src,
                peak_hemi=h, peak_vertex=vsurf, peak_value=val,
                peak_mni_x=float(xyz[0]), peak_mni_y=float(xyz[1]), peak_mni_z=float(xyz[2]),
                peak_parcel=parcel)


def plot_peaks(pk, subjects_dir):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for _, r in pk.iterrows():
        name = f"{r.session_id}_{r.method}_{r.condition}"
        try:
            stc = mne.read_source_estimate(str(STC_DIR / name), subject="fsaverage")
            brain = stc.plot(subject="fsaverage", subjects_dir=subjects_dir, hemi=r.peak_hemi,
                             views="lateral", initial_time=0.0, time_unit="s", size=(800, 800),
                             smoothing_steps=5, clim=dict(kind="percent", lims=[80, 90, 99]))
            brain.add_foci(int(r.peak_vertex), coords_as_verts=True, hemi=r.peak_hemi, color="blue")
            brain.add_text(0.1, 0.9, f"{name} peak: {r.peak_parcel}", "title", font_size=14)
            brain.save_image(str(PLOT_DIR / f"{name}.png"))
            brain.close()
        except Exception as e:
            print(f"plot failed for {name}: {type(e).__name__}: {e}")


def source_alpha(ep, inv, o, sid, cond, method):
    fwd, labels = _template(o["subjects_dir"])
    idx = np.unique(np.linspace(0, len(ep) - 1, min(len(ep), o["max_src_epochs"])).astype(int))
    gen = compute_source_psd_epochs(ep[idx], inv, lambda2=LAMBDA2, method=method,
                                    fmin=PSD_RANGE[0], fmax=PSD_RANGE[1],
                                    bandwidth=PSD_BANDWIDTH, return_generator=True)
    acc, n, last = None, 0, None
    for stc in gen:
        acc = stc.data.copy() if acc is None else acc + stc.data
        n, last = n + 1, stc
    if n == 0:
        raise ValueError(f"{cond}: no epochs reached the inverse")
    spec = acc / n                                       # mean over epochs: vertices x frequencies
    freqs = last.times                                   # for PSD estimates, stc "times" are the frequencies
    in_alpha = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha = spec[:, in_alpha].mean(axis=1)               # then mean over the 8-13 Hz bins
    stc_a = mne.SourceEstimate(alpha[:, None], last.vertices, tmin=0.0, tstep=1.0, subject="fsaverage")
    stc_a.save(str(STC_DIR / f"{sid}_{method}_{cond}"), overwrite=True)

    info = inspect_stc(stc_a, fwd["src"], labels, o["subjects_dir"])

    vals = mne.extract_label_time_course(stc_a, labels, fwd["src"], mode="mean", allow_empty=True)[:, 0]
    vals = np.where(vals == 0, np.nan, vals)
    rel = vals / np.nanmean(vals)                        # relative to the within-condition mean
    rows = [dict(label=l.name, hemi=l.hemi, n_vertices=int(nv), alpha_source=v, alpha_source_rel=r,
                 n_epochs_clean=len(ep), n_epochs_used=len(idx))
            for l, nv, v, r in zip(labels, label_sizes(fwd, labels), vals, rel)]

    stc_spec = mne.SourceEstimate(spec, last.vertices, tmin=freqs[0], tstep=freqs[1] - freqs[0], subject="fsaverage")
    parc_spec = mne.extract_label_time_course(stc_spec, labels, fwd["src"], mode="mean", allow_empty=True)
    spectra = [dict(label=l.name, freq=round(float(f), 2), power=float(pw))
               for l, row in zip(labels, parc_spec) for f, pw in zip(freqs, row)]
    return rows, info, spectra


def run_file(job):
    path, o = job
    sid = session_id(path)
    group = o["groups"].get(sid, "NA")
    s = dict(session_id=sid, group=group, file=path, status="ok", window_source="", note="",
             eo_start=np.nan, eo_end=np.nan, ec_start=np.nan, ec_end=np.nan, n_good_channels=np.nan,
             bads="", n_epochs_eyes_open=np.nan, n_epochs_eyes_closed=np.nan)
    log, rows, peaks, spectra = [sid], [], [], []
    try:
        pj = OUTPUT_PATH / f"{sid}.json"
        params = json.loads(pj.read_text()) if pj.exists() else {}

        win, note = None, ""
        if is_epochs(path):
            inst = mont_and_bads(mne.read_epochs(path, preload=True), params, o["exclude"])
            eps, how = split_epochs(inst, o)
        else:
            inst = mont_and_bads(mne.io.read_raw_fif(path, preload=True), params, o["exclude"])
            win, how, note = get_windows(inst, params, o)
            log.append(f"    windows ({how}): "
                       + "   ".join(f"{c} {_fmt_blocks(win[c])}" for c in CONDITIONS) + note)
            eps = epochs_from_raw(inst, win, o)
            s.update(eo_start=win["eyes-open"][0][0], eo_end=win["eyes-open"][-1][1],
                     ec_start=win["eyes-closed"][0][0], ec_end=win["eyes-closed"][-1][1])
        s.update(window_source=how, note=note.strip(), bads=" ".join(inst.info["bads"]),
                 n_good_channels=len(mne.pick_types(inst.info, eeg=True, exclude="bads")),
                 n_epochs_eyes_open=len(eps["eyes-open"]), n_epochs_eyes_closed=len(eps["eyes-closed"]))
        log.append(f"    bads: {s['bads'] or 'none'}")
        for c in CONDITIONS:
            if len(eps[c]) < o["min_epochs"]:
                raise ValueError(f"{c}: only {len(eps[c])} clean epochs (< {o['min_epochs']})")

        fwd, _ = _template(o["subjects_dir"])
        inv = make_inverse_operator(inst.info, fwd, mne.make_ad_hoc_cov(inst.info), loose=LOOSE, depth=DEPTH)
        for method in o["methods"]:
            for c in CONDITIONS:
                rws, info, spec = source_alpha(eps[c], inv, o, sid, c, method)
                for r in rws:
                    r.update(session_id=sid, group=group, method=method, condition=c, window_source=how)
                rows += rws
                spectra += [dict(session_id=sid, group=group, method=method, condition=c, **r) for r in spec]
                peaks.append(dict(session_id=sid, group=group, method=method, condition=c, window_source=how,
                                  blocks=_fmt_blocks(win[c]) if win else "", **info))
                log.append(f"    {method} {c}: peak {info['peak_hemi']} {info['peak_parcel']}")
    except Exception as e:
        s["status"] = f"error: {type(e).__name__}: {e}"
        log.append(f"    ERROR {type(e).__name__}: {e}")
        (ERR_DIR / f"{sid}.txt").write_text(traceback.format_exc())
        rows, peaks, spectra = [], [], []
    return dict(session=s, log=log, rows=rows, peaks=peaks, spectra=spectra)


def main():
    for d in (OUTPUT_PATH, SRC_PATH, STC_DIR, ERR_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUTPUT_PATH / "source.log")])
    log = logging.getLogger("src")

    files, missing = find_files(INPUT_PATH, list(SESSIONS), "**/*.fif")
    if missing:
        log.warning("%d sessions have no .fif: %s", len(missing), ", ".join(missing))
    log.info("%d sessions to process", len(files))

    sdir = build_template()
    fwd, labels = _template(sdir)
    sizes = label_sizes(fwd, labels)
    pd.DataFrame(dict(label=[l.name for l in labels], hemi=[l.hemi for l in labels],
                      n_vertices_source_space=sizes, low_vertex_count=sizes < MIN_PARCEL_VERTICES)
                 ).to_csv(PARCEL_TABLE_CSV, index=False)

    o = dict(methods=SRC_METHODS, max_src_epochs=MAX_SRC_EPOCHS, epoch_len=EPOCH_LEN, trim=TRIM,
             reject_uv=REJECT_UV, min_epochs=MIN_EPOCHS, stim1_window=STIM1_WINDOW, first_stim=FIRST_STIM,
             min_window=MIN_WINDOW, allow_fixed=True, exclude=EXCLUDE_CHANNELS,
             subjects_dir=sdir, groups=SESSIONS)

    sessions, rows, peaks, spectra = [], [], [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_file, [(str(p), o) for p in files]):
            for line in r["log"]:
                log.info(line)
            sessions.append(r["session"])
            rows += r["rows"]
            peaks += r["peaks"]
            spectra += r["spectra"]

    sessions = pd.DataFrame(sessions)
    sessions.to_csv(SESSIONS_CSV, index=False)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(peaks).to_csv(PEAK_CSV, index=False)
    pd.DataFrame(spectra).to_csv(SPECTRA_CSV, index=False)
    # plot_peaks(pd.DataFrame(peaks), sdir)         
    n_failed = int((sessions.status != "ok").sum()) if len(sessions) else 0
    log.info("Done. %d sessions, %d failed -> %s (tracebacks in %s)", len(files), n_failed, OUT_CSV, ERR_DIR)


if __name__ == "__main__":
    main()