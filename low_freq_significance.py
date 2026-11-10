"""
this code tests for periodicity in the canonical resting EEG bands by
fitting a 1/f aperiodic component to the log-log power spectrum and then looking for 
peaks. this is the fooof methodology, but the code does not use the mne fooof package.
peaks count as significant if after removing the fitted 1/f aperiodic
component, its height exceeds  the peak threshold specified into the standard deviation of the 
flattened spectrum. the criterion fooof uses is 2 SD above the aperiodic-corrected spectral noise
with an absolute floor minimum peak height. 

code written by vaaruni with debugging assistance from claude. 
the fooof method is described in Donoghue et al., 2020, Neuron, 105(5): 1-15.
"""

import argparse
import glob
import os
import re
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
import mne

BANDS = {
    "delta": (2.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
FIT_RANGE = (1.0, 40.0)      # freq range for the aperiodic + peak fit
PEAK_THRESHOLD = 2.0         
MIN_PEAK_HEIGHT = 0.05       
MAX_N_PEAKS = 8              
WIDTH_LIMITS = (0.5, 12.0)   # peak FWHM limits 
GROUP_FRACTION = 0.6         # >= this fraction of subjects -> band is SIGNIFICANT
MIN_FIT_POINTS = 10          # need at least this many usable freq bins to fit

_FWHM_TO_SD = 2.0 * np.sqrt(2.0 * np.log(2.0))

def _aperiodic_nk(freqs, offset, exponent):
    return offset - exponent * np.log10(freqs) # no-knee model for aperiodic compnent


def _gaussian(freqs, ctr, height, width):
    return height * np.exp(-((freqs - ctr) ** 2) / (2.0 * width ** 2))


def _init_aperiodic_guess(freqs, log_psd):
    logf = np.log10(freqs)
    A = np.vstack([np.ones_like(logf), -logf]).T
    (offset, exponent), *_ = np.linalg.lstsq(A, log_psd, rcond=None)
    return [float(offset), float(max(exponent, 0.0))]


def fit_aperiodic(freqs, log_psd, n_iter=3):
    popt = _init_aperiodic_guess(freqs, log_psd)
    try:
        popt, _ = curve_fit(_aperiodic_nk, freqs, log_psd, p0=popt, maxfev=10000)
    except (RuntimeError, ValueError):
        pass
    for _ in range(n_iter):
        resid = log_psd - _aperiodic_nk(freqs, *popt)
        keep = resid <= np.percentile(resid, 90)   # drop top 10% (putative peaks)
        if keep.sum() < MIN_FIT_POINTS:
            break
        try:
            popt, _ = curve_fit(_aperiodic_nk, freqs[keep], log_psd[keep],
                                p0=popt, maxfev=10000)
        except (RuntimeError, ValueError):
            break
    return popt  


def fit_peaks(freqs, flat):
    flat = flat.copy()
    std_flat = np.std(flat)
    if not np.isfinite(std_flat) or std_flat == 0:
        return []
    thresh = max(PEAK_THRESHOLD * std_flat, MIN_PEAK_HEIGHT)
    lo_sd, hi_sd = WIDTH_LIMITS[0] / _FWHM_TO_SD, WIDTH_LIMITS[1] / _FWHM_TO_SD
    peaks = []
    for _ in range(MAX_N_PEAKS):
        idx = int(np.argmax(flat))
        h = flat[idx]
        if h < thresh:
            break
        try:
            popt, _ = curve_fit(
                _gaussian, freqs, flat,
                p0=[freqs[idx], h, 1.0],
                bounds=([freqs[0], 0.0, lo_sd], [freqs[-1], np.inf, hi_sd]),
                maxfev=10000)
        except (RuntimeError, ValueError):
            break
        ctr, height, width = popt
        peaks.append((float(ctr), float(height), float(width * _FWHM_TO_SD)))
        flat = flat - _gaussian(freqs, *popt)
    return peaks


def parameterize(freqs, psd):
    m = (freqs >= FIT_RANGE[0]) & (freqs <= FIT_RANGE[1])
    f = np.asarray(freqs, float)[m]
    p = np.asarray(psd, float)[m]
    good = np.isfinite(p) & (p > 0)
    if good.sum() < MIN_FIT_POINTS:
        return np.nan, np.nan, []
    f, logp = f[good], np.log10(p[good])
    offset, exponent = fit_aperiodic(f, logp)
    flat = logp - _aperiodic_nk(f, offset, exponent)
    peaks = fit_peaks(f, flat)
    return offset, exponent, peaks


def band_peak(peaks, lo, hi):
    inband = [p for p in peaks if lo <= p[0] < hi]
    return max(inband, key=lambda p: p[1]) if inband else None


def channel_spectra(inst, win_sec=2.0):
    sfreq = inst.info["sfreq"]
    n_times = inst.get_data().shape[-1]
    n_per_seg = int(min(n_times, round(win_sec * sfreq)))
    spec = inst.compute_psd(method="welch", fmin=FIT_RANGE[0], fmax=FIT_RANGE[1],
                            n_fft=n_per_seg, n_per_seg=n_per_seg,
                            n_overlap=n_per_seg // 2, verbose="error")
    psds, freqs = spec.get_data(return_freqs=True)
    if psds.ndim == 3:          
        psds = psds.mean(axis=0)
    return freqs, psds          


def analyze_recording(inst, win_sec=2.0):
    freqs, psds = channel_spectra(inst, win_sec=win_sec)
    n_ch = psds.shape[0]
    out = {"ch_names": inst.ch_names,
           "exponent": np.full(n_ch, np.nan),
           "offset": np.full(n_ch, np.nan)}
    for b in BANDS:
        out[b] = {"hit": np.zeros(n_ch),
                  "cf": np.full(n_ch, np.nan),
                  "pw": np.full(n_ch, np.nan)}
    for ci in range(n_ch):
        offset, exponent, peaks = parameterize(freqs, psds[ci])
        out["offset"][ci], out["exponent"][ci] = offset, exponent
        for b, (lo, hi) in BANDS.items():
            p = band_peak(peaks, lo, hi)
            if p is not None:
                out[b]["hit"][ci] = 1.0
                out[b]["cf"][ci] = p[0]
                out[b]["pw"][ci] = p[1]
    return out


def _safe_nanmean(x):
    x = np.asarray(x, float)
    return np.nan if x.size == 0 or np.all(np.isnan(x)) else np.nanmean(x)


def group_report(results_by_cond):
    print("\n" + "=" * 70)
    print("BAND SIGNIFICANCE REPORT")
    print("=" * 70)

    for cond, subjects in results_by_cond.items():
        n = len(subjects)
        print(f"\n[{cond}]  n = {n} subjects")
        if n == 0:
            print("  (no subjects)")
            continue
        mean_exp = _safe_nanmean([_safe_nanmean(s['exponent']) for s in subjects])
        print(f"  mean aperiodic exponent = {mean_exp:.2f}")
        for b in BANDS:
            subj_has = np.array([bool(s[b]['hit'].any()) for s in subjects])
            extent = np.array([s[b]['hit'].mean() for s in subjects])
            cfs = np.concatenate([s[b]['cf'][~np.isnan(s[b]['cf'])]
                                  for s in subjects])
            frac = subj_has.mean()
            print(f"  {b:<6}: peak in {int(subj_has.sum())}/{n} subjects "
                  f"({frac*100:4.0f}%) | mean extent "
                  f"{extent.mean()*100:4.0f}% of channels", end="")
            if cfs.size:
                print(f" | center {np.mean(cfs):.2f} +/- {np.std(cfs):.2f} Hz")
            else:
                print(" | no peaks")
            verdict = ("SIGNIFICANT" if frac >= GROUP_FRACTION else
                       "WEAK/ABSENT -- likely aperiodic or artifact")
            print(f"          verdict: {verdict}")

    conds = list(results_by_cond)
    if len(conds) == 2 and all(len(results_by_cond[c]) for c in conds):
        c1, c2 = conds
        if len(results_by_cond[c1]) == len(results_by_cond[c2]):
            print(f"\n[paired {c1} vs {c2}] (subjects assumed aligned in order)")
            for b in BANDS:
                p1 = np.array([_safe_nanmean(s[b]['pw']) for s in results_by_cond[c1]])
                p2 = np.array([_safe_nanmean(s[b]['pw']) for s in results_by_cond[c2]])
                ok = ~(np.isnan(p1) | np.isnan(p2))
                if ok.sum() > 2 and np.std(p1[ok] - p2[ok]) > 0:
                    t, p = ttest_rel(p1[ok], p2[ok])
                    print(f"  {b:<6} periodic power: t={t:+.2f}, p={p:.4f} "
                          f"(n={int(ok.sum())})")
            e1 = np.array([_safe_nanmean(s['exponent']) for s in results_by_cond[c1]])
            e2 = np.array([_safe_nanmean(s['exponent']) for s in results_by_cond[c2]])
            ok = ~(np.isnan(e1) | np.isnan(e2))
            if ok.sum() > 2 and np.std(e1[ok] - e2[ok]) > 0:
                t, p = ttest_rel(e1[ok], e2[ok])
                print(f"  aperiodic exponent: t={t:+.2f}, p={p:.4f} (n={int(ok.sum())})")


def plot_band_topomap(results, info, band, cond=""):
    import matplotlib.pyplot as plt
    import mne
    frac = np.mean([s[band]['hit'] for s in results], axis=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    try:
        im, _ = mne.viz.plot_topomap(frac, info, axes=ax, show=False,
                                     vlim=(0, 1), cmap="Reds", contours=6,
                                     extrapolate="head")
    except TypeError:
        # older MNE without the extrapolate kwarg
        im, _ = mne.viz.plot_topomap(frac, info, axes=ax, show=False,
                                     vlim=(0, 1), cmap="Reds", contours=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of subjects")
    ax.set_title(f"{band} peak prevalence {cond}\n(fraction of subjects)")
    fig.tight_layout()
    return fig


def save_topomaps(results_by_cond, info, out_dir="."):
    if info is None:
        print("[topomap] no channel info captured - skipping topomaps")
        return
    has_pos = any(np.any(ch["loc"][:3]) for ch in info["chs"])
    if not has_pos:
        print("[topomap] channel positions missing - skipping topomaps "
              "(need a montage for topomaps/source)")
        return
    import matplotlib
    matplotlib.use("Agg")            
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for cond, subjects in results_by_cond.items():
        if not subjects:
            continue
        for b in BANDS:
            fig = plot_band_topomap(subjects, info, b, cond=cond)
            path = os.path.join(out_dir, f"topo_{cond}_{b}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[topomap] wrote {path}")
# ---------------------------------------------------------------------------
# this section is specific to the CALM-Brain multi-annptation triggered EEG data

DATA_DIR = "/home/calm-brain/vsuresh/eeg_feature_extraction/output_files"              
BLOCK_SEC = 300.0            
STIM1_MAX_ONSET = 30.0       
EO_SEGMENT = 1               
EC_SEGMENT = 2              
REST_TOKEN = None   
NON_STIM_PREFIXES = ("bad", "edge", "boundary")

_REC_RE = re.compile(r"sub-D\d+_ses-\d+")


def _find_file(recording_id):
    rid = str(recording_id)
    rid = rid if rid.startswith("sub-") else f"sub-{rid}"
    if "_ses-" in rid:                       
        pats = [f"{rid}.fif", f"{rid}_*.fif"]
    hits = sorted({h for p in pats for h in glob.glob(os.path.join(DATA_DIR, p))})
    if REST_TOKEN:  
        rest = [h for h in hits if REST_TOKEN.lower() in os.path.basename(h).lower()]
        hits = rest
    return hits[0]


def load_recording_ids(path):
    ids = []
    with open(path) as fh:
        for line in fh:
            m = _REC_RE.search(line)
            if m:
                ids.append(m.group(0))
    return sorted(set(ids))


def _read_any(path):
    f = mne.io.read_raw_fif(path, preload=True, verbose="error")
    return f

def _identify_segments(raw):
    end = float(raw.times[-1])
    stims = sorted((float(o), str(d).strip().lower())
                   for o, d in zip(raw.annotations.onset, raw.annotations.description)
                   if not str(d).strip().lower().startswith(NON_STIM_PREFIXES))
    if not stims:                                   # no triggers -> time split
        s1, s2 = 0.0, min(BLOCK_SEC, end)
        note = "no stimulus markers - fell back to time split"
    else:
        s1, s1_label = stims[0]
        s2 = next((o for o, d in stims if d != s1_label), None)  # next distinct type
        note = ""
        if s1 > STIM1_MAX_ONSET:
            note += f"Stim1 at {s1:.0f}s (> {STIM1_MAX_ONSET:.0f}s window); "
        if s2 is None:
            s2 = min(s1 + BLOCK_SEC, end)
            note += "only one stimulus type - eyes-closed derived; "
        note = note or "ok"
    return {"baseline": (0.0, s1), "seg1": (s1, s2), "seg2": (s2, end),
            "s1": s1, "s2": s2, "note": note}


def _resting_split(raw, condition):
    seg = _identify_segments(raw)
    seg_key = "seg1" if {"EO": EO_SEGMENT, "EC": EC_SEGMENT}[condition] == 1 else "seg2"
    tmin, tmax = seg[seg_key]
    if seg["note"] != "ok":
        print(f"  [note:{condition}] {seg['note']}")
    return raw.copy().crop(tmin, tmax)


def _condition_from_inst(inst, condition):
    if isinstance(inst, mne.io.BaseRaw):
        return _resting_split(inst, condition)
    picks = [k for k in inst.event_id
             if any(w in k.lower() for w in ("open" if condition == "EO" else "clos"))]
    if picks:
        return inst[picks]
    raise ValueError(f"Could not match {condition} in epochs event_id "
                     f"{inst.event_id}; edit _condition_from_inst.")


def load_subject(subject_id, condition):
    cond = _condition_from_inst(_read_any(_find_file(subject_id)), condition)
    return cond

def inspect_markers(subject_id):
    import mne
    path = _find_file(subject_id)
    inst = _read_any(path)
    has_pos = any(np.any(ch["loc"][:3]) for ch in inst.info["chs"])
    print(os.path.basename(path))
    print(f"  type={type(inst).__name__}  n_ch={len(inst.ch_names)}  "
          f"sfreq={inst.info['sfreq']:.0f}  "
          f"channel_positions={'yes' if has_pos else 'NO (needed for topomap/source)'}")
    if isinstance(inst, mne.io.BaseRaw):
        print(f"  duration={inst.times[-1]:.1f}s")
        ann = inst.annotations
        if len(ann):
            for lab in np.unique(ann.description):
                sel = ann.description == lab
                onsets = np.sort(ann.onset[sel])
                shown = ", ".join(f"{o:.1f}" for o in onsets[:4])
                print(f"  trigger {lab!r}: {int(sel.sum())}x  onsets=[{shown}]"
                      f"{' ...' if sel.sum() > 4 else ''}s")
        else:
            print("  no annotations -> will fall back to a time split")
        seg = _identify_segments(inst)
        print(f"  DETECTED: Stim1 @ {seg['s1']:.1f}s, Stim2 @ {seg['s2']:.1f}s "
              f"[{seg['note']}]")
        print(f"            baseline {seg['baseline'][1]-seg['baseline'][0]:.0f}s "
              f"(discarded) | Seg1/EO {seg['seg1'][1]-seg['seg1'][0]:.0f}s | "
              f"Seg2/EC {seg['seg2'][1]-seg['seg2'][0]:.0f}s")
    else:  # Epochs
        print(f"  n_epochs={len(inst)}  event_id={inst.event_id}")
        if inst.metadata is not None:
            print(f"  metadata columns={list(inst.metadata.columns)}")
    return inst


def run(recording_ids, conditions=("EO", "EC"), win_sec=2.0, topomap_dir="."):
    results_by_cond = {c: [] for c in conditions}
    info_ref = None
    skipped = []
    total = len(recording_ids)
    for i, rid in enumerate(recording_ids, 1):
        try:
            inst = _read_any(_find_file(rid))
            per_cond = {c: analyze_recording(_condition_from_inst(inst, c),
                                             win_sec=win_sec) for c in conditions}
        except Exception as e:
            skipped.append((rid, str(e).splitlines()[0][:90]))
            continue
        for c in conditions:            # commit only if every condition succeeded
            results_by_cond[c].append(per_cond[c])
        if info_ref is None:
            info_ref = inst.info
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] processed ({len(skipped)} skipped so far)")

    if skipped:
        print(f"\n[skipped {len(skipped)} recordings that failed to load/split]")
        for rid, why in skipped[:15]:
            print(f"  {rid}: {why}")
        if len(skipped) > 15:
            print(f"  ... and {len(skipped) - 15} more")

    group_report(results_by_cond)
    if topomap_dir is not None:
        save_topomaps(results_by_cond, info_ref, out_dir=topomap_dir)
    return results_by_cond, info_ref

DEFAULT_RECORDINGS = [
    'sub-D00019_ses-111',
    'sub-D00020_ses-101',
    'sub-D00025_ses-111',
    'sub-D00057_ses-101',
    'sub-D00057_ses-111',
    'sub-D00059_ses-101',
    'sub-D00080_ses-101',
    'sub-D00094_ses-101',
    'sub-D00101_ses-101',
    'sub-D00246_ses-101',
    'sub-D00263_ses-101',
    'sub-D00273_ses-101',
    'sub-D00274_ses-101',
    'sub-D00283_ses-101',
    'sub-D00283_ses-111',
    'sub-D00301_ses-101',
    'sub-D00301_ses-111',
    'sub-D00322_ses-111',
    'sub-D00324_ses-101',
    'sub-D00327_ses-101',
    'sub-D00336_ses-111',
    'sub-D00348_ses-111',
    'sub-D00349_ses-111',
    'sub-D00352_ses-101',
    'sub-D00353_ses-101',
    'sub-D00353_ses-111',
    'sub-D00354_ses-101',
    'sub-D00354_ses-111',
    'sub-D00355_ses-101',
    'sub-D00359_ses-111',
    'sub-D00360_ses-111',
    'sub-D00380_ses-101',
    'sub-D00386_ses-101',
    'sub-D00399_ses-101',
    'sub-D00401_ses-101',
    'sub-D00412_ses-101',
    'sub-D00417_ses-101',
    'sub-D00424_ses-101',
    'sub-D00429_ses-101',
    'sub-D00431_ses-101',
    'sub-D00432_ses-101',
    'sub-D00434_ses-101',
    'sub-D00454_ses-101',
    'sub-D00470_ses-101',
    'sub-D00471_ses-101',
    'sub-D00472_ses-101',
    'sub-D00476_ses-101',
    'sub-D00486_ses-101',
    'sub-D00486_ses-111',
    'sub-D00487_ses-111',
    'sub-D00500_ses-111',
    'sub-D00502_ses-101',
    'sub-D00510_ses-101',
    'sub-D00513_ses-101',
    'sub-D00521_ses-101',
    'sub-D00531_ses-101',
    'sub-D00536_ses-101',
    'sub-D00543_ses-101',
    'sub-D00548_ses-101',
    'sub-D00548_ses-111',
    'sub-D00549_ses-111',
    'sub-D00551_ses-101',
    'sub-D00551_ses-111',
    'sub-D00558_ses-101',
    'sub-D00586_ses-101',
    'sub-D00617_ses-101',
    'sub-D00752_ses-101',
    'sub-D00763_ses-101',
    'sub-D00763_ses-111',
    'sub-D00767_ses-101',
    'sub-D00775_ses-111',
    'sub-D00796_ses-101',
    'sub-D00797_ses-101',
    'sub-D00798_ses-111',
    'sub-D00800_ses-111',
    'sub-D00803_ses-101',
    'sub-D00806_ses-101',
    'sub-D00810_ses-101',
    'sub-D00815_ses-101',
    'sub-D00829_ses-111',
    'sub-D00830_ses-101',
    'sub-D00834_ses-111',
    'sub-D00850_ses-101',
    'sub-D00851_ses-101',
    'sub-D00854_ses-111',
    'sub-D00862_ses-101',
    'sub-D00885_ses-101',
    'sub-D00886_ses-101',
    'sub-D00893_ses-101',
    'sub-D00894_ses-101',
    'sub-D00912_ses-101',
    'sub-D00923_ses-101',
    'sub-D00935_ses-101',
    'sub-D00948_ses-101',
    'sub-D00955_ses-101',
    'sub-D00961_ses-111',
    'sub-D00966_ses-101',
    'sub-D00967_ses-101',
    'sub-D00973_ses-101',
    'sub-D00974_ses-101',
    'sub-D00974_ses-111',
    'sub-D00975_ses-101',
    'sub-D00976_ses-101',
    'sub-D00988_ses-101',
    'sub-D00989_ses-101',
    'sub-D01000_ses-101',
    'sub-D01000_ses-111',
    'sub-D01002_ses-101',
    'sub-D01006_ses-111',
    'sub-D01007_ses-111',
    'sub-D01008_ses-111',
    'sub-D01011_ses-101',
    'sub-D01013_ses-101',
    'sub-D01020_ses-111',
    'sub-D01022_ses-101',
    'sub-D01023_ses-101',
    'sub-D01026_ses-101',
    'sub-D01028_ses-101',
    'sub-D01029_ses-101',
    'sub-D01029_ses-111',
    'sub-D01030_ses-101',
    'sub-D01037_ses-101',
    'sub-D01037_ses-111',
    'sub-D01038_ses-102',
    'sub-D01044_ses-101',
    'sub-D01050_ses-101',
    'sub-D01050_ses-111',
    'sub-D01054_ses-101',
    'sub-D01067_ses-101',
    'sub-D01086_ses-101',
    'sub-D01092_ses-101',
    'sub-D01097_ses-101',
    'sub-D01110_ses-101',
    'sub-D01115_ses-111',
    'sub-D01119_ses-101',
    'sub-D01123_ses-111',
    'sub-D01127_ses-101',
    'sub-D01127_ses-111',
    'sub-D01141_ses-111',
    'sub-D01143_ses-101',
    'sub-D01148_ses-101',
    'sub-D01154_ses-101',
    'sub-D01156_ses-101',
    'sub-D01161_ses-101',
    'sub-D01181_ses-101',
    'sub-D01183_ses-101',
    'sub-D01190_ses-111',
    'sub-D01192_ses-101',
    'sub-D01192_ses-111',
    'sub-D01193_ses-101',
    'sub-D01216_ses-101',
    'sub-D01223_ses-111',
    'sub-D01226_ses-111',
    'sub-D01239_ses-101',
    'sub-D01241_ses-111',
    'sub-D01242_ses-111',
    'sub-D01258_ses-101',
    'sub-D01304_ses-111',
    'sub-D01305_ses-111',
    'sub-D01334_ses-111',
    'sub-D01336_ses-111',
    'sub-D01337_ses-101',
    'sub-D01339_ses-111',
    'sub-D01341_ses-101',
    'sub-D01341_ses-111',
    'sub-D01344_ses-111',
    'sub-D01351_ses-101',
    'sub-D01357_ses-101',
    'sub-D01398_ses-101',
    'sub-D01442_ses-111',
    'sub-D01443_ses-101',
    'sub-D01446_ses-101',
    'sub-D01461_ses-101',
    'sub-D01465_ses-111',
    'sub-D01489_ses-101',
    'sub-D01489_ses-111',
    'sub-D01519_ses-101',
    'sub-D01533_ses-101',
    'sub-D01535_ses-101',
    'sub-D01537_ses-101',
    'sub-D01565_ses-101',
    'sub-D01578_ses-111',
    'sub-D01581_ses-111',
    'sub-D01594_ses-101',
    'sub-D01598_ses-101',
    'sub-D01602_ses-111',
    'sub-D01603_ses-111',
    'sub-D01605_ses-111',
    'sub-D01607_ses-101',
    'sub-D01608_ses-101',
    'sub-D01609_ses-101',
    'sub-D01611_ses-101',
    'sub-D01614_ses-101',
    'sub-D01615_ses-111',
    'sub-D01621_ses-101',
    'sub-D01623_ses-101',
    'sub-D01626_ses-101',
    'sub-D01627_ses-101',
    'sub-D01638_ses-101',
    'sub-D01644_ses-101',
    'sub-D01646_ses-101',
    'sub-D01670_ses-101',
    'sub-D01675_ses-101',
    'sub-D01680_ses-101',
    'sub-D01681_ses-111',
    'sub-D01682_ses-111',
    'sub-D01695_ses-101',
    'sub-D01695_ses-111',
    'sub-D01708_ses-111',
    'sub-D01720_ses-101',
    'sub-D01723_ses-101',
    'sub-D01730_ses-101',
    'sub-D01738_ses-101',
    'sub-D01739_ses-111',
    'sub-D01746_ses-101',
    'sub-D01746_ses-111',
    'sub-D01754_ses-101',
    'sub-D01774_ses-101',
    'sub-D01774_ses-111',
    'sub-D01775_ses-101',
    'sub-D01775_ses-111',
    'sub-D01776_ses-111',
    'sub-D01777_ses-111',
    'sub-D01778_ses-111',
    'sub-D01813_ses-101',
    'sub-D01818_ses-111',
    'sub-D01820_ses-101',
    'sub-D01821_ses-111',
    'sub-D01822_ses-101',
    'sub-D01856_ses-101',
    'sub-D01860_ses-101',
    'sub-D01870_ses-101',
    'sub-D01871_ses-101',
    'sub-D01883_ses-111',
    'sub-D01902_ses-101',
    'sub-D01920_ses-111',
    'sub-D01930_ses-111',
    'sub-D01932_ses-111',
    'sub-D01945_ses-101',
    'sub-D01946_ses-101',
    'sub-D01960_ses-101',
    'sub-D01974_ses-101',
    'sub-D01983_ses-101',
    'sub-D01993_ses-111',
    'sub-D02016_ses-111',
    'sub-D02017_ses-111',
    'sub-D02034_ses-111',
    'sub-D02068_ses-111',
    'sub-D02085_ses-101',
    'sub-D02090_ses-101',
    'sub-D02094_ses-101',
    'sub-D02097_ses-101',
    'sub-D02106_ses-101',
    'sub-D02126_ses-101',
    'sub-D02127_ses-101',
    'sub-D02129_ses-101',
    'sub-D02133_ses-101',
    'sub-D02135_ses-101',
    'sub-D02140_ses-101',
    'sub-D02150_ses-101',
    'sub-D02152_ses-101',
    'sub-D02157_ses-101',
    'sub-D02158_ses-111',
    'sub-D02160_ses-101',
    'sub-D02161_ses-101',
    'sub-D02164_ses-101',
    'sub-D02169_ses-101',
    'sub-D02170_ses-101',
    'sub-D02171_ses-101',
    'sub-D02172_ses-101',
    'sub-D02173_ses-101',
    'sub-D02176_ses-101',
    'sub-D02177_ses-101',
    'sub-D02191_ses-101',
    'sub-D02196_ses-101',
    'sub-D02203_ses-101',
    'sub-D02205_ses-101',
    'sub-D02214_ses-101',
    'sub-D02221_ses-101',
    'sub-D02231_ses-101',
    'sub-D02253_ses-101',
    'sub-D02255_ses-101',
    'sub-D02264_ses-101',
    'sub-D02268_ses-101',
    'sub-D02280_ses-101',
    'sub-D02286_ses-101',
    'sub-D02306_ses-101',
    'sub-D02309_ses-101',
    'sub-D02312_ses-101',
    'sub-D02313_ses-101',
    'sub-D02320_ses-101',
    'sub-D02333_ses-101',
    'sub-D02334_ses-101',
    'sub-D02337_ses-101',
    'sub-D02338_ses-101',
    'sub-D02339_ses-101',
    'sub-D02351_ses-101',
    'sub-D02360_ses-101',
    'sub-D02362_ses-101',
    'sub-D02370_ses-101',
    'sub-D02374_ses-101',
    'sub-D02380_ses-101',
    'sub-D02381_ses-101',
    'sub-D02385_ses-101',
    'sub-D02386_ses-101',
    'sub-D02387_ses-101',
    'sub-D02398_ses-101',
    'sub-D02401_ses-101',
    'sub-D02402_ses-101',
    'sub-D02408_ses-101',
    'sub-D02409_ses-101',
    'sub-D02410_ses-101',
    'sub-D02415_ses-101',
    'sub-D02418_ses-101',
    'sub-D02432_ses-101',
    'sub-D02433_ses-101',
    'sub-D02437_ses-101',
    'sub-D02445_ses-101',
    'sub-D02459_ses-101',
    'sub-D02460_ses-101',
    'sub-D02476_ses-101',
    'sub-D02478_ses-101',
    'sub-D02479_ses-101',
    'sub-D02483_ses-101',
    'sub-D02497_ses-101',
    'sub-D02522_ses-101',
    'sub-D02534_ses-101',
    'sub-D02544_ses-101',
    'sub-D02564_ses-101',
    'sub-D02565_ses-101',
    'sub-D02566_ses-101',
    'sub-D02588_ses-101',
    'sub-D02591_ses-101',
    'sub-D02598_ses-101',
    'sub-D02599_ses-101',
    'sub-D02612_ses-101',
    'sub-D02613_ses-101',
    'sub-D02615_ses-101',
    'sub-D02633_ses-101',
    'sub-D02641_ses-101',
    'sub-D02643_ses-101',
    'sub-D02651_ses-101',
    'sub-D02664_ses-101',
    'sub-D02665_ses-101',
    'sub-D02667_ses-101',
    'sub-D02674_ses-101',
    'sub-D02677_ses-101',
    'sub-D02697_ses-101',
    'sub-D02706_ses-101',
    'sub-D02707_ses-101',
    'sub-D02721_ses-101',
    'sub-D03262_ses-101',
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", metavar="SUBJECT",
                    help="print how EO/EC are coded for one subject, then exit")
    ap.add_argument("--recordings", metavar="FILE",
                    help="file of recording ids / PNG names (e.g. needed_recordings.txt)")
    ap.add_argument("--data-dir", help="override DATA_DIR (folder with the .fif files)")
    ap.add_argument("--win-sec", type=float, default=2.0,
                    help="PSD window length in seconds (>=2 for delta, >=4 preferred)")
    ap.add_argument("--topomap-dir", default=".",
                    help="folder for topo_{cond}_{band}.png (use '' to skip topomaps)")
    args = ap.parse_args()
    if args.data_dir:
        DATA_DIR = args.data_dir
    topo_dir = args.topomap_dir if args.topomap_dir else None
    if args.inspect:
        inspect_markers(args.inspect)
    elif args.recordings:
        ids = load_recording_ids(args.recordings)
        print(f"loaded {len(ids)} recordings from {args.recordings}")
        run(ids, conditions=("EO", "EC"), win_sec=args.win_sec, topomap_dir=topo_dir)
    else:
        # no --recordings given -> process the full embedded cohort
        print(f"running default cohort of {len(DEFAULT_RECORDINGS)} recordings")
        run(DEFAULT_RECORDINGS, conditions=("EO", "EC"),
            win_sec=args.win_sec, topomap_dir=topo_dir)