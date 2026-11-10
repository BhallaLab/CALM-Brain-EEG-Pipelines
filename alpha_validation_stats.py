"""
source loc validation and qc figure and table generation for eLORETA source alpha results.
can be extende to other source methods if needed. operations:
1. select subjects and parcels, log10 transform alpha power
2. parcel data and drop parcels with fewer than desirable vertices
3. positive control: occipital alpha must be higher with eyes closed (Berger effect)
4. parcel-wise testing for hedge's g and 95% CI, Benjamini-Hochberg FDR correction across parcels within each contrast
5. whole-brain level testing (mean over parcels of log10 power)
6. QC tables and figures:
        fig01_windows             condition windows per session
        fig02_alpha_box           whole-brain alpha power, EO vs EC; SCZ left, HC right
        fig03_parcel_vertices     parcel size in the source space
        fig04_peak_parcels        most frequent peak parcels, EO and EC
        fig05_berger              occipital alpha, EO -> EC per subject
        fig06_reactivity_profile  log10(EC/EO) per parcel
        fig07_psd                 spectra of the most active parcels, HC vs SCZ (needs source_spectra.csv)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_PATH = Path("/home/calm-brain/vsuresh/eeg_feature_extraction/source_localisation")
SESSIONS_CSV = OUTPUT_PATH / "source_sessions.csv"
OUT_CSV = OUTPUT_PATH / "source_alpha.csv"
PEAK_CSV = OUTPUT_PATH / "source_alpha_peaks.csv"
PARCEL_TABLE_CSV = OUTPUT_PATH / "parcellation_table.csv"
SPECTRA_CSV = OUTPUT_PATH / "source_spectra.csv"
FWD_PATH = OUTPUT_PATH / "source_loc" / "fsaverage-oct6-eeg-fwd.fif"

METHOD = "eLORETA"
CONDITIONS = ("eyes-open", "eyes-closed")
ALPHA_BAND = (8.0, 13.0)
MIN_PARCEL_VERTICES = 3
N_TOP_PARCELS = 6                                  
BRAIN_MAPS = False

EXCLUDE_SESSIONS = {"sub-D00542_ses-101"}           # windows not settled yet
ALLOWED_WINDOW_SOURCES = ("params", "event_id", "markers", "fixed")   
OCCIPITAL_LABELS = ("lateraloccipital", "lingual", "pericalcarine", "cuneus")
FDR_ALPHA = 0.05
MIN_N_PER_TEST = 5                                  

CONDITION_CONTRASTS = [("EC-EO | all subjects", None), ("EC-EO | HC", "HC"), ("EC-EO | SCZ", "SCZ")]
GROUP_CONTRASTS = [("SCZ-HC | EO", "EO"), ("SCZ-HC | EC", "EC"), ("SCZ-HC | EC-EO", "REACT"),
                   ("SCZ-HC | EO (relative)", "EO_rel"), ("SCZ-HC | EC (relative)", "EC_rel")]


def welch(a, b):
    num = a.mean(0) - b.mean(0)
    den = (np.var(a, axis = 0, ddof = 1)/len(a) + np.var(b, axis = 0, ddof = 1)/len(b)) ** 0.5
    w = num/den
    return w


def hedges_g(a, b):
    na, nb = len(a), len(b)
    tot = len(a) + len(b)
    sp = ((np.var(a, axis = 0, ddof = 1)*(na - 1) + np.var(b, axis = 0, ddof = 1)*(nb - 1))/(tot - 2)) ** 0.5
    n = (a.mean(0) - b.mean(0))/sp * (1 - 3/(4*tot - 9))
    return n


def berger(df, s):
    keep = df[df.subject.isin(s)]
    m = keep.groupby(['subject', 'group', 'cond']).val.mean()
    m = m.unstack()
    m['ratio'] = m['ec']/m['eo']
    print(m.sort_values('ratio'))
    n_low = (m['ratio'] < 1).sum()
    print(f"{n_low}/{len(m)} subjects have lower eyes-closed value")
    return m, n_low


def bh_fdr(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def test_paired(A, B):
    n = A.shape[0]
    D = A - B
    md, sd = D.mean(0), D.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = md / (sd / np.sqrt(n))
        g = md / sd * (1 - 3 / (4 * (n - 1) - 1))
    se = np.sqrt(1 / n + g ** 2 / (2 * n))
    return pd.DataFrame(dict(n_a=n, n_b=n, mean_a=A.mean(0), mean_b=B.mean(0), diff=md, ratio=10 ** md,
                             t=t, df=n - 1, p=2 * stats.t.sf(np.abs(t), n - 1),
                             g=g, g_lo=g - 1.96 * se, g_hi=g + 1.96 * se))


def test_welch(A, B):
    n1, n2 = len(A), len(B)
    m1, m2, v1, v2 = A.mean(0), B.mean(0), A.var(0, ddof=1), B.var(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = welch(A, B)
        g = hedges_g(A, B)
        se2 = v1 / n1 + v2 / n2
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return pd.DataFrame(dict(n_a=n1, n_b=n2, mean_a=m1, mean_b=m2, diff=m1 - m2, ratio=10 ** (m1 - m2),
                             t=t, df=df, p=2 * stats.t.sf(np.abs(t), df),
                             g=g, g_lo=g - 1.96 * se, g_hi=g + 1.96 * se))


def run_contrasts(meas, grp, labels, fdr):
    grp = np.asarray(grp)
    parts = []

    def add(res, family, contrast, a_name, b_name):
        res = res.copy()
        res.insert(0, "label", labels)
        res.insert(0, "b_name", b_name)
        res.insert(0, "a_name", a_name)
        res.insert(0, "contrast", contrast)
        res.insert(0, "family", family)
        res["p_fdr"] = bh_fdr(res.p.to_numpy()) if fdr else np.nan
        res["sig_fdr"] = res.p_fdr < FDR_ALPHA if fdr else False
        parts.append(res)

    if "EC" in meas and "EO" in meas:
        for name, sub in CONDITION_CONTRASTS:
            mask = np.ones(len(grp), bool) if sub is None else grp == sub
            if mask.sum() >= MIN_N_PER_TEST:
                add(test_paired(meas["EC"][mask], meas["EO"][mask]), "condition", name, "EC", "EO")
    scz = grp == "SCZ"
    if scz.sum() >= MIN_N_PER_TEST and (~scz).sum() >= MIN_N_PER_TEST:
        for name, m in GROUP_CONTRASTS:
            if m in meas:
                add(test_welch(meas[m][scz], meas[m][~scz]), "group", name, "SCZ", "HC")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_subjects(sessions):
    s = sessions.copy()
    s["subject"] = s.session_id.str.extract(r"(sub-[^_]+)", expand=False)
    s["included"], s["reason"] = True, ""

    def drop(mask, why):
        m = mask & s.included
        s.loc[m, "reason"] = why
        s.loc[m, "included"] = False

    drop(~s.status.eq("ok"), "source_alpha.py failed")
    drop(s.session_id.isin(EXCLUDE_SESSIONS), "manually excluded (EXCLUDE_SESSIONS)")
    drop(~s.window_source.isin(ALLOWED_WINDOW_SOURCES), "window source not allowed (e.g. fixed split)")
    drop(~s.group.isin(["SCZ", "HC"]), "group not SCZ/HC")
    keep = s[s.included].sort_values("session_id").drop_duplicates("subject").index
    dup = s.index[s.included & ~s.index.isin(keep)]
    s.loc[dup, "reason"] = "second session of this subject (earliest session id kept)"
    s.loc[dup, "included"] = False
    return s


def parcel_matrices(alpha, selected, groups):
    d = alpha[alpha.session_id.isin(selected) & (alpha.n_vertices >= MIN_PARCEL_VERTICES)]
    d = d[np.isfinite(d.alpha_source) & (d.alpha_source > 0)]
    wide = {c: np.log10(d[d.condition == c].pivot(index="session_id", columns="label", values="alpha_source"))
            for c in CONDITIONS}
    eo, ec = wide["eyes-open"], wide["eyes-closed"]
    subjects = eo.index.intersection(ec.index)
    eo, ec = eo.loc[subjects], ec.loc[subjects]
    labels = eo.columns.intersection(ec.columns)
    labels = labels[eo[labels].notna().all().to_numpy() & ec[labels].notna().all().to_numpy()]
    eo, ec = eo[labels], ec[labels]
    rel = lambda x: x.sub(x.mean(axis=1), axis=0)                    # parcel minus the subject's mean over parcels
    return dict(eo=eo, ec=ec, labels=list(labels), subjects=subjects, grp=groups.loc[subjects].to_numpy(),
                arrays={"EO": eo.to_numpy(), "EC": ec.to_numpy(), "REACT": (ec - eo).to_numpy(),
                        "EO_rel": rel(eo).to_numpy(), "EC_rel": rel(ec).to_numpy()})


def is_occipital(name):
    return name.rsplit("-", 1)[0] in OCCIPITAL_LABELS


def long_table(M, labels):
    parts = []
    for cond, wide in (("eo", M["eo"]), ("ec", M["ec"])):
        d = (10 ** wide[labels]).stack().rename("val").reset_index()      # back from log10 to power
        d["cond"] = cond
        parts.append(d)
    d = pd.concat(parts).rename(columns={"session_id": "subject"})
    d["group"] = d.subject.map(dict(zip(M["subjects"], M["grp"])))
    return d


def positive_control(M):
    occ = [l for l in M["labels"] if is_occipital(l)]
    rest = [l for l in M["labels"] if not is_occipital(l)]
    if not occ:
        raise ValueError(f"no occipital labels among {OCCIPITAL_LABELS}")
    print("--- occipital parcels")
    occ_m, _ = berger(long_table(M, occ), M["subjects"])
    print("--- rest of the brain")
    rest_m, _ = berger(long_table(M, rest), M["subjects"])
    o = occ_m.reset_index()
    rest_ratio = rest_m.reset_index().set_index("subject").loc[o.subject, "ratio"].to_numpy()
    subj = pd.DataFrame(dict(session_id=o.subject, group=o.group,
                             occ_eo_log10=np.log10(o.eo), occ_ec_log10=np.log10(o.ec),
                             log10_ratio=np.log10(o.ratio), ratio=o.ratio,
                             rest_log10_ratio=np.log10(rest_ratio)))
    rows = []
    for subset in ("all", "HC", "SCZ"):
        d = subj if subset == "all" else subj[subj.group == subset]
        n = len(d)
        if n < 3:
            continue
        x = d.log10_ratio.to_numpy()
        k = int((x > 0).sum())
        t, p = stats.ttest_1samp(x, 0.0)
        gz = test_paired(d[["occ_ec_log10"]].to_numpy(), d[["occ_eo_log10"]].to_numpy()).iloc[0]
        diff = d.log10_ratio - d.rest_log10_ratio
        _, p_topo = stats.ttest_1samp(diff, 0.0)
        sign_p = stats.binomtest(k, n, 0.5).pvalue
        passed = bool(p < 0.05 and sign_p < 0.05 and x.mean() > 0 and diff.mean() > 0 and p_topo < 0.05)
        rows.append(dict(subset=subset, n=n, n_ec_gt_eo=k, frac_ec_gt_eo=k / n,
                         geomean_ratio_ec_eo=10 ** x.mean(), t=t, p=p, sign_test_p=sign_p,
                         hedges_gz=gz.g, g_lo=gz.g_lo, g_hi=gz.g_hi,
                         occ_minus_rest_log10=diff.mean(), occ_vs_rest_p=p_topo, PASS=passed))
    return subj, pd.DataFrame(rows)


def run_stats(sessions, alpha):
    out = OUTPUT_PATH
    alpha = alpha[alpha.method == METHOD]

    sel = select_subjects(sessions)
    sel.to_csv(out / "subject_selection.csv", index=False)
    chosen = sel[sel.included]
    groups = chosen.set_index("session_id").group
    print(f"{len(chosen)} subjects: {(chosen.group == 'SCZ').sum()} SCZ, {(chosen.group == 'HC').sum()} HC")
    print(sel[~sel.included].groupby("reason").size().to_string() if (~sel.included).any() else "    no exclusions")

    M = parcel_matrices(alpha, set(chosen.session_id), groups)
    print(f"\n=== {METHOD}: {len(M['subjects'])} subjects, {len(M['labels'])}/{alpha.label.nunique()} parcels kept ===")

    bs, bsum = positive_control(M)
    if (bsum["subset"] == "all").any():
        a = bsum[bsum["subset"] == "all"].iloc[0]
        print(f"Berger: {a.n_ec_gt_eo}/{a.n} subjects EC>EO, geometric-mean EC/EO = "
              f"{a.geomean_ratio_ec_eo:.2f}, p = {a.p:.2g}, g_z = {a.hedges_gz:.2f}, "
              f"occipital>rest p = {a.occ_vs_rest_p:.2g} -> {'PASS' if a.PASS else 'FAIL'}")
    else:
        print("Berger: fewer than 3 subjects, not tested")

    ps = run_contrasts(M["arrays"], M["grp"], M["labels"], fdr=True)
    glob = {k: M["arrays"][k].mean(axis=1, keepdims=True) for k in ("EO", "EC", "REACT")}
    gs = run_contrasts(glob, M["grp"], ["whole-brain"], fdr=False)

    bs.to_csv(out / "berger_subject_level.csv", index=False)
    bsum.to_csv(out / "berger_summary.csv", index=False)
    cols = ["family", "contrast", "label", "a_name", "b_name", "n_a", "n_b", "mean_a", "mean_b", "diff",
            "ratio", "t", "df", "p", "g", "g_lo", "g_hi", "p_fdr", "sig_fdr"]
    ps[cols].to_csv(out / "parcel_stats.csv", index=False)
    gs[cols].to_csv(out / "global_stats.csv", index=False)

    summ = (ps.groupby(["family", "contrast"], sort=False)
              .agg(n_parcels=("label", "size"), n_sig_fdr=("sig_fdr", "sum"),
                   n_sig_uncorrected=("p", lambda x: int((x < .05).sum())),
                   min_p=("p", "min"), min_p_fdr=("p_fdr", "min"),
                   max_abs_g=("g", lambda x: float(np.nanmax(np.abs(x))))))
    summ.reset_index().to_csv(out / "significant_summary.csv", index=False)

    print("\nparcels significant after FDR")
    print(summ[["n_parcels", "n_sig_fdr"]].to_string())
    print("\nwhole-brain tests (uncorrected p, g)")
    print(gs[["contrast", "t", "p", "g"]].round(3).to_string(index=False))
    return M, ps, bs, bsum

GCOL = {"SCZ":"#dc097d", "HC": "#12aed1"}

def save(fig, name):
    fig.savefig(OUTPUT_PATH / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def qc_sessions(s):
    s[s.status != "ok"].to_csv(OUTPUT_PATH / "qc_failed_sessions.csv", index=False)
    s.groupby(["group", "window_source"]).size().unstack(fill_value=0).to_csv(OUTPUT_PATH / "qc_window_source.csv")
    ok = s[s.status == "ok"]
    ok[["n_epochs_eyes_open", "n_epochs_eyes_closed", "n_good_channels"]].describe().T.to_csv(OUTPUT_PATH / "qc_epochs.csv")

    w = ok[ok.eo_start.notna()].sort_values(["group", "session_id"]).reset_index(drop=True)
    if len(w):
        fig, ax = plt.subplots(figsize=(10, max(3, 0.16 * len(w) + 1)))
        for i, r in w.iterrows():
            ax.barh(i, r.eo_end - r.eo_start, left=r.eo_start, color="#fdae61", height=0.7)
            ax.barh(i, r.ec_end - r.ec_start, left=r.ec_start, color="#4575b4", height=0.7)
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels([f"{r.session_id} {r.group} [{r.window_source}]" for r in w.itertuples()], fontsize=5)
        ax.invert_yaxis()
        ax.set_xlabel("time in recording (s)")
        ax.set_title("Condition windows (orange = eyes open, blue = eyes closed)")
        save(fig, "fig01_windows.png")


def fig_alpha_box(M):
    eo, ec = M["eo"].mean(axis=1).to_numpy(), M["ec"].mean(axis=1).to_numpy()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    for ax, g in zip(axes, ("SCZ", "HC")):
        sel = M["grp"] == g
        vals = [eo[sel], ec[sel]]
        ax.boxplot(vals, positions=[0, 1], widths=0.5, showfliers=False)
        for j, v in enumerate(vals):
            ax.scatter(j + rng.uniform(-.12, .12, len(v)), v, s=14, color=GCOL[g], alpha=.6, zorder=3)
        ax.set_xticks([0, 1], ["eyes open", "eyes closed"])
        ax.set_title(f"{g} (n = {sel.sum()})")
    axes[0].set_ylabel(f"whole-brain alpha power, log10 ({METHOD})")
    fig.suptitle("Alpha power, eyes open vs eyes closed")
    save(fig, "fig02_alpha_box.png")


def qc_parcels(pt):
    pt[pt.low_vertex_count].to_csv(OUTPUT_PATH / "qc_small_parcels.csv", index=False)
    d = pt.sort_values("n_vertices_source_space").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.17 * len(d) + 1.5))
    ax.barh(d.label, d.n_vertices_source_space, color=np.where(d.low_vertex_count, "#c0392b", "#7f8c8d"))
    ax.axvline(MIN_PARCEL_VERTICES, color="k", ls="--", lw=1)
    ax.set_xlabel("source-space vertices in parcel")
    ax.set_title("Parcel size in the oct6 source space")
    ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig03_parcel_vertices.png")


def qc_peaks(pk):
    pk = pk[pk.method == METHOD]
    cnt = pk.groupby(["condition", "peak_parcel"]).size().rename("n").reset_index()
    cnt.sort_values(["condition", "n"], ascending=[True, False]).to_csv(OUTPUT_PATH / "qc_peak_parcels.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(wspace=0.7))
    for ax, cond in zip(axes, CONDITIONS):
        top = pk[pk.condition == cond].peak_parcel.value_counts().head(12)[::-1]
        ax.barh(top.index, top.values, color="#555")
        ax.set_title(f"{cond}: most frequent peak parcels")
        ax.set_xlabel("number of sessions")
        ax.tick_params(axis="y", labelsize=8)
    save(fig, "fig04_peak_parcels.png")


def fig_berger(bs, bsum):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for r in bs.itertuples():
        ax.plot([0, 1], [r.occ_eo_log10, r.occ_ec_log10], color=GCOL[r.group], alpha=.45, lw=.8,
                marker="o", ms=4)
    for g in ("HC", "SCZ"):
        e = bs[bs.group == g]
        ax.plot([0, 1], [e.occ_eo_log10.mean(), e.occ_ec_log10.mean()], color=GCOL[g], lw=3,
                marker="o", ms=10, label=f"{g} mean (n = {len(e)})")
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1], ["eyes open", "eyes closed"])
    ax.set_ylabel(f"occipital alpha power, log10 ({METHOD})")
    a = bsum[bsum["subset"] == "all"]
    if len(a):
        a = a.iloc[0]
        ax.set_title(f"Berger effect: EC>EO in {a.n_ec_gt_eo}/{a.n}\nratio {a.geomean_ratio_ec_eo:.2f}, "
                     f"g_z {a.hedges_gz:.2f}  [{'PASS' if a.PASS else 'FAIL'}]", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig05_berger.png")


def fig_reactivity(ps):
    d = ps[ps.contrast == "EC-EO | all subjects"].copy()
    d["sem"] = (d["diff"] / d["t"]).abs()
    d = d.sort_values("diff")
    occ = [is_occipital(l) for l in d.label]
    fig, ax = plt.subplots(figsize=(6, 0.17 * len(d) + 2))
    ax.barh(range(len(d)), d["diff"], xerr=d["sem"], color=np.where(occ, "#d95f0e", "#7f8c8d"),
            ecolor="k", error_kw=dict(lw=.5))
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(range(len(d)), d.label, fontsize=6)
    ax.set_xlabel("mean log10(EC/EO) ± SEM")
    ax.set_title(f"Alpha reactivity per parcel ({METHOD}; occipital in orange)")
    save(fig, "fig06_reactivity_profile.png")


def fig_psd(M, spectra):
    top = pd.concat([M["eo"], M["ec"]]).mean().sort_values(ascending=False).index[:N_TOP_PARCELS].tolist()
    sp = spectra[(spectra.method == METHOD) & spectra.session_id.isin(M["subjects"]) & spectra.label.isin(top)]
    spec = np.log10(sp.groupby(["session_id", "condition", "freq"]).power.mean()).unstack("freq")
    eo, ec = spec.xs("eyes-open", level="condition"), spec.xs("eyes-closed", level="condition")
    grp = pd.Series(M["grp"], index=M["subjects"])
    freqs = spec.columns.to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [("eyes open", eo), ("eyes closed", ec), ("change, eyes closed vs open", ec - eo)]
    for ax, (title, d) in zip(axes, panels):
        for g in ("HC", "SCZ"):
            x = d[(grp.reindex(d.index) == g).to_numpy()]
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            ax.plot(freqs, m, color=GCOL[g], lw=2, label=f"{g} (n = {len(x)})")
            ax.fill_between(freqs, m - se, m + se, color=GCOL[g], alpha=.25, lw=0)
        ax.axvspan(*ALPHA_BAND, color="grey", alpha=.12)
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])          
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[0].set_ylabel(f"power, log10 ({METHOD})")
    axes[2].set_ylabel("log10(EC/EO)")
    axes[2].axhline(0, color="k", lw=.8)
    fig.suptitle(f"Spectra of the {len(top)} parcels with the highest alpha power (mean ± SEM; alpha band shaded)\n"
                 + ", ".join(top), fontsize=10, y=1.08)
    save(fig, "fig07_psd.png")


def brain_maps(ps):
    import mne
    sdir = str(Path(mne.datasets.fetch_fsaverage(verbose=False)).parent)
    src = mne.read_forward_solution(str(FWD_PATH), verbose=False)["src"]
    labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=sdir, verbose=False)
    vertno = [src[0]["vertno"], src[1]["vertno"]]
    n_lh = len(vertno[0])
    rows_of = {l.name: np.where(np.isin(vertno[h], l.vertices))[0] + (0 if h == 0 else n_lh)
               for l in labels for h in [0 if l.hemi == "lh" else 1]}
    shown = ["EC-EO | all subjects", "SCZ-HC | EO", "SCZ-HC | EC", "SCZ-HC | EC-EO"]
    for c, d in ps[ps.contrast.isin(shown)].groupby("contrast"):
        vals = np.zeros(n_lh + len(vertno[1]))
        for name, g in d.set_index("label").g.items():
            if name in rows_of and np.isfinite(g):
                vals[rows_of[name]] = g
        vmax = max(0.3, float(np.abs(vals).max()))
        try:
            stc = mne.SourceEstimate(vals[:, None], vertno, 0.0, 1.0, subject="fsaverage")
            brain = stc.plot(subject="fsaverage", subjects_dir=sdir, hemi="split", views=["lateral", "medial"],
                             surface="inflated", background="white", size=(1000, 600), time_viewer=False,
                             show_traces=False, clim=dict(kind="value", pos_lims=[0, vmax / 2, vmax]))
            name = f"brain_{c}".replace(" ", "").replace("|", "_").replace("/", "-") + ".png"
            brain.save_image(str(OUTPUT_PATH / name))
            brain.close()
            print(f"  {name}")
        except Exception as e:                                       
            print(f"  brain plot failed for {c}: {type(e).__name__}: {e}")


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    sessions = pd.read_csv(SESSIONS_CSV)
    alpha = pd.read_csv(OUT_CSV)

    M, ps, bs, bsum = run_stats(sessions, alpha)

    qc_sessions(sessions)                                # fig01
    fig_alpha_box(M)                                     # fig02
    qc_parcels(pd.read_csv(PARCEL_TABLE_CSV))            # fig03
    qc_peaks(pd.read_csv(PEAK_CSV))                      # fig04
    fig_berger(bs, bsum)                                 # fig05
    fig_reactivity(ps)                                   # fig06
    fig_psd(M, pd.read_csv(SPECTRA_CSV))             # fig07
    brain_maps(ps)

if __name__ == "__main__":
    main()