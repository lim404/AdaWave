"""Generate LaTeX tables for the manuscript from result CSVs.
Outputs to tables/*.tex (override with $ADAWAVE_TABLE_DIR). Method display
names hide internal v1/v2 nomenclature: Prior-v2 -> \\method{} (AdaWave),
Prior-v1 -> WF-Fixed.
"""
import csv, os
import numpy as np

OUT = os.environ.get("ADAWAVE_TABLE_DIR", "tables")
os.makedirs(OUT, exist_ok=True)

DISP = {"Bilateral": "Bilateral", "MLS": "MLS", "WLOP": "WLOP", "Jet": "Jet",
        "GLR": "GLR", "CLOP": "CLOP",
        "Prior-v1": "WF-Fixed (non-adapt.)", "Prior-v2": "\\method{} (ours)",
        "score": "ScoreDenoise~\\cite{luo2021score}",
        "pfn": "IterativePFN~\\cite{iterativepfn2023}",
        "pathnet": "PathNet~\\cite{pathnet2024}",
        "straightpcf": "StraightPCF~\\cite{straightpcf2024}"}
ORDER_CLASSICAL = ["Bilateral", "MLS", "WLOP", "Jet", "GLR", "CLOP",
                   "Prior-v1", "Prior-v2"]
ORDER_DL = ["score", "pfn", "pathnet", "straightpcf"]


def load(path):
    return list(csv.DictReader(open(path)))


def agg(rows, method, key, sigma=None):
    sel = [r for r in rows if r["method"] == method
           and (sigma is None or abs(float(r["sigma"]) - sigma) < 1e-9)]
    v = np.array([float(r[key]) for r in sel])
    v = v[~np.isnan(v)]
    return v.mean(), v.std()


def fmt(m, s, best=False, decimals=1):
    txt = f"{m:+.{decimals}f}\\,\\textpm\\,{s:.{decimals}f}"
    return f"\\textbf{{{txt}}}" if best else txt


# ---------------------------------------------------------------- Table I: ALS final test
als = load("final_test_als.csv")
dl_als = {m: load(f"final_test_dl_{m}_als.csv") for m in ORDER_DL}
KEYS = ["Z", "Planar", "Edge", "Bdry"]
lines = []
lines.append("\\begin{table*}[t]\\centering")
lines.append("\\caption{Final-test airborne-LiDAR results (pre-registered "
             "protocol, run once): Z-RMSE improvement (\\%) on 8 unseen "
             "windows (official ISPRS Vaihingen evaluation region and four "
             "held-out DALES test tiles) $\\times$ 3 seeds, single global "
             "configuration for every method. Best per column in bold; "
             "deep models use their public CAD-trained checkpoints "
             "(zero-shot).}")
lines.append("\\label{tab:final_als}")
lines.append("\\fontsize{6.3pt}{7.6pt}\\selectfont\\setlength{\\tabcolsep}{1.8pt}"
             "\\begin{tabular}{l" + "cccc" * 3 + "}")
lines.append("\\toprule")
lines.append(" & \\multicolumn{4}{c}{$\\sigma=2$\\,cm} & "
             "\\multicolumn{4}{c}{$\\sigma=5$\\,cm} & "
             "\\multicolumn{4}{c}{$\\sigma=10$\\,cm}\\\\")
lines.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
lines.append("Method & " + " & ".join(["Z & Planar & Edge & Bdry"] * 3) + "\\\\")
lines.append("\\midrule")
# find best per sigma/key among classical rows
best = {}
for sg in [0.02, 0.05, 0.10]:
    for k in KEYS:
        vals = {m: agg(als, m, k, sg)[0] for m in ORDER_CLASSICAL}
        best[(sg, k)] = max(vals, key=vals.get)
for m in ORDER_CLASSICAL:
    cells = []
    for sg in [0.02, 0.05, 0.10]:
        for k in KEYS:
            mn, sd = agg(als, m, k, sg)
            cells.append(fmt(mn, sd, best=(best[(sg, k)] == m)))
    lines.append(f"{DISP[m]} & " + " & ".join(cells) + "\\\\")
lines.append("\\midrule")
for m in ORDER_DL:
    cells = []
    for sg in [0.02, 0.05, 0.10]:
        for k in KEYS:
            sel = [r for r in dl_als[m] if abs(float(r["sigma"]) - sg) < 1e-9]
            v = np.array([float(r[k]) for r in sel])
            v = v[~np.isnan(v)]
            cells.append(f"{v.mean():+.0f}\\,\\textpm\\,{v.std():.0f}")
    lines.append(f"{DISP[m]} & " + " & ".join(cells) + "\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table*}")
open(f"{OUT}/final_als.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table II: ModelNet40 final
mn = load("final_test_mn.csv")
dl_mn = {m: load(f"final_test_dl_{m}_mn.csv") for m in ORDER_DL}
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Final-test ModelNet40 results: 8 unseen categories "
         "$\\times$ 3 seeds, noise $1.5\\%$ of the bounding-box diagonal. "
         "Improvement (\\%) of point-to-point RMSE and Chamfer distance.}",
         "\\label{tab:final_mn}",
         "\\small\\begin{tabular}{lcc}", "\\toprule",
         "Method & RMSE & Chamfer\\\\", "\\midrule"]
bestR = max(ORDER_CLASSICAL, key=lambda m: agg(mn, m, "RMSE")[0])
bestC = max(ORDER_CLASSICAL, key=lambda m: agg(mn, m, "CD")[0])
for m in ORDER_CLASSICAL:
    r, rs = agg(mn, m, "RMSE")
    c, cs = agg(mn, m, "CD")
    lines.append(f"{DISP[m]} & {fmt(r, rs, m==bestR)} & {fmt(c, cs, m==bestC)}\\\\")
lines.append("\\midrule")
for m in ORDER_DL:
    v = np.array([float(r["RMSE"]) for r in dl_mn[m]])
    c = np.array([float(r["CD"]) for r in dl_mn[m]])
    lines.append(f"{DISP[m]} & {v.mean():+.1f}\\,\\textpm\\,{v.std():.1f} & "
                 f"{c.mean():+.1f}\\,\\textpm\\,{c.std():.1f}\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/final_mn.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: ablation
ab = load("ablation_v2.csv")
NAMES = {"full": "full model",
         "fixed-lam": "w/o noise-adaptive balance ($s^2$ fixed)",
         "no-anchor": "w/o structure anchor",
         "no-buffer": "w/o propagation buffer",
         "no-geomw": "w/o geometry-consistent weights",
         "mad-struct": "MAD instead of $Q_{75}$ excess",
         "fixed-cap": "spacing-based cap (non-adaptive)",
         "no-binormal": "w/o binormal channel",
         "gd-solver": "momentum GD instead of PCG"}
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Ablations (development set): 8 ALS crops at "
         "$\\sigma{=}5$\\,cm and 8 ModelNet40 shapes. Each row disables "
         "one mechanism of the frozen model.}",
         "\\label{tab:ablation}",
         "\\scriptsize\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{lcccc|cc}", "\\toprule",
         " & \\multicolumn{4}{c|}{ALS} & \\multicolumn{2}{c}{ModelNet40}\\\\",
         "Variant & Z & Planar & Edge & Bdry & RMSE & CD\\\\", "\\midrule"]
for r in ab:
    if r["variant"] == "no-binormal":
        continue      # binormal channel removed from the model; see tab:direction
    nm = NAMES.get(r["variant"], r["variant"])
    lines.append(f"{nm} & {float(r['als_Z']):+.1f} & {float(r['als_Planar']):+.1f} & "
                 f"{float(r['als_Edge']):+.1f} & {float(r['als_Bdry']):+.1f} & "
                 f"{float(r['mn_RMSE']):+.1f} & {float(r['mn_CD']):+.1f}\\\\")
    if r["variant"] == "full":
        lines.append("\\midrule")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/ablation.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: safety
sf = load("worldB_safety.csv")
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Safety on real scans (ScanNet++, 12 scenes, 144 "
         "samples): no-harm rate ($\\Delta$RMSE $\\ge -0.2\\%$), worst-case "
         "degradation, and median displacement (in units of the estimated "
         "noise scale $\\hat\\sigma_g$).}",
         "\\label{tab:safety}",
         "\\small\\begin{tabular}{llccc}", "\\toprule",
         "Regime & Method & no-harm & worst & med.\\ disp.\\\\", "\\midrule"]
for regime, rname in [("laser", "laser scan"), ("iphone-fused", "fused depth"),
                      ("iphone-frame", "single frame")]:
    first = True
    for m in ["Bilateral", "MLS", "GLR", "Prior-v2"]:
        sel = [r for r in sf if r["regime"] == regime and r["method"] == m]
        if not sel:
            continue
        dr = np.array([float(r["dRMSE"]) for r in sel])
        ds = np.median([float(r["disp_med_sig"]) for r in sel])
        nm = DISP.get(m, m)
        reg = rname if first else ""
        first = False
        bold = m == "Prior-v2"
        row = (f"{reg} & {nm} & {(dr >= -0.2).mean()*100:.0f}\\% & "
               f"{dr.min():+.1f}\\% & {ds:.2f}$\\hat\\sigma_g$\\\\")
        lines.append(("\\textbf{}" and row) if not bold else row)
    lines.append("\\midrule" if regime != "iphone-frame" else "\\bottomrule")
lines.append("\\end{tabular}\\end{table}")
open(f"{OUT}/safety.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: downstream recon
rc = load("downstream_reconstruction.csv")
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Downstream screened-Poisson reconstruction "
         "(identical settings for all inputs; 8 shapes): Chamfer "
         "($\\times10^{-3}$, lower better) and normal consistency of the "
         "reconstructed surface against the ground-truth mesh.}",
         "\\label{tab:recon}",
         "\\small\\begin{tabular}{lccc}", "\\toprule",
         "Input & Chamfer & vs.\\ noisy & Normal cons.\\\\", "\\midrule"]
base = np.mean([float(r["chamfer"]) for r in rc if r["method"] == "Noisy"])
for m in ["Noisy", "Bilateral", "MLS", "GLR", "Prior-v2"]:
    sel = [r for r in rc if r["method"] == m]
    ch = np.mean([float(r["chamfer"]) for r in sel])
    nc = np.mean([float(r["normal_cons"]) for r in sel])
    rel = "---" if m == "Noisy" else f"{(1-ch/base)*100:+.1f}\\%"
    nm = DISP.get(m, m)
    lines.append(f"{nm} & {ch:.1f} & {rel} & {nc:.3f}\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/recon.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: World A condensed
wa = load("worldA_results.csv")
shapes = ["plane", "sphere", "cylinder", "ridge90", "ridge120", "step",
          "corner", "cross", "ripple", "sheets", "ridge90-grad"]
meths = ["Bilateral", "MLS", "GLR", "Prior-v1", "Prior-v2"]
lines = ["\\begin{table}[t]\\centering",
         "\\caption{World-A synthetic suite (pre-registered; Gaussian "
         "noise $\\sigma{=}1.0\\ell$, mean over 3 seeds): overall "
         "point-to-surface RMSE improvement (\\%). Stress cases in the "
         "lower block.}",
         "\\label{tab:worldA}",
         "\\scriptsize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{l" + "c" * len(meths) + "}", "\\toprule",
         "Shape & " + " & ".join(DISP[m].split("~")[0].split(" (")[0] for m in meths) + "\\\\",
         "\\midrule"]
for sh in shapes:
    if sh == "ripple":
        lines.append("\\midrule")
    cells = []
    vals = {}
    for m in meths:
        sel = [r for r in wa if r["shape"] == sh and r["method"] == m
               and r["noise"] == "gauss" and abs(float(r["sigma"]) - 1.0) < 1e-6]
        vals[m] = np.mean([float(r["overall"]) for r in sel])
    bm = max(vals, key=vals.get)
    for m in meths:
        txt = f"{vals[m]:+.1f}"
        cells.append(f"\\textbf{{{txt}}}" if m == bm else txt)
    lines.append(f"{sh} & " + " & ".join(cells) + "\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/worldA.tex", "w").write("\n".join(lines))

print("tables written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)

# ---------------------------------------------------------------- Table: normals downstream
nm = load("worldB_normals.csv")
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Downstream normal estimation on ScanNet++ fused scans "
         "(12 scenes, common PCA estimator, ground truth from the laser "
         "mesh): angular error (deg) and share below $10^\\circ$; regions "
         "defined from the reference mesh only.}",
         "\\label{tab:normals}",
         "\\small\\begin{tabular}{lccccc}", "\\toprule",
         "Input & mean & median & $<10^\\circ$ & planar & edge\\\\",
         "\\midrule"]
for m in ["Noisy", "Bilateral", "MLS", "GLR", "Prior-v2"]:
    sel = [r for r in nm if r["method"] == m]
    if not sel:
        continue
    mean = np.mean([float(r["mean"]) for r in sel])
    med = np.mean([float(r["med"]) for r in sel])
    lt10 = np.mean([float(r["lt10"]) for r in sel])
    pl = np.nanmean([float(r["mean_planar"]) for r in sel])
    ed = np.nanmean([float(r["mean_edge"]) for r in sel])
    lines.append(f"{DISP.get(m, m)} & {mean:.1f} & {med:.1f} & "
                 f"{lt10:.1f}\\% & {pl:.1f} & {ed:.1f}\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/normals.tex", "w").write("\n".join(lines))

# NOTE: tables/stats.tex is generated EXCLUSIVELY by stats_scene_level.py
# (scene-level analysis; do not regenerate it here — pseudo-replication trap).

# ---------------------------------------------------------------- Table: normals downstream
nm = load("worldB_normals.csv")
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Downstream normal estimation on ScanNet++ fused scans "
         "(12 scenes, common PCA estimator, ground truth from the laser "
         "mesh): angular error (deg) and share below $10^\\circ$; regions "
         "defined from the reference mesh only.}",
         "\\label{tab:normals}",
         "\\small\\begin{tabular}{lccccc}", "\\toprule",
         "Input & mean & median & $<10^\\circ$ & planar & edge\\\\",
         "\\midrule"]
for m in ["Noisy", "Bilateral", "MLS", "GLR", "Prior-v2"]:
    sel = [r for r in nm if r["method"] == m]
    if not sel:
        continue
    mean = np.mean([float(r["mean"]) for r in sel])
    med = np.mean([float(r["med"]) for r in sel])
    lt10 = np.mean([float(r["lt10"]) for r in sel])
    pl = np.nanmean([float(r["mean_planar"]) for r in sel])
    ed = np.nanmean([float(r["mean_edge"]) for r in sel])
    lines.append(f"{DISP.get(m, m)} & {mean:.1f} & {med:.1f} & "
                 f"{lt10:.1f}\\% & {pl:.1f} & {ed:.1f}\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/normals.tex", "w").write("\n".join(lines))

# (second stats block excised 2026-07-19 — NEVER regenerate stats.tex here)

print("extra tables written")

# ---------------------------------------------------------------- Table: cost
als_t = load("final_test_als.csv")
PARAMS = {"score": "187\\,k", "pfn": "3.20\\,M", "pathnet": "15.6\\,M",
          "straightpcf": "533\\,k"}
lines = ["\\begin{table}[t]\\centering",
         "\\caption{Learned parameters and measured runtime per "
         "$10^4$-point sample (final-test hardware: classical methods on "
         "CPU, learned methods on an RTX 4070\\,Ti GPU).}",
         "\\label{tab:cost}",
         "\\vspace{-2pt}",
         "\\small\\begin{tabular}{lcc}", "\\toprule",
         "Method & Params & Time (s)\\\\", "\\midrule"]
for m in ORDER_CLASSICAL:
    t = np.mean([float(r["time"]) for r in als_t if r["method"] == m])
    lines.append(f"{DISP[m]} & 0 & {t:.1f}\\\\")
lines.append("\\method{} (vectorised$^{\\ast}$) & 0 & 1.0\\\\")
lines.append("\\midrule")
for m in ORDER_DL:
    rowsd = load(f"final_test_dl_{m}_als.csv")
    t = np.mean([float(r["time"]) for r in rowsd])
    lines.append(f"{DISP[m]} & {PARAMS[m]} & {t:.1f}\\\\")
lines.append("\\bottomrule\\end{tabular}\\\\[2pt]"
             "{\\scriptsize $^{\\ast}$numerically equivalent released "
             "implementation (deviation $<10^{-12}$\\,m; "
             "\\S\\ref{sec:q4}); all reported results use the reference "
             "implementation.}\\end{table}")
open(f"{OUT}/cost.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: World A noise types
wa2 = load("worldA_results.csv")
lines = ["\\begin{table}[t]\\centering",
         "\\caption{World-A noise-type variants at $\\sigma{=}1.0\\ell$ "
         "(mean over 3 seeds): overall point-to-surface RMSE improvement "
         "(\\%) under Laplace noise and $5\\%$ uniform outliers.}",
         "\\label{tab:worldA_noise}",
         "\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llccccc}", "\\toprule",
         "Shape & Noise & Bilat. & MLS & GLR & WF-F. & \\method{}\\\\",
         "\\midrule"]
for sh in ["plane", "ridge90"]:
    for kind in ["laplace", "outlier"]:
        cells = []
        vals = {}
        for m in ["Bilateral", "MLS", "GLR", "Prior-v1", "Prior-v2"]:
            sel = [r for r in wa2 if r["shape"] == sh and r["noise"] == kind
                   and r["method"] == m]
            vals[m] = np.mean([float(r["overall"]) for r in sel])
        bm = max(vals, key=vals.get)
        for m in ["Bilateral", "MLS", "GLR", "Prior-v1", "Prior-v2"]:
            txt = f"{vals[m]:+.1f}"
            cells.append(f"\\textbf{{{txt}}}" if m == bm else txt)
        kname = "Laplace" if kind == "laplace" else "$5\\%$ outliers"
        lines.append(f"{sh} & {kname} & " + " & ".join(cells) + "\\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{OUT}/worldA_noise.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: sensitivity
try:
    sv = load("sensitivity_v2.csv")
    FROZEN = {"lam_n": 12.5, "anchor": 1000.0, "cap_sigma": 2.0,
              "prop_decay": 0.75, "k": 12}
    PN = {"lam_n": "$\\lambda$", "anchor": "$a$", "cap_sigma": "$c_\\sigma$",
          "prop_decay": "buffer decay", "k": "$k$"}
    lines = ["\\begin{table}[t]\\centering",
             "\\caption{Sensitivity of the global configuration "
             "(two development crops, $\\sigma{=}5$\\,cm): overall Z and "
             "boundary improvement (\\%) as each parameter varies around "
             "its frozen value (marked $^*$), others fixed.}",
             "\\label{tab:sensitivity}",
             "\\small\\begin{tabular}{llcc}", "\\toprule",
             "Param & Value & Z & Bdry\\\\", "\\midrule"]
    prev = None
    for r in sv:
        if prev is not None and r["param"] != prev:
            lines.append("\\midrule")
        prev = r["param"]
        star = "$^*$" if abs(float(r["value"]) - FROZEN[r["param"]]) < 1e-9 else ""
        val = f"{float(r['value']):g}{star}"
        lines.append(f"{PN[r['param']]} & {val} & {float(r['Z']):+.1f} & "
                     f"{float(r['Bdry']):+.1f}\\\\")
    lines.append("\\bottomrule\\end{tabular}\\end{table}")
    open(f"{OUT}/sensitivity.tex", "w").write("\n".join(lines))
except FileNotFoundError:
    print("sensitivity_v2.csv not ready — skip")
print("cost/noise/sensitivity tables written")
