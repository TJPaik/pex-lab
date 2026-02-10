#!/usr/bin/env python3
"""
Compare FasterCap reference CSV and Python BEM prediction CSV.

Inputs are CSV files with columns:
  net1,net2,coupling_cap_fF

Outputs:
  - pairwise comparison CSV
  - summary text report
  - optional scatter plot PNG (if matplotlib is available)
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


def canonical_pair(n1: str, n2: str) -> tuple[str, str]:
    return (n1, n2) if n1 <= n2 else (n2, n1)


def read_caps(path: Path) -> dict[tuple[str, str], float]:
    caps: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n1 = row["net1"].strip()
            n2 = row["net2"].strip()
            c = float(row["coupling_cap_fF"])
            caps[canonical_pair(n1, n2)] = c
    return caps


def compute_metrics(fc_vals: list[float], bem_vals: list[float]) -> dict[str, float]:
    if not fc_vals:
        return {
            "count": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "medape": math.nan,
            "corr": math.nan,
        }

    abs_err = [abs(b - f) for f, b in zip(fc_vals, bem_vals)]
    rel_err = [abs(b - f) / f * 100.0 if f > 1e-18 else 0.0 for f, b in zip(fc_vals, bem_vals)]
    mae = statistics.mean(abs_err)
    rmse = math.sqrt(statistics.mean([x * x for x in abs_err]))
    mape = statistics.mean(rel_err)
    medape = statistics.median(rel_err)

    corr = math.nan
    if len(fc_vals) >= 2:
        mean_fc = statistics.mean(fc_vals)
        mean_bem = statistics.mean(bem_vals)
        num = sum((x - mean_fc) * (y - mean_bem) for x, y in zip(fc_vals, bem_vals))
        den1 = math.sqrt(sum((x - mean_fc) ** 2 for x in fc_vals))
        den2 = math.sqrt(sum((y - mean_bem) ** 2 for y in bem_vals))
        if den1 > 0 and den2 > 0:
            corr = num / (den1 * den2)

    return {
        "count": len(fc_vals),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "medape": medape,
        "corr": corr,
    }


def try_make_scatter(fc_vals: list[float], bem_vals: list[float],
                     is_gnd: list[bool], out_png: Path,
                     lowcap_threshold: float = 0.5) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib unavailable; skipped scatter plot"

    if not fc_vals:
        return "no common pairs; skipped scatter plot"

    mn = min(min(fc_vals), min(bem_vals))
    mx = max(max(fc_vals), max(bem_vals))
    if mn <= 0:
        mn = 1e-6

    low_thr = max(1e-6, float(lowcap_threshold))
    low_idx = [i for i, x in enumerate(fc_vals) if x <= low_thr]
    low_fc = [fc_vals[i] for i in low_idx]
    low_bem = [bem_vals[i] for i in low_idx]
    low_gnd = [is_gnd[i] for i in low_idx]

    fc_sig = [x for x, g in zip(fc_vals, is_gnd) if not g]
    bem_sig = [y for y, g in zip(bem_vals, is_gnd) if not g]
    fc_gnd = [x for x, g in zip(fc_vals, is_gnd) if g]
    bem_gnd = [y for y, g in zip(bem_vals, is_gnd) if g]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # Left: full-range log-log scatter.
    ax = axes[0]
    if fc_sig:
        ax.scatter(fc_sig, bem_sig, s=24, alpha=0.78, color="#1f77b4",
                   label="Signal pairs")
    if fc_gnd:
        ax.scatter(fc_gnd, bem_gnd, s=34, alpha=0.88, color="#d62728",
                   marker="x", label="GND pairs")
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FasterCap (fF)")
    ax.set_ylabel("Python BEM (fF)")
    ax.set_title("Full Scatter (log-log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()

    # Right: low-cap zoom in linear scale.
    ax = axes[1]
    low_fc_sig = [x for x, g in zip(low_fc, low_gnd) if not g]
    low_bem_sig = [y for y, g in zip(low_bem, low_gnd) if not g]
    low_fc_gnd = [x for x, g in zip(low_fc, low_gnd) if g]
    low_bem_gnd = [y for y, g in zip(low_bem, low_gnd) if g]
    if low_fc_sig:
        ax.scatter(low_fc_sig, low_bem_sig, s=26, alpha=0.82, color="#1f77b4",
                   label="Signal pairs")
    if low_fc_gnd:
        ax.scatter(low_fc_gnd, low_bem_gnd, s=36, alpha=0.9, color="#d62728",
                   marker="x", label="GND pairs")
    ax.plot([0.0, low_thr], [0.0, low_thr], "r--", linewidth=1.2, label="y=x")
    ax.set_xlim(0.0, low_thr)
    ax.set_ylim(0.0, low_thr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("FasterCap (fF)")
    ax.set_ylabel("Python BEM (fF)")
    ax.set_title(
        f"Low-Cap Zoom (FC <= {low_thr:g} fF, n={len(low_fc)}, "
        f"GND={sum(low_gnd)})"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.suptitle("FasterCap vs Python BEM")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return f"wrote scatter: {out_png}"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare FasterCap and Python BEM CSV outputs")
    p.add_argument("--fastercap", required=True, help="FasterCap reference CSV")
    p.add_argument("--bem", required=True, help="Python BEM prediction CSV")
    p.add_argument("--out-csv", required=True, help="Pairwise comparison CSV output")
    p.add_argument("--summary", required=True, help="Summary text output")
    p.add_argument("--scatter", default=None,
                   help="Scatter PNG output (default: derived from --out-csv)")
    p.add_argument("--lowcap-threshold", type=float, default=0.5,
                   help="Low-cap zoom threshold in fF (default: 0.5)")
    args = p.parse_args()

    fc_path = Path(args.fastercap)
    bem_path = Path(args.bem)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.summary)
    out_scatter = Path(args.scatter) if args.scatter else out_csv.with_suffix(".png")

    fc = read_caps(fc_path)
    bem = read_caps(bem_path)

    common = sorted(set(fc.keys()) & set(bem.keys()))
    fc_only = sorted(set(fc.keys()) - set(bem.keys()))
    bem_only = sorted(set(bem.keys()) - set(fc.keys()))

    rows: list[tuple[str, str, float, float, float, float]] = []
    fc_vals: list[float] = []
    bem_vals: list[float] = []
    is_gnd: list[bool] = []
    fc_vals_sig: list[float] = []
    bem_vals_sig: list[float] = []

    for pair in common:
        f = fc[pair]
        b = bem[pair]
        d = b - f
        a = abs(d)
        r = (a / f * 100.0) if f > 1e-18 else 0.0
        rows.append((pair[0], pair[1], f, b, d, r))
        fc_vals.append(f)
        bem_vals.append(b)
        gnd_pair = "GND" in pair
        is_gnd.append(gnd_pair)
        if not gnd_pair:
            fc_vals_sig.append(f)
            bem_vals_sig.append(b)

    rows.sort(key=lambda x: abs(x[4]), reverse=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "net1",
                "net2",
                "fastercap_fF",
                "bem_fF",
                "delta_bem_minus_fc_fF",
                "abs_rel_error_percent",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r[0],
                    r[1],
                    f"{r[2]:.6f}",
                    f"{r[3]:.6f}",
                    f"{r[4]:.6f}",
                    f"{r[5]:.4f}",
                ]
            )

    m_all = compute_metrics(fc_vals, bem_vals)
    m_sig = compute_metrics(fc_vals_sig, bem_vals_sig)

    scatter_note = try_make_scatter(
        fc_vals, bem_vals, is_gnd, out_scatter, lowcap_threshold=args.lowcap_threshold
    )

    top_lines = []
    for r in rows[:10]:
        top_lines.append(
            f"  {r[0]}-{r[1]}: FC={r[2]:.5f} fF, BEM={r[3]:.5f} fF, "
            f"delta={r[4]:+.5f} fF, rel={r[5]:.2f}%"
        )

    with out_summary.open("w", encoding="utf-8") as f:
        f.write(f"FasterCap file: {fc_path}\n")
        f.write(f"Python BEM file: {bem_path}\n")
        f.write(f"Common pairs: {len(common)}\n")
        f.write(f"FasterCap-only pairs: {len(fc_only)}\n")
        f.write(f"BEM-only pairs: {len(bem_only)}\n")
        if fc_only:
            f.write("FasterCap-only list: " + ", ".join(f"{a}-{b}" for a, b in fc_only) + "\n")
        if bem_only:
            f.write("BEM-only list: " + ", ".join(f"{a}-{b}" for a, b in bem_only) + "\n")
        f.write("All-pair metrics (including GND):\n")
        f.write(f"  MAE (fF): {m_all['mae']:.6f}\n")
        f.write(f"  RMSE (fF): {m_all['rmse']:.6f}\n")
        f.write(f"  MAPE (%): {m_all['mape']:.4f}\n")
        f.write(f"  Median APE (%): {m_all['medape']:.4f}\n")
        f.write(f"  Pearson corr: {m_all['corr']:.6f}\n")
        f.write("Signal-only metrics (exclude any pair containing GND):\n")
        f.write(f"  Count: {m_sig['count']}\n")
        f.write(f"  MAE (fF): {m_sig['mae']:.6f}\n")
        f.write(f"  RMSE (fF): {m_sig['rmse']:.6f}\n")
        f.write(f"  MAPE (%): {m_sig['mape']:.4f}\n")
        f.write(f"  Median APE (%): {m_sig['medape']:.4f}\n")
        f.write(f"  Pearson corr: {m_sig['corr']:.6f}\n")
        f.write(f"{scatter_note}\n")
        f.write("\nTop |delta| pairs:\n")
        for line in top_lines:
            f.write(line + "\n")

    print(f"Wrote comparison CSV: {out_csv}")
    print(f"Wrote summary: {out_summary}")
    print(scatter_note)


if __name__ == "__main__":
    main()
