#!/usr/bin/env python3
"""
Generate a sky130A process stack JSON from an official skywater-pdk checkout.

This reads:
  - docs/_static/metal_stack.ps
  - docs/rules/rcx/capacitance-parallel.tsv
  - docs/rules/rcx/capacitance-fringe-upward.tsv

and writes a JSON compatible with ProcessStack.from_json().
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_layer_name(name: str) -> str:
    s = name.strip().lower()
    mapping = {
        "local interconnect": "li",
        "metal1": "met1",
        "metal2": "met2",
        "metal3": "met3",
        "metal4": "met4",
        "metal5": "met5",
        "poly": "poly",
    }
    return mapping.get(s, s)


def _parse_k_values_from_metal_stack(ps_text: str) -> dict:
    matches = re.findall(r"\(([A-Z0-9_]+)\s+K=([0-9.]+)\)", ps_text)
    if not matches:
        raise ValueError("Failed to parse dielectric K labels from metal_stack.ps")
    return {name: float(val) for name, val in matches}


def _require_ps_values(ps_text: str, values: list[str]) -> None:
    missing = []
    for value in values:
        if re.search(rf"\({re.escape(value)}\)", ps_text) is None:
            missing.append(value)
    if missing:
        raise ValueError(
            "metal_stack.ps is missing expected numeric markers: " + ", ".join(missing)
        )


def _parse_tsv_matrix(path: Path) -> dict:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty TSV file: {path}")

    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"Unexpected TSV header in: {path}")
    cols = [_normalize_layer_name(x) for x in header[1:]]

    matrix = {}
    for row in rows[1:]:
        if not row:
            continue
        row_name = _normalize_layer_name(row[0])
        for i, val in enumerate(row[1:]):
            sval = val.strip()
            if not sval:
                continue
            try:
                matrix[(row_name, cols[i])] = float(sval)
            except ValueError as exc:
                raise ValueError(f"Bad numeric value '{sval}' in {path}") from exc
    return matrix


def _git_commit_short(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_stack_json(pdk_root: Path) -> dict:
    ps_path = pdk_root / "docs" / "_static" / "metal_stack.ps"
    cap_par_path = pdk_root / "docs" / "rules" / "rcx" / "capacitance-parallel.tsv"
    cap_frg_up_path = (
        pdk_root / "docs" / "rules" / "rcx" / "capacitance-fringe-upward.tsv"
    )

    ps_text = _read_text(ps_path)
    k_map = _parse_k_values_from_metal_stack(ps_text)

    # Required values from the official Sky130A metal stack document.
    _require_ps_values(
        ps_text,
        [
            "0.12",
            "0.18",
            "0.1",
            "0.36",
            "0.845",
            "1.26",
            "0.4299",
            "0.27",
            "0.42",
            "0.39",
            "0.505",
            "0.3262",
            "0.9361",
            "1.3761",
            "2.0061",
            "2.7861",
            "4.0211",
            "5.3711",
        ],
    )

    # Bottom heights (um) from metal_stack.ps axis labels.
    z_poly = 0.3262
    z_li = 0.9361
    z_m1 = 1.3761
    z_m2 = 2.0061
    z_m3 = 2.7861
    z_m4 = 4.0211
    z_m5 = 5.3711

    # Thicknesses (um) from metal_stack.ps annotations.
    t_diff = 0.12
    t_poly = 0.18
    t_li = 0.10
    t_m1 = 0.36
    t_m2 = 0.36
    t_m3 = 0.845
    t_m4 = 0.845
    t_m5 = 1.26
    t_polycont = 0.4299
    t_via1 = 0.27
    t_via2 = 0.42
    t_via3 = 0.39
    t_via4 = 0.505
    # mcon/viali thickness derived from li top to metal1 bottom.
    t_viali = round(z_m1 - (z_li + t_li), 4)

    par = _parse_tsv_matrix(cap_par_path)
    frg = _parse_tsv_matrix(cap_frg_up_path)

    stack = {
        "name": "sky130A",
        "units": "centnm",
        "scale_to_um": 0.01,
        "substrate_epsilon_r": 11.7,
        "_source": (
            f"{pdk_root} "
            "(docs/_static/metal_stack.ps + docs/rules/rcx/*.tsv)"
        ),
        "_source_commit": _git_commit_short(pdk_root),
        "_generated_by": "generate_stack_from_pdk.py",
        "layers": {
            "nwell": {"name": "nwell", "z_bottom": 0.0, "thickness": 0.0, "is_via": False},
            "ndiff": {"name": "ndiff", "z_bottom": 0.0, "thickness": t_diff, "is_via": False},
            "pdiff": {"name": "pdiff", "z_bottom": 0.0, "thickness": t_diff, "is_via": False},
            "ndiffc": {"name": "ndiffc", "z_bottom": 0.0, "thickness": z_li, "is_via": True},
            "pdiffc": {"name": "pdiffc", "z_bottom": 0.0, "thickness": z_li, "is_via": True},
            "nsubdiff": {"name": "nsubdiff", "z_bottom": 0.0, "thickness": t_diff, "is_via": False},
            "nsubdiffcont": {"name": "nsubdiffcont", "z_bottom": 0.0, "thickness": z_li, "is_via": True},
            "poly": {"name": "poly", "z_bottom": z_poly, "thickness": t_poly, "is_via": False},
            "nmos": {"name": "nmos", "z_bottom": 0.0, "thickness": 0.0, "is_via": False},
            "pmos": {"name": "pmos", "z_bottom": 0.0, "thickness": 0.0, "is_via": False},
            "polycont": {
                "name": "polycont",
                "z_bottom": z_poly + t_poly,
                "thickness": t_polycont,
                "is_via": True,
            },
            "locali": {"name": "locali", "z_bottom": z_li, "thickness": t_li, "is_via": False},
            "viali": {"name": "viali", "z_bottom": z_li + t_li, "thickness": t_viali, "is_via": True},
            "metal1": {"name": "metal1", "z_bottom": z_m1, "thickness": t_m1, "is_via": False},
            "via1": {"name": "via1", "z_bottom": z_m1 + t_m1, "thickness": t_via1, "is_via": True},
            "metal2": {"name": "metal2", "z_bottom": z_m2, "thickness": t_m2, "is_via": False},
            "via2": {"name": "via2", "z_bottom": z_m2 + t_m2, "thickness": t_via2, "is_via": True},
            "metal3": {"name": "metal3", "z_bottom": z_m3, "thickness": t_m3, "is_via": False},
            "via3": {"name": "via3", "z_bottom": z_m3 + t_m3, "thickness": t_via3, "is_via": True},
            "metal4": {"name": "metal4", "z_bottom": z_m4, "thickness": t_m4, "is_via": False},
            "via4": {"name": "via4", "z_bottom": z_m4 + t_m4, "thickness": t_via4, "is_via": True},
            "metal5": {"name": "metal5", "z_bottom": z_m5, "thickness": t_m5, "is_via": False},
        },
        "dielectrics": [
            {"z_bottom": -1.0, "z_top": 0.0, "epsilon_r": 11.7, "_name": "Si substrate"},
            {"z_bottom": 0.0, "z_top": z_poly, "epsilon_r": k_map["FOX"], "_name": "FOX + PSG"},
            {"z_bottom": z_poly, "z_top": z_li, "epsilon_r": k_map["LINT"], "_name": "LINT (poly to li)"},
            {"z_bottom": z_li, "z_top": z_m1, "epsilon_r": k_map["NILD2"], "_name": "NILD2 (li to met1)"},
            {"z_bottom": z_m1, "z_top": z_m2, "epsilon_r": k_map["NILD3"], "_name": "NILD3 (met1 to met2)"},
            {"z_bottom": z_m2, "z_top": z_m3, "epsilon_r": k_map["NILD4"], "_name": "NILD4 (met2 to met3)"},
            {"z_bottom": z_m3, "z_top": z_m4, "epsilon_r": k_map["NILD5"], "_name": "NILD5 (met3 to met4)"},
            {"z_bottom": z_m4, "z_top": z_m5, "epsilon_r": k_map["NILD6"], "_name": "NILD6 (met4 to met5)"},
            {"z_bottom": z_m5, "z_top": 15.0, "epsilon_r": k_map["TOPOX"], "_name": "TOPOX + air"},
        ],
        "parallel_plate_caps_aF_per_um2": {
            "_source": str(cap_par_path),
            "poly_to_li": par[("poly", "li")],
            "poly_to_met1": par[("poly", "met1")],
            "li_to_met1": par[("li", "met1")],
            "met1_to_met2": par[("met1", "met2")],
            "met2_to_met3": par[("met2", "met3")],
            "met3_to_met4": par[("met3", "met4")],
            "met4_to_met5": par[("met4", "met5")],
        },
        "fringe_caps_aF_per_um": {
            "_source": str(cap_frg_up_path),
            "poly_to_li_up": frg[("poly", "li")],
            "li_to_met1_up": frg[("li", "met1")],
            "met1_to_met2_up": frg[("met1", "met2")],
            "met2_to_met3_up": frg[("met2", "met3")],
            "met3_to_met4_up": frg[("met3", "met4")],
            "met4_to_met5_up": frg[("met4", "met5")],
        },
    }
    return stack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sky130A stack JSON from official skywater-pdk docs"
    )
    parser.add_argument(
        "--pdk-root",
        default="/home/paiktj/skywater-pdk",
        help="Path to skywater-pdk repository root",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="sky130a_stack_from_pdk.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    pdk_root = Path(args.pdk_root).resolve()
    output = Path(args.output).resolve()

    data = build_stack_json(pdk_root)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
