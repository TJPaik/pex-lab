"""
Export 3D conductor geometry to FasterCap input format.

Generates:
  - One .qui file per net (quad panels for conductor surfaces)
  - One .lst file referencing all .qui files with dielectric info
  - Optional CSV output from FasterCap solved matrix

Coordinates are written in micrometers (um). FasterCap is unit-agnostic,
so we scale the solved matrix by 1e9 to convert to fF.
"""

import csv
import os
import re
import numpy as np
from typing import Dict, List, Tuple

from polygon_parser import Rect, NetGeometry, parse_polygons
from process_stack import ProcessStack, LayerInfo

# Layers to skip (not physical conductors or zero thickness)
SKIP_LAYERS = {"nwell", "nmos", "pmos"}


def rect_to_quads(rect: Rect, layer: LayerInfo, scale: float,
                  net_name: str, max_size: float) -> List[str]:
    """Convert a rectangle + layer info into FasterCap Q lines.

    Each quad is: Q <name> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
    Coordinates in meters.
    """
    # Use micrometers as the coordinate unit.
    # FasterCap is unit-agnostic; capacitance scales with geometry.
    # Output will be in Farads assuming meters, so we must convert
    # the result: C_actual = C_fastercap * 1e-6  (um -> m factor)
    x1 = rect.x1 * scale
    y1 = rect.y1 * scale
    x2 = rect.x2 * scale
    y2 = rect.y2 * scale
    z_bot = layer.z_bottom
    z_top = layer.z_top

    lines = []

    def subdivide_face(corners, name):
        """Subdivide a rectangular face into smaller quads if needed."""
        # corners: list of 4 (x,y,z) tuples defining the quad
        # For simplicity, output the full quad - FasterCap auto-refines
        c = corners
        lines.append(
            f"Q {name}  "
            f"{c[0][0]:.10e} {c[0][1]:.10e} {c[0][2]:.10e}  "
            f"{c[1][0]:.10e} {c[1][1]:.10e} {c[1][2]:.10e}  "
            f"{c[2][0]:.10e} {c[2][1]:.10e} {c[2][2]:.10e}  "
            f"{c[3][0]:.10e} {c[3][1]:.10e} {c[3][2]:.10e}"
        )

    name = net_name

    # Top surface (z = z_top, normal +z)
    if not layer.is_via:
        subdivide_face([
            (x1, y1, z_top), (x2, y1, z_top),
            (x2, y2, z_top), (x1, y2, z_top)
        ], name)
        # Bottom surface (z = z_bot, normal -z)
        subdivide_face([
            (x1, y2, z_bot), (x2, y2, z_bot),
            (x2, y1, z_bot), (x1, y1, z_bot)
        ], name)

    # Four sidewalls
    # Front wall (y = y2, normal +y)
    subdivide_face([
        (x1, y2, z_bot), (x2, y2, z_bot),
        (x2, y2, z_top), (x1, y2, z_top)
    ], name)
    # Back wall (y = y1, normal -y)
    subdivide_face([
        (x2, y1, z_bot), (x1, y1, z_bot),
        (x1, y1, z_top), (x2, y1, z_top)
    ], name)
    # Right wall (x = x2, normal +x)
    subdivide_face([
        (x2, y1, z_bot), (x2, y2, z_bot),
        (x2, y2, z_top), (x2, y1, z_top)
    ], name)
    # Left wall (x = x1, normal -x)
    subdivide_face([
        (x1, y2, z_bot), (x1, y1, z_bot),
        (x1, y1, z_top), (x1, y2, z_top)
    ], name)

    return lines


def _panel_key(corners):
    """Canonical key for a quad panel (order-independent)."""
    return tuple(sorted(
        (round(c[0], 4), round(c[1], 4), round(c[2], 4)) for c in corners
    ))


def _collect_panels_for_net(net: NetGeometry, stack: ProcessStack,
                            net_name: str) -> List[str]:
    """Collect all panels, remove internal faces (shared between adjacent boxes)."""
    from collections import Counter

    scale = stack.scale_to_um
    all_panels = []  # (key, corners_list)

    for rect in net.rects:
        if rect.layer in SKIP_LAYERS:
            continue
        layer_info = stack.get_layer(rect.layer)
        if layer_info is None or layer_info.thickness <= 0:
            continue

        x1 = rect.x1 * scale
        y1 = rect.y1 * scale
        x2 = rect.x2 * scale
        y2 = rect.y2 * scale
        z_bot = layer_info.z_bottom
        z_top = layer_info.z_top

        w = abs(x2 - x1)
        h = abs(y2 - y1)
        t = z_top - z_bot

        if w < 1e-4 and h < 1e-4:
            continue

        if not layer_info.is_via and w > 1e-4 and h > 1e-4:
            corners = [(x1, y1, z_top), (x2, y1, z_top),
                       (x2, y2, z_top), (x1, y2, z_top)]
            all_panels.append((_panel_key(corners), corners))
            corners = [(x1, y2, z_bot), (x2, y2, z_bot),
                       (x2, y1, z_bot), (x1, y1, z_bot)]
            all_panels.append((_panel_key(corners), corners))

        if t < 1e-4:
            continue

        if h > 1e-4:
            corners = [(x1, y2, z_bot), (x2, y2, z_bot),
                       (x2, y2, z_top), (x1, y2, z_top)]
            all_panels.append((_panel_key(corners), corners))
            corners = [(x2, y1, z_bot), (x1, y1, z_bot),
                       (x1, y1, z_top), (x2, y1, z_top)]
            all_panels.append((_panel_key(corners), corners))

        if w > 1e-4:
            corners = [(x2, y1, z_bot), (x2, y2, z_bot),
                       (x2, y2, z_top), (x2, y1, z_top)]
            all_panels.append((_panel_key(corners), corners))
            corners = [(x1, y2, z_bot), (x1, y1, z_bot),
                       (x1, y1, z_top), (x1, y2, z_top)]
            all_panels.append((_panel_key(corners), corners))

    # Remove internal faces (panels appearing more than once)
    key_counts = Counter(key for key, _ in all_panels)
    lines = []
    seen = set()
    for key, corners in all_panels:
        if key_counts[key] == 1 and key not in seen:
            c = corners
            lines.append(
                f"Q {net_name}  "
                f"{c[0][0]:.10e} {c[0][1]:.10e} {c[0][2]:.10e}  "
                f"{c[1][0]:.10e} {c[1][1]:.10e} {c[1][2]:.10e}  "
                f"{c[2][0]:.10e} {c[2][1]:.10e} {c[2][2]:.10e}  "
                f"{c[3][0]:.10e} {c[3][1]:.10e} {c[3][2]:.10e}"
            )
            seen.add(key)

    internal = sum(1 for c in key_counts.values() if c > 1)
    return lines, len(lines), internal


def export_fastercap(nets: Dict[str, NetGeometry], stack: ProcessStack,
                     output_dir: str, max_size: float = 1.0) -> str:
    """Export all nets to FasterCap format.

    Returns the path to the .lst file.
    """
    os.makedirs(output_dir, exist_ok=True)

    net_names = sorted(nets.keys())
    qui_files = []

    for net_name in net_names:
        net = nets[net_name]
        panel_lines, panel_count, internal_count = _collect_panels_for_net(
            net, stack, net_name)

        if panel_count == 0:
            continue

        lines = [f"0 FasterCap geometry for net: {net_name} "
                 f"({panel_count} panels, {internal_count} internal removed)"]
        lines.extend(panel_lines)

        qui_path = os.path.join(output_dir, f"{net_name}.qui")
        with open(qui_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        qui_files.append((net_name, qui_path))
        print(f"  {net_name}: {panel_count} panels ({internal_count} internal removed)")

    # Find the dominant dielectric permittivity (weighted average or use
    # the dielectric at metal1 height as representative)
    eps_r = stack.get_effective_epsilon(1.5)  # around metal1 height

    # Write .lst file
    lst_path = os.path.join(output_dir, "input.lst")
    with open(lst_path, 'w') as f:
        f.write(f"* FasterCap input for {stack.name} process\n")
        f.write(f"* Dielectric permittivity: {eps_r}\n")
        for net_name, qui_path in qui_files:
            fname = os.path.basename(qui_path)
            # C <file> <outer_perm> <tx> <ty> <tz>
            f.write(f"C {fname} {eps_r} 0 0 0\n")

    print(f"FasterCap export: {len(qui_files)} nets, lst file: {lst_path}")
    return lst_path


def run_fastercap(lst_path: str, fastercap_bin: str,
                  accuracy: float = 0.01,
                  timeout_s: int = 600,
                  use_galerkin: bool = False) -> str:
    """Run FasterCap and return the output text."""
    import subprocess
    cmd = [
        fastercap_bin,
        "-b",  # batch/console mode
        f"-a{accuracy}",
    ]
    if use_galerkin:
        cmd.append("-g")
    cmd.append(lst_path)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=os.path.dirname(lst_path),
                           timeout=timeout_s)
    return result.stdout + result.stderr


def parse_fastercap_output(output_text: str) -> Dict[Tuple[str, str], float]:
    """Parse FasterCap output to extract capacitance matrix.

    Returns dict of (net1, net2) -> capacitance in fF.
    """
    names, matrix_fF = parse_fastercap_matrix(output_text)
    caps: Dict[Tuple[str, str], float] = {}
    if not names:
        return caps
    n = len(names)
    for i in range(n):
        for j in range(n):
            caps[(names[i], names[j])] = float(matrix_fF[i, j])
    return caps


def _strip_group_prefix(name: str) -> str:
    """Convert FasterCap names like g1_VDD -> VDD."""
    m = re.match(r"^g\d+_(.+)$", name)
    return m.group(1) if m else name


def parse_fastercap_matrix(output_text: str) -> Tuple[List[str], np.ndarray]:
    """Parse the final FasterCap matrix block.

    Returns:
        names: conductor names (group prefix removed)
        matrix_fF: NxN Maxwell matrix in fF
    """
    lines = output_text.splitlines()
    dim_re = re.compile(r"^Dimension\s+(\d+)\s+x\s+(\d+)")

    last_names: List[str] = []
    last_matrix: np.ndarray | None = None

    i = 0
    while i < len(lines):
        if "Capacitance matrix is:" not in lines[i]:
            i += 1
            continue

        j = i + 1
        dim = None
        while j < len(lines):
            m = dim_re.match(lines[j].strip())
            if m:
                nrow = int(m.group(1))
                ncol = int(m.group(2))
                if nrow == ncol and nrow > 0:
                    dim = nrow
                break
            j += 1
        if dim is None:
            i += 1
            continue

        names: List[str] = []
        rows: List[List[float]] = []
        k = j + 1
        while k < len(lines) and len(rows) < dim:
            row = lines[k].strip()
            if not row:
                k += 1
                continue
            parts = row.split()
            if len(parts) < dim + 1:
                break
            name = _strip_group_prefix(parts[0])
            try:
                vals = [float(x) for x in parts[1:dim + 1]]
            except ValueError:
                break
            names.append(name)
            rows.append(vals)
            k += 1

        if len(rows) == dim:
            # Geometry is in um, FasterCap numerics assume meter scale.
            # Convert to fF: *1e-6 (um->m) *1e15 (F->fF) => *1e9
            last_names = names
            last_matrix = np.array(rows, dtype=float) * 1e9
            i = k
        else:
            i += 1

    if last_matrix is None:
        return [], np.zeros((0, 0))
    return last_names, last_matrix


def matrix_to_coupling_rows(names: List[str], matrix_fF: np.ndarray,
                            min_cap_fF: float = 1e-6) -> List[Tuple[str, str, float]]:
    """Convert Maxwell matrix to net-pair coupling rows."""
    rows: List[Tuple[str, str, float]] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            coupling = -float(matrix_fF[i, j])
            if coupling > min_cap_fF:
                rows.append((names[i], names[j], coupling))
    for i in range(n):
        c_gnd = float(matrix_fF[i, i] + np.sum(matrix_fF[i, :]) - matrix_fF[i, i])
        if c_gnd > min_cap_fF:
            rows.append((names[i], "GND", c_gnd))
    return rows


def write_caps_csv(rows: List[Tuple[str, str, float]], csv_path: str) -> None:
    """Write net coupling rows to CSV."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["net1", "net2", "coupling_cap_fF"])
        for n1, n2, cap in rows:
            w.writerow([n1, n2, f"{cap:.5f}"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export to FasterCap and optionally run"
    )
    parser.add_argument("input", help="polygons.txt or GDS with --from-gds")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for FasterCap files")
    parser.add_argument("--stack", default=None)
    parser.add_argument("--from-gds", action="store_true")
    parser.add_argument("--polygons-out", default=None)
    parser.add_argument("--panel-size", type=float, default=1.0)
    parser.add_argument("--run", action="store_true",
                        help="Run FasterCap after export")
    parser.add_argument("--fastercap-bin",
                        default="/home/paiktj/FasterCap/build/FasterCap")
    parser.add_argument("--accuracy", type=float, default=0.01)
    parser.add_argument("--timeout", type=int, default=600,
                        help="FasterCap subprocess timeout in seconds")
    parser.add_argument("--galerkin", action="store_true",
                        help="Use FasterCap Galerkin mode (-g), slower but more accurate")
    parser.add_argument("--csv-out", default=None,
                        help="Optional output CSV path for coupling caps")
    parser.add_argument("--min-cap", type=float, default=1e-6,
                        help="Minimum cap (fF) to include in CSV/output")
    args = parser.parse_args()

    # Load input
    if args.from_gds:
        from gds_to_polygons import GDSToPolygons
        polygons_path = args.polygons_out or args.input.replace(
            '.gds', '_from_gds_polygons.txt')
        converter = GDSToPolygons(args.input)
        converter.run(polygons_path)
    else:
        polygons_path = args.input

    nets = parse_polygons(polygons_path)
    print(f"Parsed {len(nets)} nets")

    if args.stack:
        stack = ProcessStack.from_json(args.stack)
    else:
        from process_stack import default_sky130a_stack
        stack = default_sky130a_stack()

    lst_path = export_fastercap(nets, stack, args.output_dir, args.panel_size)

    if args.run:
        output = run_fastercap(
            lst_path,
            args.fastercap_bin,
            args.accuracy,
            args.timeout,
            args.galerkin,
        )
        print(output)
        names, matrix_fF = parse_fastercap_matrix(output)
        if not names:
            print("Warning: failed to parse FasterCap matrix from output")
        else:
            rows = matrix_to_coupling_rows(names, matrix_fF, args.min_cap)
            for n1, n2, cap in rows:
                print(f"  {n1} <-> {n2}: {cap:.5f} fF")
            if args.csv_out:
                csv_path = args.csv_out
            else:
                csv_path = os.path.join(args.output_dir, "fastercap_output.csv")
            write_caps_csv(rows, csv_path)
            print(f"Wrote parsed FasterCap CSV: {csv_path}")
