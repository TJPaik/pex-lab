"""
Parasitic capacitance extraction using BEM (Boundary Element Method).

Main CLI that orchestrates the full pipeline:
  polygons.txt -> parse -> mesh -> BEM solve -> output CSV

Usage:
    python cap_extract.py OTA_FF_992_0_polygons.txt -o output.csv
    python cap_extract.py OTA_FF_992_0_polygons.txt -o output.csv --stack sky130a_stack.json
    python cap_extract.py --from-gds layout.gds -o output.csv
    python cap_extract.py OTA_FF_992_0_polygons.txt -o output.csv --panel-size 0.5
"""

import argparse
import csv
import sys
import time
from typing import Dict, List, Tuple

from polygon_parser import NetGeometry, Rect, parse_polygons
from process_stack import LayerInfo, ProcessStack, default_sky130a_stack
from mesh import SKIP_LAYERS, mesh_all_nets
from bem_solver import BEMSolver


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parasitic capacitance extraction using BEM"
    )
    parser.add_argument("input", help="Input polygons.txt (or GDS with --from-gds)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output CSV file path")
    parser.add_argument("--stack", default=None,
                        help="Process stack JSON file (default: sky130A)")
    parser.add_argument("--from-gds", action="store_true",
                        help="Input is a GDS file; convert first")
    parser.add_argument("--polygons-out", default=None,
                        help="Output path for polygons.txt (with --from-gds)")
    parser.add_argument("--panel-size", type=float, default=1.0,
                        help="Max panel edge length in um (default: 1.0)")
    parser.add_argument("--min-panel-size", type=float, default=0.2,
                        help="Minimum adaptive panel size in um (default: 0.2)")
    parser.add_argument("--adaptive-mesh", action="store_true",
                        help="Enable proximity-aware adaptive panel sizing")
    parser.add_argument("--proximity-distance", type=float, default=2.0,
                        help="Distance threshold (um) for adaptive refinement")
    parser.add_argument("--proximity-factor", type=float, default=0.6,
                        help="Near-region panel scaling factor in adaptive mesh")
    parser.add_argument("--edge-refine-factor", type=float, default=1.0,
                        help="Edge panel size scale (0<factor<=1, default: 1.0)")
    parser.add_argument("--edge-refine-fraction", type=float, default=0.0,
                        help="Fraction of each face edge to refine (0~0.49, default: 0)")
    parser.add_argument("--remove-internal-faces", action="store_true",
                        help="Remove shared internal faces within each net")
    parser.add_argument("--no-ground-plane", action="store_true",
                        help="Disable substrate ground plane image")
    parser.add_argument("--near-field-factor", type=float, default=0.0,
                        help="Enable near-field kernel refinement for close panels")
    parser.add_argument("--near-field-samples", type=int, default=3,
                        help="Subsamples per panel axis for near-field integration")
    parser.add_argument("--uniform-epsilon", type=float, default=None,
                        help="Use one global epsilon_r for all panel interactions")
    parser.add_argument("--ground-net", default="GND",
                        help="Name used for substrate ground output rows (default: GND)")
    parser.add_argument("--signal-scale", type=float, default=1.0,
                        help="Scale factor applied to net-to-net coupling rows")
    parser.add_argument("--ground-scale", type=float, default=1.0,
                        help="Scale factor applied only to ground-cap output rows")
    parser.add_argument("--ground-model", choices=["analytic", "matrix", "both"],
                        default="analytic",
                        help="Ground cap model: analytic, matrix, or both")
    parser.add_argument("--match-fastercap", action="store_true",
                        help="Apply settings that better match FasterCap export assumptions")
    parser.add_argument("--explicit-ground-plane", action="store_true",
                        help="Add an explicit GND conductor plane from layout bbox")
    parser.add_argument("--ground-plane-layer", default="nsubdiff",
                        help="Layer name used for explicit GND plane (default: nsubdiff)")
    parser.add_argument("--ground-plane-z-bottom", type=float, default=None,
                        help="If set, create/override explicit GND layer at this z (um)")
    parser.add_argument("--ground-plane-thickness", type=float, default=0.2,
                        help="Thickness (um) for explicit GND layer when z-bottom is set")
    parser.add_argument("--ground-plane-margin", type=float, default=15.0,
                        help="Margin (um) added around bbox for explicit GND plane")
    parser.add_argument("--ground-plane-panel-size", type=float, default=None,
                        help="Optional fixed panel size (um) only for explicit GND plane")
    parser.add_argument("--min-cap", type=float, default=1e-6,
                        help="Minimum capacitance (fF) to keep in output")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress info")
    return parser.parse_args()


def write_csv(results: List[Tuple[str, str, float]], output_path: str):
    """Write coupling capacitances to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["net1", "net2", "coupling_cap_fF"])
        for net1, net2, cap in results:
            writer.writerow([net1, net2, f"{cap:.5f}"])


def add_explicit_ground_plane(nets: Dict[str, NetGeometry], stack: ProcessStack,
                              ground_net: str, ground_layer: str,
                              margin_um: float,
                              ground_z_bottom: float = None,
                              ground_thickness: float = 0.2) -> bool:
    """Insert one bbox-based GND rectangle as a physical conductor."""
    if ground_z_bottom is not None:
        stack.layers[ground_layer] = LayerInfo(
            name=ground_layer,
            z_bottom=float(ground_z_bottom),
            thickness=max(1e-6, float(ground_thickness)),
            is_via=False,
        )
    if ground_layer not in stack.layers:
        raise ValueError(
            f"ground-plane layer '{ground_layer}' not found in process stack"
        )

    min_x = None
    min_y = None
    max_x = None
    max_y = None
    for net_name, net in nets.items():
        if net_name == ground_net:
            continue
        for rect in net.rects:
            if rect.layer in SKIP_LAYERS:
                continue
            layer = stack.get_layer(rect.layer)
            if layer is None or layer.thickness <= 0:
                continue
            if min_x is None or rect.x1 < min_x:
                min_x = rect.x1
            if min_y is None or rect.y1 < min_y:
                min_y = rect.y1
            if max_x is None or rect.x2 > max_x:
                max_x = rect.x2
            if max_y is None or rect.y2 > max_y:
                max_y = rect.y2

    if min_x is None or min_y is None or max_x is None or max_y is None:
        return False

    margin_units = int(round(max(0.0, margin_um) / stack.scale_to_um))
    gnd_rect = Rect(
        x1=min_x - margin_units,
        y1=min_y - margin_units,
        x2=max_x + margin_units,
        y2=max_y + margin_units,
        layer=ground_layer,
    )

    if gnd_rect.x2 <= gnd_rect.x1 or gnd_rect.y2 <= gnd_rect.y1:
        return False

    if ground_net in nets:
        nets[ground_net].rects.append(gnd_rect)
    else:
        nets[ground_net] = NetGeometry(name=ground_net, rects=[gnd_rect])
    return True


def main():
    args = parse_args()
    t0 = time.time()

    # Step 1: Load or convert input
    if args.from_gds:
        from gds_to_polygons import GDSToPolygons
        polygons_path = args.polygons_out or args.input.replace('.gds', '_from_gds_polygons.txt')
        converter = GDSToPolygons(args.input)
        converter.run(polygons_path)
        print(f"Converted GDS -> {polygons_path}")
    else:
        polygons_path = args.input

    # Step 2: Parse polygons
    nets = parse_polygons(polygons_path)
    total_rects = sum(len(n.rects) for n in nets.values())
    print(f"Parsed {len(nets)} nets, {total_rects} rectangles")

    # Step 3: Load process stack
    if args.stack:
        stack = ProcessStack.from_json(args.stack)
    else:
        stack = default_sky130a_stack()
    print(f"Process stack: {stack.name}")

    if args.match_fastercap:
        args.no_ground_plane = True
        args.ground_model = "matrix"
        args.remove_internal_faces = True
        if args.uniform_epsilon is None:
            args.uniform_epsilon = stack.get_effective_epsilon(1.5)
        print(
            "Applied --match-fastercap: "
            "no_ground_plane=True, ground_model=matrix, "
            "remove_internal_faces=True, "
            f"uniform_epsilon={args.uniform_epsilon:.4f}"
        )

    added_explicit_ground = False
    net_max_panel_size = None
    if args.explicit_ground_plane:
        added_explicit_ground = add_explicit_ground_plane(
            nets=nets,
            stack=stack,
            ground_net=args.ground_net,
            ground_layer=args.ground_plane_layer,
            margin_um=args.ground_plane_margin,
            ground_z_bottom=args.ground_plane_z_bottom,
            ground_thickness=args.ground_plane_thickness,
        )
        if added_explicit_ground:
            layer_info = stack.get_layer(args.ground_plane_layer)
            print(
                f"Added explicit ground plane net '{args.ground_net}' "
                f"(layer={args.ground_plane_layer}, z={layer_info.z_bottom:.3f} um, "
                f"t={layer_info.thickness:.3f} um, "
                f"margin={args.ground_plane_margin} um)"
            )
            if args.ground_plane_panel_size is not None:
                net_max_panel_size = {
                    args.ground_net: max(1e-6, args.ground_plane_panel_size)
                }
                print(
                    f"Ground plane fixed panel size: "
                    f"{net_max_panel_size[args.ground_net]} um"
                )
        else:
            print("Warning: explicit ground plane requested, but no valid geometry found")

    # Step 4: Mesh conductors into panels
    panels, net_indices = mesh_all_nets(
        nets,
        stack,
        max_panel_size=args.panel_size,
        min_panel_size=args.min_panel_size,
        adaptive_mesh=args.adaptive_mesh,
        proximity_distance=args.proximity_distance,
        proximity_factor=args.proximity_factor,
        remove_internal_faces=args.remove_internal_faces,
        net_max_panel_size=net_max_panel_size,
        edge_refine_factor=args.edge_refine_factor,
        edge_refine_fraction=args.edge_refine_fraction,
    )
    print(f"Generated {len(panels)} BEM panels (max_size={args.panel_size} um)")

    # Step 5: Solve BEM
    def solve_for_mode(mode: str):
        use_ground_plane = not args.no_ground_plane
        if added_explicit_ground and use_ground_plane:
            use_ground_plane = False
            print("Explicit ground plane active: disabling image-ground model")
        solver = BEMSolver(
            panels,
            net_indices,
            stack,
            use_ground_plane=use_ground_plane,
            near_field_factor=args.near_field_factor,
            near_field_samples=args.near_field_samples,
            uniform_epsilon_r=args.uniform_epsilon,
        )
        nets_data = nets if mode == "analytic" else None
        results_local = solver.extract_coupling_caps(
            ground_net=args.ground_net,
            nets_data=nets_data,
            min_cap_fF=args.min_cap,
            signal_scale=args.signal_scale,
            ground_scale=args.ground_scale,
        )
        return results_local

    if args.ground_model == "both":
        if args.output.lower().endswith(".csv"):
            base = args.output[:-4]
        else:
            base = args.output
        out_analytic = f"{base}_analytic.csv"
        out_matrix = f"{base}_matrix.csv"

        results_analytic = solve_for_mode("analytic")
        write_csv(results_analytic, out_analytic)
        print(f"Extracted {len(results_analytic)} coupling capacitances (analytic ground)")
        print(f"Results written to {out_analytic}")

        results_matrix = solve_for_mode("matrix")
        write_csv(results_matrix, out_matrix)
        print(f"Extracted {len(results_matrix)} coupling capacitances (matrix ground)")
        print(f"Results written to {out_matrix}")
    else:
        results = solve_for_mode(args.ground_model)
        print(f"Extracted {len(results)} coupling capacitances ({args.ground_model} ground)")
        # Step 6: Write output
        write_csv(results, args.output)
        print(f"Results written to {args.output}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
