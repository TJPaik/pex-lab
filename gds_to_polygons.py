"""
GDS to polygons.txt converter.

Reads a GDS file, extracts polygons and text labels, builds net connectivity
using Union-Find, and outputs polygons.txt format.

Usage:
    python gds_to_polygons.py input.gds -o output_polygons.txt
"""

import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    import gdstk
except ImportError:
    print("Error: gdstk is required. Install with: pip install gdstk")
    sys.exit(1)

import numpy as np


# sky130A GDS layer map: (layer_number, datatype) -> layer_name
# Based on sky130A PDK layer definitions
# sky130A GDS layer map from https://github.com/google/skywater-pdk
# Source: docs/rules/gds_layers.csv
SKY130A_LAYER_MAP = {
    # N-well
    (64, 20): "nwell",
    # Diffusion (active area) - type depends on well underneath
    (65, 20): "ndiff",   # diff:drawing
    # Tap (substrate contact diffusion)
    (65, 44): "nsubdiff",  # tap:drawing
    # Poly
    (66, 20): "poly",     # poly:drawing
    # Licon1 (contact to local interconnect)
    (66, 44): "polycont",  # licon1:drawing
    # Local interconnect
    (67, 20): "locali",   # li1:drawing
    # Mcon (li1 to met1 contact)
    (67, 44): "viali",    # mcon:drawing
    # Metal 1
    (68, 20): "metal1",   # met1:drawing
    (68, 44): "via1",     # via:drawing (met1 to met2)
    # Metal 2
    (69, 20): "metal2",   # met2:drawing
    (69, 44): "via2",     # via2:drawing (met2 to met3)
    # Metal 3
    (70, 20): "metal3",   # met3:drawing
    (70, 44): "via3",     # via3:drawing (met3 to met4)
    # Metal 4
    (71, 20): "metal4",   # met4:drawing
    (71, 44): "via4",     # via4:drawing (met4 to met5)
    # Metal 5
    (72, 20): "metal5",   # met5:drawing
}

# Label layers from gds_layers.csv (text type layers)
SKY130A_LABEL_LAYER_TO_CONDUCTOR = {
    (65, 6):  "ndiff",    # diff:label
    (66, 5):  "poly",     # poly:label
    (67, 5):  "locali",   # li1:label
    (68, 5):  "metal1",   # met1:label
    (69, 5):  "metal2",   # met2:label
    (70, 5):  "metal3",   # met3:label
    (71, 5):  "metal4",   # met4:label
    (72, 5):  "metal5",   # met5:label
}


@dataclass
class GDSRect:
    """Rectangle extracted from GDS."""
    x1: int
    y1: int
    x2: int
    y2: int
    layer: str
    gds_layer: int
    gds_datatype: int


class UnionFind:
    """Union-Find data structure for connectivity grouping."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def polygon_to_rect(vertices: np.ndarray, scale: float) -> Optional[Tuple[int, int, int, int]]:
    """Convert polygon vertices to axis-aligned rectangle coordinates.

    Args:
        vertices: Nx2 array of polygon vertices
        scale: multiply coordinates by this factor to get integer units

    Returns:
        (x1, y1, x2, y2) in integer units, or None if not a rectangle
    """
    if len(vertices) < 3:
        return None

    xs = vertices[:, 0]
    ys = vertices[:, 1]
    x1 = int(round(np.min(xs) * scale))
    y1 = int(round(np.min(ys) * scale))
    x2 = int(round(np.max(xs) * scale))
    y2 = int(round(np.max(ys) * scale))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def rects_overlap(r1: GDSRect, r2: GDSRect) -> bool:
    """Check if two rectangles overlap or touch (share boundary)."""
    return not (r1.x2 < r2.x1 or r2.x2 < r1.x1 or
                r1.y2 < r2.y1 or r2.y2 < r1.y1)


def point_in_rect(px: int, py: int, r: GDSRect) -> bool:
    """Check if point (px, py) is inside or on boundary of rect."""
    return r.x1 <= px <= r.x2 and r.y1 <= py <= r.y2


# Via-to-layer connectivity: which layers does each via connect?
# Note: In sky130A GDS, licon1 (66,44) is used for BOTH poly contacts and
# diffusion contacts. We map it to "polycont" in the layer map, but need to
# connect it to ndiff/pdiff/nsubdiff as well (since in the actual GDS, the
# same licon rectangle overlaps either poly or diffusion to form the contact).
VIA_CONNECTIVITY = {
    "ndiffc":       ("ndiff", "locali"),
    "pdiffc":       ("pdiff", "locali"),
    "nsubdiffcont": ("nsubdiff", "locali"),
    "polycont":     ("poly", "locali"),
    "viali":        ("locali", "metal1"),
    "via1":         ("metal1", "metal2"),
    "via2":         ("metal2", "metal3"),
    "via3":         ("metal3", "metal4"),
}

# Additional connectivity for licon1 (polycont in our mapping) to diffusion.
# licon1 overlapping diffusion acts as a diffusion contact.
LICON_DIFF_CONNECTIVITY = {
    "polycont": ["ndiff", "pdiff", "nsubdiff"],
}


class GDSToPolygons:
    """Convert GDS file to polygons.txt format."""

    def __init__(self, gds_path: str, layer_map: dict = None,
                 coord_scale: float = None):
        """
        Args:
            gds_path: Path to GDS file
            layer_map: GDS (layer, datatype) -> layer_name mapping
            coord_scale: Scale factor from GDS units to polygon integer units
                         (centnm). If None, auto-detect from GDS library unit.
        """
        self.lib = gdstk.read_gds(gds_path)
        self.layer_map = layer_map or SKY130A_LAYER_MAP
        if coord_scale is not None:
            self.coord_scale = coord_scale
        else:
            # Auto-detect: GDS unit -> centnm (0.01 um = 1e-8 m)
            # lib.unit is meters per GDS unit (e.g., 1e-6 for um, 1.0 for m)
            centnm_in_meters = 1e-8  # 0.01 um = 1e-8 m
            self.coord_scale = self.lib.unit / centnm_in_meters
            print(f"  GDS unit={self.lib.unit}, auto coord_scale={self.coord_scale}")
        self.top_cell = self.lib.top_level()[0]

    def extract_all(self) -> Tuple[List[GDSRect], Dict[str, List[Tuple[int, int, str]]]]:
        """Extract all polygons and labels from the GDS.

        Returns:
            rects: List of GDSRect
            labels: Dict of label_text -> [(x, y, conductor_layer), ...]
        """
        rects = []
        # Flatten hierarchy to resolve all cell references
        flat_cell = self.top_cell.copy(self.top_cell.name + "_flat")
        flat_cell.flatten()
        polygons = flat_cell.polygons

        for poly in polygons:
            key = (poly.layer, poly.datatype)
            layer_name = self.layer_map.get(key)
            if layer_name is None:
                continue

            coords = polygon_to_rect(poly.points, self.coord_scale)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            rects.append(GDSRect(x1, y1, x2, y2, layer_name,
                                 poly.layer, poly.datatype))

        # Extract labels (from flattened cell)
        labels: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        all_labels = flat_cell.labels

        for label in all_labels:
            lkey = (label.layer, label.texttype)
            conductor_layer = SKY130A_LABEL_LAYER_TO_CONDUCTOR.get(lkey)
            if conductor_layer is None:
                # Try to find the conductor layer from the main layer map
                conductor_layer = self.layer_map.get(lkey)
            if conductor_layer is None:
                continue

            lx = int(round(label.origin[0] * self.coord_scale))
            ly = int(round(label.origin[1] * self.coord_scale))
            labels[label.text].append((lx, ly, conductor_layer))

        return rects, labels

    def split_diff_at_poly(self, rects: List[GDSRect]) -> List[GDSRect]:
        """Split diffusion rects at poly gate crossings.

        In a MOSFET, poly crossing over diffusion creates a gate that
        electrically separates source and drain. This function splits
        the diffusion rect into separate source/drain regions.
        """
        DIFF_LAYERS = {"ndiff", "nsubdiff"}  # layers that can be split by poly
        poly_rects = [r for r in rects if r.layer == "poly"]
        if not poly_rects:
            return rects

        new_rects = []
        for r in rects:
            if r.layer not in DIFF_LAYERS:
                new_rects.append(r)
                continue

            # Find poly rects that cross this diffusion vertically
            # (poly runs in x-direction across diffusion)
            # A poly "crosses" if it overlaps the diff AND extends beyond
            # it in the y-direction (forming a gate channel)
            cut_xs = []  # x-positions where poly cuts the diff
            for pr in poly_rects:
                # Poly must overlap the diff region
                if pr.x2 <= r.x1 or pr.x1 >= r.x2:
                    continue
                if pr.y2 <= r.y1 or pr.y1 >= r.y2:
                    continue
                # Poly gate region within the diff
                gate_x1 = max(pr.x1, r.x1)
                gate_x2 = min(pr.x2, r.x2)
                if gate_x2 > gate_x1:
                    cut_xs.append((gate_x1, gate_x2))

            if not cut_xs:
                new_rects.append(r)
                continue

            # Sort cuts by x position
            cut_xs.sort()

            # Split the diff rect at each poly gate
            # Create sub-rects for the regions between gates
            segments = []
            prev_x = r.x1
            for gx1, gx2 in cut_xs:
                if gx1 > prev_x:
                    # Source/drain region before gate
                    segments.append((prev_x, r.y1, gx1, r.y2))
                # Gate region itself (skip - poly covers this)
                prev_x = gx2
            if prev_x < r.x2:
                # Source/drain region after last gate
                segments.append((prev_x, r.y1, r.x2, r.y2))

            if segments:
                for sx1, sy1, sx2, sy2 in segments:
                    if sx2 > sx1:
                        new_rects.append(GDSRect(sx1, sy1, sx2, sy2,
                                                 r.layer, r.gds_layer,
                                                 r.gds_datatype))
            else:
                # Entire diff is under gate - keep original
                new_rects.append(r)

        return new_rects

    def build_connectivity(self, rects: List[GDSRect],
                           labels: Dict[str, List[Tuple[int, int, str]]]
                           ) -> Dict[str, List[GDSRect]]:
        """Group rectangles into nets using Union-Find.

        Steps:
        1. Seed nets from text labels
        2. Merge overlapping/touching polygons on same layer
        3. Merge through via connectivity
        4. Propagate net names
        """
        n = len(rects)
        if n == 0:
            return {}

        uf = UnionFind(n)

        # Index rects by layer for efficient lookup
        layer_rects: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(rects):
            layer_rects[r.layer].append(i)

        # Step 1: Merge overlapping rects on same layer
        for layer_name, indices in layer_rects.items():
            # Sort by x1 for sweep
            indices_sorted = sorted(indices, key=lambda i: rects[i].x1)
            for a_pos in range(len(indices_sorted)):
                ia = indices_sorted[a_pos]
                ra = rects[ia]
                for b_pos in range(a_pos + 1, len(indices_sorted)):
                    ib = indices_sorted[b_pos]
                    rb = rects[ib]
                    if rb.x1 > ra.x2:
                        break  # no more overlaps possible
                    if rects_overlap(ra, rb):
                        uf.union(ia, ib)

        # Step 2: Via connectivity
        for via_layer, (lower, upper) in VIA_CONNECTIVITY.items():
            via_indices = layer_rects.get(via_layer, [])
            lower_indices = layer_rects.get(lower, [])
            upper_indices = layer_rects.get(upper, [])

            for vi in via_indices:
                vr = rects[vi]
                for li in lower_indices:
                    if rects_overlap(vr, rects[li]):
                        uf.union(vi, li)
                for ui in upper_indices:
                    if rects_overlap(vr, rects[ui]):
                        uf.union(vi, ui)

        # Step 2b: licon1 (polycont) also connects to diffusion layers
        # In sky130A GDS, licon1 (66,44) serves as both poly contact and
        # diffusion contact depending on what it overlaps.
        # A licon is a DIFF contact only if it does NOT overlap any poly.
        poly_indices = layer_rects.get("poly", [])
        for via_layer, diff_layers in LICON_DIFF_CONNECTIVITY.items():
            via_indices = layer_rects.get(via_layer, [])
            for vi in via_indices:
                vr = rects[vi]
                # Check if this licon overlaps any poly
                overlaps_poly = any(
                    rects_overlap(vr, rects[pi]) for pi in poly_indices
                )
                if overlaps_poly:
                    continue  # this is a poly contact, skip diff connection
                # This licon is a diff contact
                for diff_layer in diff_layers:
                    for di in layer_rects.get(diff_layer, []):
                        if rects_overlap(vr, rects[di]):
                            uf.union(vi, di)

        # Step 3: Seed net names from labels
        root_to_name: Dict[int, str] = {}
        for label_text, positions in labels.items():
            for lx, ly, conductor_layer in positions:
                for i in layer_rects.get(conductor_layer, []):
                    if point_in_rect(lx, ly, rects[i]):
                        root = uf.find(i)
                        root_to_name[root] = label_text
                        break

        # Step 4: Group rects by net
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)

        # Assign names
        net_rects: Dict[str, List[GDSRect]] = {}
        unnamed_count = 0
        for root, members in groups.items():
            name = root_to_name.get(root)
            if name is None:
                unnamed_count += 1
                name = f"net_{unnamed_count}"
            net_rects[name] = [rects[i] for i in members]

        return net_rects

    def run(self, output_path: str):
        """Full pipeline: extract, connect, write."""
        print(f"Reading GDS...")
        rects, labels = self.extract_all()
        print(f"  Found {len(rects)} polygons, {len(labels)} labels")

        # Split diffusion at poly gate crossings (MOSFET source/drain separation)
        rects = self.split_diff_at_poly(rects)
        print(f"  After diff splitting: {len(rects)} polygons")

        print(f"Building connectivity...")
        net_rects = self.build_connectivity(rects, labels)
        print(f"  Found {len(net_rects)} nets")

        print(f"Writing {output_path}...")
        self.write_polygons_txt(net_rects, output_path)
        print(f"  Done.")

    @staticmethod
    def write_polygons_txt(net_rects: Dict[str, List[GDSRect]],
                           output_path: str):
        """Write output in polygons.txt format."""
        with open(output_path, 'w') as f:
            for net_name in sorted(net_rects.keys()):
                rects = net_rects[net_name]
                f.write(f"Net: {net_name}\n")
                for r in rects:
                    f.write(f"  rect {r.x1} {r.y1} {r.x2} {r.y2} {r.layer}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GDS file to polygons.txt format"
    )
    parser.add_argument("gds_file", help="Input GDS file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output polygons.txt path")
    parser.add_argument("--scale", type=float, default=None,
                        help="Coordinate scale factor (default: auto-detect from GDS unit)")
    args = parser.parse_args()

    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")

    converter = GDSToPolygons(args.gds_file, coord_scale=args.scale)
    converter.run(args.output)


if __name__ == "__main__":
    main()
