"""
Surface meshing of 3D conductors for BEM capacitance extraction.

Each conducting rectangle is extruded to a 3D box using the process stack,
then the exposed surfaces are discretized into panels.
"""

import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict, Optional

from polygon_parser import Rect, NetGeometry
from process_stack import ProcessStack, LayerInfo


@dataclass
class Panel:
    """A single surface panel for BEM."""
    cx: float   # center x (um)
    cy: float   # center y (um)
    cz: float   # center z (um)
    nx: float   # normal x
    ny: float   # normal y
    nz: float   # normal z
    area: float  # panel area (um^2)
    net_idx: int
    # Local in-panel axes for near-field integration.
    u_axis: int = 0  # 0:x, 1:y, 2:z
    v_axis: int = 1  # 0:x, 1:y, 2:z
    du: float = 0.0  # panel length along u axis (um)
    dv: float = 0.0  # panel length along v axis (um)


# Layers to skip (not physical conductors or zero thickness)
SKIP_LAYERS = {"nwell", "nmos", "pmos"}


def _axis_segments(start: float, end: float, max_size: float,
                   edge_refine_factor: float = 1.0,
                   edge_refine_fraction: float = 0.0) -> List[Tuple[float, float, float, float]]:
    """Split one axis into segments, optionally refining near both edges."""
    span = end - start
    if span <= 0:
        return []

    max_size = max(max_size, 1e-9)
    edge_refine_factor = max(min(edge_refine_factor, 1.0), 1e-3)
    edge_refine_fraction = max(min(edge_refine_fraction, 0.49), 0.0)

    use_edge_refine = (
        edge_refine_factor < 1.0 and edge_refine_fraction > 0.0 and span > max_size
    )

    if not use_edge_refine:
        n = max(1, int(np.ceil(span / max_size)))
        d = span / n
        segs = []
        x = start
        for i in range(n):
            x_next = end if i == n - 1 else x + d
            segs.append((x, x_next, 0.5 * (x + x_next), x_next - x))
            x = x_next
        return segs

    edge_span = span * edge_refine_fraction
    center_span = max(0.0, span - 2.0 * edge_span)
    edge_size = max_size * edge_refine_factor

    n_edge = max(1, int(np.ceil(edge_span / edge_size))) if edge_span > 0 else 0
    n_center = max(1, int(np.ceil(center_span / max_size))) if center_span > 0 else 0

    seg_lengths: List[float] = []
    if n_edge > 0:
        seg_lengths.extend([edge_span / n_edge] * n_edge)
    if n_center > 0:
        seg_lengths.extend([center_span / n_center] * n_center)
    if n_edge > 0:
        seg_lengths.extend([edge_span / n_edge] * n_edge)

    if not seg_lengths:
        return []

    segs = []
    x = start
    for i, d in enumerate(seg_lengths):
        x_next = end if i == len(seg_lengths) - 1 else x + d
        segs.append((x, x_next, 0.5 * (x + x_next), x_next - x))
        x = x_next
    return segs


def subdivide_horizontal(x1: float, y1: float, x2: float, y2: float,
                         z: float, nz_sign: float, net_idx: int,
                         max_size: float,
                         edge_refine_factor: float = 1.0,
                         edge_refine_fraction: float = 0.0) -> List[Panel]:
    """Subdivide a horizontal surface into panels."""
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return []
    x_segments = _axis_segments(
        x1, x2, max_size, edge_refine_factor=edge_refine_factor,
        edge_refine_fraction=edge_refine_fraction
    )
    y_segments = _axis_segments(
        y1, y2, max_size, edge_refine_factor=edge_refine_factor,
        edge_refine_fraction=edge_refine_fraction
    )

    panels = []
    for _, _, cx, dx in x_segments:
        for _, _, cy, dy in y_segments:
            area = dx * dy
            panels.append(
                Panel(
                    cx, cy, z, 0.0, 0.0, nz_sign, area, net_idx,
                    0, 1, dx, dy
                )
            )
    return panels


def subdivide_vertical(u1: float, u2: float, z1: float, z2: float,
                       fixed: float, orient: str, net_idx: int,
                       max_size: float,
                       edge_refine_factor: float = 1.0,
                       edge_refine_fraction: float = 0.0) -> List[Panel]:
    """Subdivide a vertical (sidewall) surface into panels.

    u1..u2: span along the wall
    z1..z2: vertical span
    fixed: the fixed coordinate perpendicular to the wall
    orient: 'x+', 'x-', 'y+', 'y-'
    """
    span = u2 - u1
    height = z2 - z1
    if span <= 0 or height <= 0:
        return []

    u_segments = _axis_segments(
        u1, u2, max_size, edge_refine_factor=edge_refine_factor,
        edge_refine_fraction=edge_refine_fraction
    )
    z_segments = _axis_segments(
        z1, z2, max_size, edge_refine_factor=edge_refine_factor,
        edge_refine_fraction=edge_refine_fraction
    )

    normal_map = {
        'x+': (1.0, 0.0, 0.0), 'x-': (-1.0, 0.0, 0.0),
        'y+': (0.0, 1.0, 0.0), 'y-': (0.0, -1.0, 0.0),
    }
    nvec = normal_map[orient]

    panels = []
    for _, _, cu, du in u_segments:
        for _, _, cz, dz in z_segments:
            area = du * dz
            if orient.startswith('x'):
                panels.append(
                    Panel(
                        fixed, cu, cz, *nvec, area, net_idx,
                        1, 2, du, dz
                    )
                )
            else:
                panels.append(
                    Panel(
                        cu, fixed, cz, *nvec, area, net_idx,
                        0, 2, du, dz
                    )
                )
    return panels


def _panel_key(corners: List[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], ...]:
    """Canonical key for a quad face, order-independent."""
    return tuple(sorted(
        (round(c[0], 4), round(c[1], 4), round(c[2], 4)) for c in corners
    ))


def _rect_face_keys(rect: Rect, layer: LayerInfo, scale: float) -> Dict[str, Tuple]:
    """Return canonical keys for each geometric face of one extruded rect."""
    x1 = rect.x1 * scale
    y1 = rect.y1 * scale
    x2 = rect.x2 * scale
    y2 = rect.y2 * scale
    z_bot = layer.z_bottom
    z_top = layer.z_top

    w = x2 - x1
    h = y2 - y1
    t = z_top - z_bot
    if w <= 0 or h <= 0:
        return {}

    keys: Dict[str, Tuple] = {}
    if not layer.is_via:
        keys["top"] = _panel_key([
            (x1, y1, z_top), (x2, y1, z_top), (x2, y2, z_top), (x1, y2, z_top)
        ])
        keys["bot"] = _panel_key([
            (x1, y1, z_bot), (x2, y1, z_bot), (x2, y2, z_bot), (x1, y2, z_bot)
        ])

    if t > 0:
        keys["front"] = _panel_key([
            (x1, y2, z_bot), (x2, y2, z_bot), (x2, y2, z_top), (x1, y2, z_top)
        ])
        keys["back"] = _panel_key([
            (x1, y1, z_bot), (x2, y1, z_bot), (x2, y1, z_top), (x1, y1, z_top)
        ])
        keys["right"] = _panel_key([
            (x2, y1, z_bot), (x2, y2, z_bot), (x2, y2, z_top), (x2, y1, z_top)
        ])
        keys["left"] = _panel_key([
            (x1, y1, z_bot), (x1, y2, z_bot), (x1, y2, z_top), (x1, y1, z_top)
        ])
    return keys


def mesh_rect_3d(rect: Rect, layer: LayerInfo, net_idx: int,
                 scale: float, max_size: float,
                 external_face_keys: Optional[set] = None,
                 edge_refine_factor: float = 1.0,
                 edge_refine_fraction: float = 0.0) -> List[Panel]:
    """Create surface panels for a single 3D extruded rectangle."""
    x1 = rect.x1 * scale
    y1 = rect.y1 * scale
    x2 = rect.x2 * scale
    y2 = rect.y2 * scale
    z_bot = layer.z_bottom
    z_top = layer.z_top

    panels = []
    face_keys = _rect_face_keys(rect, layer, scale)

    # Horizontal surfaces (top and bottom) - skip for via layers
    if not layer.is_via and (
        external_face_keys is None or face_keys.get("top") in external_face_keys
    ):
        panels.extend(
            subdivide_horizontal(
                x1, y1, x2, y2, z_top, 1.0, net_idx, max_size,
                edge_refine_factor=edge_refine_factor,
                edge_refine_fraction=edge_refine_fraction,
            )
        )
    if not layer.is_via and (
        external_face_keys is None or face_keys.get("bot") in external_face_keys
    ):
        panels.extend(
            subdivide_horizontal(
                x1, y1, x2, y2, z_bot, -1.0, net_idx, max_size,
                edge_refine_factor=edge_refine_factor,
                edge_refine_fraction=edge_refine_fraction,
            )
        )

    # Four sidewalls
    # y = y2 wall (front, normal +y)
    if external_face_keys is None or face_keys.get("front") in external_face_keys:
        panels.extend(
            subdivide_vertical(
                x1, x2, z_bot, z_top, y2, 'y+', net_idx, max_size,
                edge_refine_factor=edge_refine_factor,
                edge_refine_fraction=edge_refine_fraction,
            )
        )
    # y = y1 wall (back, normal -y)
    if external_face_keys is None or face_keys.get("back") in external_face_keys:
        panels.extend(
            subdivide_vertical(
                x1, x2, z_bot, z_top, y1, 'y-', net_idx, max_size,
                edge_refine_factor=edge_refine_factor,
                edge_refine_fraction=edge_refine_fraction,
            )
        )
    # x = x2 wall (right, normal +x)
    if external_face_keys is None or face_keys.get("right") in external_face_keys:
        panels.extend(
            subdivide_vertical(
                y1, y2, z_bot, z_top, x2, 'x+', net_idx, max_size,
                edge_refine_factor=edge_refine_factor,
                edge_refine_fraction=edge_refine_fraction,
            )
        )
    # x = x1 wall (left, normal -x)
    if external_face_keys is None or face_keys.get("left") in external_face_keys:
        panels.extend(
            subdivide_vertical(
                y1, y2, z_bot, z_top, x1, 'x-', net_idx, max_size,
                edge_refine_factor=edge_refine_factor,
                edge_refine_fraction=edge_refine_fraction,
            )
        )

    return panels


def _box_distance(a: Tuple[float, float, float, float, float, float],
                  b: Tuple[float, float, float, float, float, float]) -> float:
    """Distance between two axis-aligned 3D boxes."""
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    dz = max(az1 - bz2, bz1 - az2, 0.0)
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _compute_adaptive_rect_panel_sizes(
    nets: Dict[str, NetGeometry],
    stack: ProcessStack,
    max_panel_size: float,
    min_panel_size: float,
    proximity_distance: float,
    proximity_factor: float,
) -> Dict[str, Dict[int, float]]:
    """Compute rect-level local panel sizes based on inter-net proximity."""
    entries = []
    scale = stack.scale_to_um
    for net_name, net in nets.items():
        for ridx, rect in enumerate(net.rects):
            if rect.layer in SKIP_LAYERS:
                continue
            layer = stack.get_layer(rect.layer)
            if layer is None or layer.thickness <= 0:
                continue
            x1 = rect.x1 * scale
            y1 = rect.y1 * scale
            x2 = rect.x2 * scale
            y2 = rect.y2 * scale
            z1 = layer.z_bottom
            z2 = layer.z_top
            if x2 <= x1 or y2 <= y1 or z2 <= z1:
                continue
            entries.append((net_name, ridx, (x1, y1, z1, x2, y2, z2)))

    panel_sizes: Dict[str, Dict[int, float]] = {}
    if not entries:
        return panel_sizes

    prox_dist = max(1e-6, proximity_distance)
    prox_factor = min(max(proximity_factor, 0.05), 1.0)

    for net_name, ridx, box in entries:
        min_dist = prox_dist
        for other_net, _, other_box in entries:
            if other_net == net_name:
                continue
            d = _box_distance(box, other_box)
            if d < min_dist:
                min_dist = d
                if min_dist <= 0.0:
                    break
        ratio = min(min_dist / prox_dist, 1.0)
        local_size = max_panel_size * (prox_factor + (1.0 - prox_factor) * ratio)
        local_size = max(min_panel_size, min(max_panel_size, local_size))
        panel_sizes.setdefault(net_name, {})[ridx] = local_size

    return panel_sizes


def mesh_net(net: NetGeometry, net_idx: int, stack: ProcessStack,
             max_panel_size: float = 1.0,
             rect_panel_sizes: Optional[Dict[int, float]] = None,
             remove_internal_faces: bool = False,
             edge_refine_factor: float = 1.0,
             edge_refine_fraction: float = 0.0) -> List[Panel]:
    """Mesh all rectangles of a net into surface panels."""
    panels = []
    external_face_keys: Optional[set] = None
    if remove_internal_faces:
        counts: Counter = Counter()
        for rect in net.rects:
            if rect.layer in SKIP_LAYERS:
                continue
            layer = stack.get_layer(rect.layer)
            if layer is None or layer.thickness <= 0:
                continue
            keys = _rect_face_keys(rect, layer, stack.scale_to_um)
            counts.update(keys.values())
        external_face_keys = {k for k, c in counts.items() if c == 1}

    for ridx, rect in enumerate(net.rects):
        if rect.layer in SKIP_LAYERS:
            continue
        layer_info = stack.get_layer(rect.layer)
        if layer_info is None or layer_info.thickness <= 0:
            continue
        local_panel_size = max_panel_size
        if rect_panel_sizes is not None:
            local_panel_size = rect_panel_sizes.get(ridx, max_panel_size)
        panels.extend(
            mesh_rect_3d(rect, layer_info, net_idx, stack.scale_to_um,
                         local_panel_size, external_face_keys,
                         edge_refine_factor=edge_refine_factor,
                         edge_refine_fraction=edge_refine_fraction)
        )
    return panels


def mesh_all_nets(nets: Dict[str, NetGeometry], stack: ProcessStack,
                  max_panel_size: float = 1.0,
                  min_panel_size: float = 0.2,
                  adaptive_mesh: bool = False,
                  proximity_distance: float = 2.0,
                  proximity_factor: float = 0.6,
                  remove_internal_faces: bool = False,
                  net_max_panel_size: Optional[Dict[str, float]] = None,
                  edge_refine_factor: float = 1.0,
                  edge_refine_fraction: float = 0.0) -> Tuple[List[Panel], Dict[str, int]]:
    """Mesh all nets.

    Returns:
        panels: List of all panels across all nets
        net_indices: Dict mapping net_name -> net_idx
    """
    net_names = sorted(nets.keys())
    net_indices: Dict[str, int] = {}
    all_panels: List[Panel] = []

    rect_panel_sizes_by_net: Dict[str, Dict[int, float]] = {}
    if adaptive_mesh:
        rect_panel_sizes_by_net = _compute_adaptive_rect_panel_sizes(
            nets,
            stack,
            max_panel_size=max_panel_size,
            min_panel_size=min_panel_size,
            proximity_distance=proximity_distance,
            proximity_factor=proximity_factor,
        )

    for name in net_names:
        idx = len(net_indices)
        net_panel_size = max_panel_size
        if net_max_panel_size is not None and name in net_max_panel_size:
            net_panel_size = max(1e-6, float(net_max_panel_size[name]))
        rect_sizes = rect_panel_sizes_by_net.get(name)
        # Keep special nets (e.g. explicit ground plane) at fixed panel size.
        if net_max_panel_size is not None and name in net_max_panel_size:
            rect_sizes = None
        panels = mesh_net(
            nets[name],
            idx,
            stack,
            net_panel_size,
            rect_panel_sizes=rect_sizes,
            remove_internal_faces=remove_internal_faces,
            edge_refine_factor=edge_refine_factor,
            edge_refine_fraction=edge_refine_fraction,
        )
        if not panels:
            continue
        net_indices[name] = idx
        all_panels.extend(panels)

    return all_panels, net_indices
