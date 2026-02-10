"""
BEM (Boundary Element Method) solver for parasitic capacitance extraction.

Uses constant-charge panels with free-space Green's function, effective
average dielectric, and method of images for substrate ground plane.

Ground (substrate) capacitance is computed analytically using parallel-plate
+ fringe capacitance model for better accuracy.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from typing import List, Dict, Tuple

from mesh import Panel
from process_stack import ProcessStack

# Physical constants
EPSILON_0 = 8.854187817e-12  # F/m
UM_TO_M = 1e-6
FF_FACTOR = 1e15  # F -> fF


class BEMSolver:
    """BEM capacitance solver."""

    def __init__(self, panels: List[Panel], net_indices: Dict[str, int],
                 stack: ProcessStack, use_ground_plane: bool = True,
                 near_field_factor: float = 0.0,
                 near_field_samples: int = 3,
                 uniform_epsilon_r: float = None):
        self.panels = panels
        self.net_indices = net_indices
        self.num_nets = len(net_indices)
        self.N = len(panels)
        self.stack = stack
        self.use_ground_plane = use_ground_plane
        self.near_field_factor = max(0.0, near_field_factor)
        self.near_field_samples = max(1, int(near_field_samples))
        self.uniform_epsilon_r = uniform_epsilon_r

        # Precompute arrays for vectorized operations
        self.centers = np.array([[p.cx, p.cy, p.cz] for p in panels])
        self.areas = np.array([p.area for p in panels])
        self.net_ids = np.array([p.net_idx for p in panels])

        # Precompute effective epsilon at each panel height
        if uniform_epsilon_r is not None:
            eps_u = max(1e-9, float(uniform_epsilon_r))
            self.eps_at_panel = np.full(len(panels), eps_u, dtype=float)
        else:
            self.eps_at_panel = np.array([
                stack.get_effective_epsilon(p.cz) for p in panels
            ])

    def _panel_sample_points(self, panel: Panel) -> np.ndarray:
        """Sample source panel with uniform subcells for near-field integration."""
        n = self.near_field_samples
        if n <= 1 or panel.du <= 0 or panel.dv <= 0:
            return np.array([[panel.cx, panel.cy, panel.cz]], dtype=float)
        u_off = ((np.arange(n) + 0.5) / n - 0.5) * panel.du
        v_off = ((np.arange(n) + 0.5) / n - 0.5) * panel.dv
        uu, vv = np.meshgrid(u_off, v_off, indexing="ij")
        pts = np.tile(
            np.array([[panel.cx, panel.cy, panel.cz]], dtype=float),
            (n * n, 1),
        )
        pts[:, panel.u_axis] += uu.ravel()
        pts[:, panel.v_axis] += vv.ravel()
        return pts

    def _self_term_rect(self, du_um: float, dv_um: float, eps_r: float) -> float:
        """Diagonal self-term for a uniformly charged rectangular panel."""
        du_m = max(du_um * UM_TO_M, 1e-15)
        dv_m = max(dv_um * UM_TO_M, 1e-15)
        a = 0.5 * du_m
        b = 0.5 * dv_m
        r = np.sqrt(a * a + b * b)
        # I = ∬_panel 1/r dA for center-observation rectangle.
        i_rect = 4.0 * (
            a * np.log((b + r) / a) + b * np.log((a + r) / b)
        )
        return i_rect / (4.0 * np.pi * EPSILON_0 * eps_r)

    def _apply_near_field_correction(self, P: np.ndarray, dist: np.ndarray,
                                     eps_avg: np.ndarray, areas_m2: np.ndarray):
        """Recompute close interactions with source-panel quadrature."""
        if self.near_field_factor <= 0.0 or self.near_field_samples <= 1:
            return
        total_pairs = 0
        centers_m = self.centers * UM_TO_M
        char_len = np.sqrt(self.areas)

        for j, panel_j in enumerate(self.panels):
            threshold_vec = self.near_field_factor * np.maximum(
                char_len, max(char_len[j], 1e-6)
            )
            near_i = np.where((dist[:, j] > 0.0) & (dist[:, j] < threshold_vec))[0]
            if near_i.size == 0:
                continue

            src_pts = self._panel_sample_points(panel_j) * UM_TO_M
            sub_area = areas_m2[j] / src_pts.shape[0]
            eps_col = eps_avg[near_i, j][:, None]
            fld_pts = centers_m[near_i]

            d = np.linalg.norm(fld_pts[:, None, :] - src_pts[None, :, :], axis=2)
            d = np.maximum(d, 1e-15)
            val = np.sum(sub_area / (4.0 * np.pi * EPSILON_0 * eps_col * d), axis=1)

            if self.use_ground_plane:
                src_img = src_pts.copy()
                src_img[:, 2] *= -1.0
                d_img = np.linalg.norm(
                    fld_pts[:, None, :] - src_img[None, :, :], axis=2
                )
                d_img = np.maximum(d_img, 1e-15)
                val -= np.sum(
                    sub_area / (4.0 * np.pi * EPSILON_0 * eps_col * d_img), axis=1
                )

            P[near_i, j] = val
            total_pairs += int(near_i.size)

        print(
            f"  Near-field correction: {total_pairs} interactions "
            f"(samples={self.near_field_samples}x{self.near_field_samples})"
        )

    def build_coefficient_matrix(self) -> np.ndarray:
        """Build the N x N potential coefficient matrix P.

        P[i,j] = potential at panel i center due to unit charge density
                  on panel j (times area_j).
        """
        print(f"  Building {self.N}x{self.N} coefficient matrix...")
        N = self.N

        cx = self.centers[:, 0]
        cy = self.centers[:, 1]
        cz = self.centers[:, 2]

        # Pairwise distances
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dz = cz[:, None] - cz[None, :]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Average epsilon between source and field panels
        eps_avg = 0.5 * (self.eps_at_panel[:, None] + self.eps_at_panel[None, :])

        # Convert to SI units
        dist_m = dist * UM_TO_M
        areas_m2 = self.areas * UM_TO_M**2

        # Off-diagonal: P[i,j] = area_j / (4*pi*eps0*eps_avg*dist)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = areas_m2[None, :] / (4.0 * np.pi * EPSILON_0 * eps_avg * dist_m)

        # Ground plane image: use full image (gamma=1) for conductor shielding
        if self.use_ground_plane:
            dz_img = cz[:, None] + cz[None, :]
            dist_img = np.sqrt(dx**2 + dy**2 + dz_img**2)
            dist_img_m = dist_img * UM_TO_M

            with np.errstate(divide='ignore', invalid='ignore'):
                P_img = areas_m2[None, :] / (
                    4.0 * np.pi * EPSILON_0 * eps_avg * dist_img_m
                )
            P_img = np.nan_to_num(P_img, nan=0.0, posinf=0.0, neginf=0.0)
            P -= P_img

        # Self-terms (diagonal): analytical rectangle self-term.
        for i in range(N):
            eps_r = self.eps_at_panel[i]
            p_i = self.panels[i]
            du = p_i.du if p_i.du > 0 else np.sqrt(max(self.areas[i], 1e-18))
            dv = p_i.dv if p_i.dv > 0 else np.sqrt(max(self.areas[i], 1e-18))
            P[i, i] = self._self_term_rect(du, dv, eps_r)
            if self.use_ground_plane:
                z_m = self.panels[i].cz * UM_TO_M
                dist_self_img = 2.0 * z_m
                if dist_self_img > 0:
                    P[i, i] -= areas_m2[i] / (
                        4.0 * np.pi * EPSILON_0 * eps_r * dist_self_img
                    )

        self._apply_near_field_correction(P, dist, eps_avg, areas_m2)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        return P

    def solve_capacitance_matrix(self) -> np.ndarray:
        """Solve for the net-level capacitance matrix."""
        print(f"  Solving BEM: {self.N} panels, {self.num_nets} nets...")
        if self.N == 0 or self.num_nets == 0:
            return np.zeros((max(self.num_nets, 1), max(self.num_nets, 1)))

        P = self.build_coefficient_matrix()

        print("  LU factorization...")
        lu, piv = lu_factor(P)

        areas_m2 = self.areas * UM_TO_M**2
        cap_matrix = np.zeros((self.num_nets, self.num_nets))

        for k in range(self.num_nets):
            V = np.zeros(self.N)
            V[self.net_ids == k] = 1.0
            sigma = lu_solve((lu, piv), V)
            for m in range(self.num_nets):
                mask_m = (self.net_ids == m)
                Q_m = np.sum(sigma[mask_m] * areas_m2[mask_m])
                cap_matrix[m, k] = Q_m

        return cap_matrix

    def _multilayer_cap_per_area(self, z_conductor: float) -> float:
        """Compute parallel-plate cap per unit area from conductor to substrate.

        Uses series capacitor model for multi-layer dielectric:
        1/C_total = sum(d_i / (eps_0 * K_i)) for each dielectric slab.

        Args:
            z_conductor: bottom of conductor in um

        Returns:
            Capacitance per unit area in F/m^2
        """
        if z_conductor <= 0:
            return 0.0

        inv_c = 0.0
        for d in self.stack.dielectrics:
            if d.z_top <= 0 or d.z_bottom >= z_conductor:
                continue
            z_lo = max(d.z_bottom, 0.0)
            z_hi = min(d.z_top, z_conductor)
            thickness_m = (z_hi - z_lo) * UM_TO_M
            if thickness_m > 0:
                inv_c += thickness_m / (EPSILON_0 * d.epsilon_r)

        if inv_c <= 0:
            return 0.0
        return 1.0 / inv_c

    def compute_ground_caps(self, nets_data) -> Dict[str, float]:
        """Compute capacitance to ground (substrate) analytically.

        Uses multi-layer parallel-plate model (series capacitor through
        dielectric stack), fringe capacitance, junction capacitance for
        diffusion, and nwell depletion capacitance.

        Args:
            nets_data: Dict[str, NetGeometry] from polygon_parser

        Returns:
            Dict mapping net_name -> ground_cap_fF
        """
        from mesh import SKIP_LAYERS

        # Layers at substrate level: junction capacitance
        SUBSTRATE_LAYERS = {"ndiff", "pdiff", "nsubdiff"}

        ground_caps = {}
        scale = self.stack.scale_to_um

        for net_name, net_geom in nets_data.items():
            total_cap_f = 0.0
            for rect in net_geom.rects:
                w = (rect.x2 - rect.x1) * scale
                h = (rect.y2 - rect.y1) * scale
                area_um2 = w * h
                perimeter_um = 2.0 * (w + h)

                # Nwell depletion cap to p-substrate
                # sky130A: C_nwell ≈ 0.038 fF/um² (reverse-biased junction)
                if rect.layer == "nwell":
                    c_nwell_fF = 0.038 * area_um2
                    total_cap_f += c_nwell_fF * 1e-15
                    continue

                if rect.layer in SKIP_LAYERS:
                    continue

                layer = self.stack.get_layer(rect.layer)
                if layer is None or layer.thickness <= 0:
                    continue
                if layer.is_via:
                    continue

                if rect.layer in SUBSTRATE_LAYERS:
                    # sky130A junction cap: Cj0 ~0.14-0.17 fF/um² + Cjsw ~0.03 fF/um
                    c_junc_fF = 0.15 * area_um2 + 0.03 * perimeter_um
                    total_cap_f += c_junc_fF * 1e-15
                    continue

                # Parallel plate cap: bottom surface to substrate
                cap_per_area = self._multilayer_cap_per_area(layer.z_bottom)
                area_m2 = area_um2 * UM_TO_M**2
                c_pp = cap_per_area * area_m2

                # Fringe capacitance (downward from edges to substrate)
                z_bot = layer.z_bottom
                t = layer.thickness
                if t > 0 and z_bot > 0:
                    eps_eff = self.stack.get_effective_epsilon(z_bot)
                    perim_m = perimeter_um * UM_TO_M
                    # Fringe per unit length: C/L = (eps_0 * eps_r / pi) * (t/h) * ln(1 + 2h/t)
                    fringe_factor = (t / z_bot) * np.log(1.0 + 2.0 * z_bot / t) / np.pi
                    c_fringe = EPSILON_0 * eps_eff * fringe_factor * perim_m
                    total_cap_f += c_fringe

                total_cap_f += c_pp

            ground_caps[net_name] = total_cap_f * FF_FACTOR

        return ground_caps

    def extract_coupling_caps(self, ground_net: str = "GND",
                              nets_data=None,
                              min_cap_fF: float = 1e-6,
                              signal_scale: float = 1.0,
                              ground_scale: float = 1.0
                              ) -> List[Tuple[str, str, float]]:
        """Extract coupling capacitances between all net pairs.

        Args:
            ground_net: Name for substrate ground net
            nets_data: Dict[str, NetGeometry] for analytical ground cap calc.
                      If None, ground cap is derived from BEM matrix diagonal.
            signal_scale: Multiplicative scaling applied to net-to-net couplings.
            ground_scale: Multiplicative scaling applied only to ground caps.

        Returns list of (net1, net2, cap_fF) tuples.
        """
        cap_matrix = self.solve_capacitance_matrix()

        cap_fF = cap_matrix * FF_FACTOR
        cap_fF = 0.5 * (cap_fF + cap_fF.T)

        idx_to_name = {v: k for k, v in self.net_indices.items()}
        results = []
        explicit_ground_present = (ground_net in self.net_indices)

        # Net-to-net coupling caps
        for i in range(self.num_nets):
            for j in range(i + 1, self.num_nets):
                coupling = -cap_fF[i, j] * signal_scale
                if coupling > min_cap_fF:
                    name_i = idx_to_name[i]
                    name_j = idx_to_name[j]
                    results.append((name_i, name_j, round(coupling, 5)))

        # Ground capacitance: analytical model
        if (not explicit_ground_present) and (nets_data is not None):
            ground_caps = self.compute_ground_caps(nets_data)
            for net_name, g_cap in ground_caps.items():
                if net_name not in self.net_indices:
                    continue
                g_cap *= ground_scale
                if g_cap > min_cap_fF:
                    results.append((net_name, ground_net, round(g_cap, 5)))
        elif not explicit_ground_present:
            # Matrix-based ground from Maxwell row sum, valid for both
            # with/without explicit image-ground modeling.
            for i in range(self.num_nets):
                c_gnd = cap_fF[i, i] + sum(
                    cap_fF[i, j] for j in range(self.num_nets) if j != i
                )
                # Row sum should be non-negative; small negative values can
                # appear from discretization/numerical error.
                if c_gnd < 0:
                    c_gnd = abs(c_gnd)
                c_gnd *= ground_scale
                if c_gnd > min_cap_fF:
                    name_i = idx_to_name[i]
                    results.append((name_i, ground_net, round(c_gnd, 5)))

        return results
