"""
Process stack definition for parasitic capacitance extraction.

Defines the physical layer stack (conductor heights, thicknesses, dielectric
constants) and supports loading from JSON config files.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerInfo:
    """Physical properties of a single conducting layer."""
    name: str
    z_bottom: float      # bottom of conductor in micrometers
    thickness: float     # conductor thickness in micrometers
    is_via: bool = False

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.thickness


@dataclass
class DielectricLayer:
    """A dielectric region between conductors."""
    z_bottom: float
    z_top: float
    epsilon_r: float


@dataclass
class ProcessStack:
    """Complete process stack definition."""
    name: str
    units: str
    scale_to_um: float
    layers: Dict[str, LayerInfo]
    dielectrics: List[DielectricLayer]
    substrate_epsilon_r: float = 11.7

    @classmethod
    def from_json(cls, path: str) -> "ProcessStack":
        with open(path) as f:
            data = json.load(f)
        layers = {}
        for lname, props in data["layers"].items():
            layers[lname] = LayerInfo(
                name=props["name"],
                z_bottom=props["z_bottom"],
                thickness=props["thickness"],
                is_via=props.get("is_via", False),
            )
        dielectrics = [
            DielectricLayer(
                z_bottom=d["z_bottom"],
                z_top=d["z_top"],
                epsilon_r=d["epsilon_r"],
            )
            for d in data["dielectrics"]
        ]
        return cls(
            name=data["name"],
            units=data.get("units", "centnm"),
            scale_to_um=data.get("scale_to_um", 0.01),
            layers=layers,
            dielectrics=dielectrics,
            substrate_epsilon_r=data.get("substrate_epsilon_r", 11.7),
        )

    def get_effective_epsilon(self, z: float) -> float:
        """Return dielectric constant at height z."""
        for d in self.dielectrics:
            if d.z_bottom <= z < d.z_top:
                return d.epsilon_r
        return 1.0  # air above top dielectric

    def get_layer(self, name: str) -> Optional[LayerInfo]:
        return self.layers.get(name)


def default_sky130a_stack() -> ProcessStack:
    """Return default sky130A process stack."""
    import os
    generated_json = os.path.join(
        os.path.dirname(__file__), "sky130a_stack_from_pdk.json")
    if os.path.exists(generated_json):
        return ProcessStack.from_json(generated_json)

    default_json = os.path.join(os.path.dirname(__file__), "sky130a_stack.json")
    if os.path.exists(default_json):
        return ProcessStack.from_json(default_json)

    # Fallback hardcoded values from google/skywater-pdk metal_stack.ps
    layers = {
        "nwell":        LayerInfo("nwell",        0.0,    0.0,    False),
        "ndiff":        LayerInfo("ndiff",        0.0,    0.12,   False),
        "pdiff":        LayerInfo("pdiff",        0.0,    0.12,   False),
        "ndiffc":       LayerInfo("ndiffc",       0.0,    0.9361, True),
        "pdiffc":       LayerInfo("pdiffc",       0.0,    0.9361, True),
        "nsubdiff":     LayerInfo("nsubdiff",     0.0,    0.12,   False),
        "nsubdiffcont": LayerInfo("nsubdiffcont", 0.0,    0.9361, True),
        "poly":         LayerInfo("poly",         0.3262, 0.18,   False),
        "nmos":         LayerInfo("nmos",         0.0,    0.0,    False),
        "pmos":         LayerInfo("pmos",         0.0,    0.0,    False),
        "polycont":     LayerInfo("polycont",     0.5062, 0.4299, True),
        "locali":       LayerInfo("locali",       0.9361, 0.10,   False),
        "viali":        LayerInfo("viali",        1.0361, 0.3400, True),
        "metal1":       LayerInfo("metal1",       1.3761, 0.36,   False),
        "via1":         LayerInfo("via1",         1.7361, 0.2700, True),
        "metal2":       LayerInfo("metal2",       2.0061, 0.36,   False),
        "via2":         LayerInfo("via2",         2.3661, 0.4200, True),
        "metal3":       LayerInfo("metal3",       2.7861, 0.845,  False),
        "via3":         LayerInfo("via3",         3.6311, 0.3900, True),
        "metal4":       LayerInfo("metal4",       4.0211, 0.845,  False),
    }
    dielectrics = [
        DielectricLayer(-1.0,    0.0,    11.7),  # Si substrate
        DielectricLayer(0.0,     0.3262, 3.9),   # FOX + PSG
        DielectricLayer(0.3262,  0.9361, 7.3),   # LINT
        DielectricLayer(0.9361,  1.3761, 4.05),  # NILD2
        DielectricLayer(1.3761,  2.0061, 4.5),   # NILD3
        DielectricLayer(2.0061,  2.7861, 4.2),   # NILD4
        DielectricLayer(2.7861,  4.0211, 4.1),   # NILD5
        DielectricLayer(4.0211,  5.3711, 4.0),   # NILD6
        DielectricLayer(5.3711,  15.0,   3.9),   # TOPOX + air
    ]
    return ProcessStack(
        name="sky130A",
        units="centnm",
        scale_to_um=0.01,
        layers=layers,
        dielectrics=dielectrics,
        substrate_epsilon_r=11.7,
    )
