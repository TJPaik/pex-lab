"""
Parse polygons.txt format into structured data for BEM processing.

Format:
    Net: <net_name>
      rect x1 y1 x2 y2 <layer_name>
      rect x1 y1 x2 y2 <layer_name>
    Net: <net_name>
      ...
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Rect:
    """A single rectangle with integer coordinates and layer name."""
    x1: int
    y1: int
    x2: int
    y2: int
    layer: str

    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class NetGeometry:
    """All rectangles belonging to a single net."""
    name: str
    rects: List[Rect] = field(default_factory=list)

    def rects_by_layer(self) -> Dict[str, List[Rect]]:
        result: Dict[str, List[Rect]] = {}
        for r in self.rects:
            result.setdefault(r.layer, []).append(r)
        return result

    def layers_present(self) -> set:
        return {r.layer for r in self.rects}


def parse_polygons(filepath: str) -> Dict[str, NetGeometry]:
    """Parse polygons.txt file into dict of net_name -> NetGeometry."""
    nets: Dict[str, NetGeometry] = {}
    current_net = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("Net:"):
                net_name = line.split(":", 1)[1].strip()
                current_net = NetGeometry(name=net_name)
                nets[net_name] = current_net

            elif line.strip().startswith("rect") and current_net is not None:
                parts = line.split()
                rect = Rect(
                    x1=int(parts[1]),
                    y1=int(parts[2]),
                    x2=int(parts[3]),
                    y2=int(parts[4]),
                    layer=parts[5],
                )
                current_net.rects.append(rect)

    return nets
