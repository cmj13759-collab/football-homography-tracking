from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FootballFieldConfig:
    field_length: float = 120.0
    field_width: float = 160.0 / 3.0
    number_stripe_offset: float = 13.0
    yard_numbers: Tuple[int, ...] = (10, 20, 30, 40, 50)
    hash_offset: float = 23.5833  # yards from sideline (NFL)

    @property
    def y_bottom_hash(self) -> float:
        return self.hash_offset

    @property
    def y_top_hash(self) -> float:
        return self.field_width - self.hash_offset

    @property
    def y_bottom_sideline(self) -> float:
        return 0.0

    @property
    def y_top_sideline(self) -> float:
        return self.field_width

    @property
    def y_bottom(self) -> float:
        return self.number_stripe_offset

    @property
    def y_top(self) -> float:
        return self.field_width - self.number_stripe_offset

    def x_left(self, yard: int) -> float:
        return 10.0 + float(yard)  # includes endzone

    def x_right(self, yard: int) -> float:
        return self.field_length - (10.0 + float(yard))

    def x_midfield(self) -> float:
        return self.field_length / 2.0

    # ---- This makes it feel like basketball: vertices + edges/labels/colors ----
    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        Ordered as:
          10: BL, TL, BR, TR
          20: BL, TL, BR, TR
          ...
          50: B, T (no LR)
        """
        verts: List[Tuple[float, float]] = []

        for yd in self.yard_numbers:
            if yd == 50:
                x = self.x_midfield()
                verts.append((x, self.y_bottom))  # 50_B
                verts.append((x, self.y_top))     # 50_T
                continue

            xl, xr = self.x_left(yd), self.x_right(yd)
            verts.extend([
                (xl, self.y_bottom),  # {yd}_BL
                (xl, self.y_top),     # {yd}_TL
                (xr, self.y_bottom),  # {yd}_BR
                (xr, self.y_top),     # {yd}_TR
            ])

        return verts

    @property
    def vertex_names(self) -> List[str]:
        names: List[str] = []
        for yd in self.yard_numbers:
            if yd == 50:
                names += ["50_BOTTOM", "50_TOP"]
            else:
                names += [f"{yd}_BL", f"{yd}_TL", f"{yd}_BR", f"{yd}_TR"]
        return names

    @property
    def vertex_index(self) -> Dict[str, int]:
        return {n: i for i, n in enumerate(self.vertex_names)}

    # optional, for debugging visuals (can be empty)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)

    def target_xy(self, yard: int, variant: str) -> Tuple[float, float]:
        x_shift = 0.0
        if yard in self.yard_numbers:  # ensures only 10/20/30/40/50
            if variant.startswith("TOP"):
                x_shift = +1.0
            elif variant.startswith("BOTTOM"):
                x_shift = -1.0

        if yard == 50:
            x = self.x_midfield() + x_shift
            y = self.y_top if variant.startswith("TOP") else self.y_bottom
            return (x, y)

        if variant == "TOP_LEFT":
            return (self.x_left(yard) + x_shift, self.y_top)

        if variant == "BOTTOM_LEFT":
            return (self.x_left(yard) + x_shift, self.y_bottom)

        if variant == "TOP_RIGHT":
            return (self.x_right(yard) + x_shift, self.y_top)

        if variant == "BOTTOM_RIGHT":
            return (self.x_right(yard) + x_shift, self.y_bottom)

        raise ValueError(f"Invalid variant {variant!r}")