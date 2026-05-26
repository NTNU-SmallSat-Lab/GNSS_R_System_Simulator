"""Target abstract base class for bistatic RCS models (ocean, ship, ...).

A `Target` knows how to turn a `BistaticGeometry` plus carrier wavelength into a per-sample
RCS in m^2, with optional target-specific diagnostic info. Concrete implementations live in
sibling `_def.py` modules (e.g. `OceanTarget_def.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry


@dataclass(frozen=True, slots=True)
class TargetRcs:
    """Result of a `Target.compute_rcs` call: RCS per sample plus target-specific diagnostics."""

    rcs_m2: np.ndarray
    info: dict = field(default_factory=dict)


class Target(ABC):
    """Abstract base class for any scattering target (ocean surface, ship, etc.)."""

    @abstractmethod
    def compute_rcs(self, geometry: "BistaticGeometry", wavelength_m: float) -> TargetRcs:
        """Return per-sample RCS in m^2 for the given bistatic geometry and wavelength."""
