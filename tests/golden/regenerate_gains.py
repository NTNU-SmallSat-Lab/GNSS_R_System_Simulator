"""Regenerate the antenna-gain golden file.

Run this manually when:
  * the expected fixed-nadir gain-pipeline behaviour intentionally changes (e.g. JSON
    sampling, multi-band, polarisation handling); OR
  * the test fixture itself is updated (new sampling grid, new pattern shape).

Usage (from repo root):
    python tests/golden/regenerate_gains.py

This script intentionally lives outside the pytest collection (no `test_` prefix) so it never
runs as part of the test suite.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "SRC"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from antenna_fixture import compute_gain_grid, gain_sampling_grid, make_test_antenna
import numpy as np


def main() -> None:
    antenna = make_test_antenna()
    gains = compute_gain_grid(antenna)
    theta_deg, phi_deg, az_steer, el_steer = gain_sampling_grid()

    out_path = Path(__file__).parent / "reference_gains.npz"
    np.savez(
        out_path,
        gains=gains,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        az_steer=az_steer,
        el_steer=el_steer,
    )
    finite = np.isfinite(gains)
    print(
        f"Wrote {out_path}\n"
        f"  shape={gains.shape}, finite={finite.sum()}/{gains.size}, "
        f"min={np.nanmin(gains):.4f}, max={np.nanmax(gains):.4f}"
    )


if __name__ == "__main__":
    main()
