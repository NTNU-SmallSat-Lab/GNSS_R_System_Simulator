"""Generate the phased-array radiation-pattern figure for antenna.html, straight from the code.

Uses the simulator's own ArrayFactor (object_definitions.AntennaPatternGenerator_def) so the figure
is exactly the pattern the link budget sees. Plots the principal-plane (phi=0) array-factor cut in dB,
normalised to a 0 dB peak, for a boresight beam and a beam steered to a scan angle, as a Cartesian cut
and a polar cut. Annotates the uniform-excitation first side-lobe level (~ -13 dB). Output PNG lands in
docs/source/images/.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: just write the PNG

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "SRC"))  # entry-point scripts live inside SRC/

from object_definitions.AntennaPatternGenerator_def import ArrayFactor  # noqa: E402
from object_definitions.Constants_def import Constants  # noqa: E402

# palette (consistent with the other docs schematics)
BORE = "#214f7b"     # boresight beam
SCAN = "#c8862a"     # steered beam
SIDELOBE = "#5a6b7a"  # side-lobe annotation
GRID = "#d4dde6"

NX = NY = 8                 # elements per axis (square aperture)
FREQ_MHZ = 1575.42         # GPS L1 carrier
SCAN_DEG = 30.0            # steered-beam scan angle in the phi=0 plane
DB_FLOOR = -40.0          # plot floor


def principal_cut(scan_deg):
    """Normalised (0 dB peak) array-factor dB cut in the phi=0 plane vs signed off-boresight angle.

    Returns (angle_deg, pattern_dB) over -90..+90 deg. Negative angles use phi=180 deg so the cut
    spans both sides of boresight; the x-axis steering phase places the main lobe at scan_deg.
    """
    freq_hz = FREQ_MHZ * Constants.MHZ_TO_HZ
    wavelength_m = Constants.SPEED_OF_LIGHT / freq_hz
    d = wavelength_m / 2.0                                 # half-wavelength element spacing
    k = 2.0 * np.pi / wavelength_m                         # wavenumber
    beta_x = -k * d * np.sin(np.radians(scan_deg))         # progressive phase steering the peak to scan_deg

    angle_deg = np.linspace(-90.0, 90.0, 2001)            # signed angle off boresight in the cut plane
    theta = np.radians(np.abs(angle_deg))                  # polar angle (>= 0)
    phi = np.where(angle_deg >= 0.0, 0.0, np.pi)           # +x half vs -x half of the cut

    af = ArrayFactor(NX, NY, freq_hz, d, d, beta_x, 0.0)
    pattern_dB = af.calculate_array_factor(theta, phi).ravel()
    return angle_deg, pattern_dB


def main():
    fig = plt.figure(figsize=(11.5, 4.8))

    ang, bore_dB = principal_cut(0.0)
    _, scan_dB = principal_cut(SCAN_DEG)

    # uniform-array first side-lobe level of the boresight pattern (highest lobe away from the main beam)
    main_mask = np.abs(ang) > 12.0                         # exclude the main lobe neighbourhood
    sidelobe_dB = bore_dB[main_mask].max()

    # -- Cartesian cut ------------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(ang, bore_dB, color=BORE, lw=2.0, label="boresight")
    ax1.plot(ang, scan_dB, color=SCAN, lw=2.0, ls="--", label=f"steered {SCAN_DEG:.0f} deg")
    ax1.axhline(sidelobe_dB, color=SIDELOBE, lw=1.0, ls=":")
    ax1.text(-88.0, sidelobe_dB + 1.2, f"first side lobe ~ {sidelobe_dB:.0f} dB (uniform excitation)",
             color=SIDELOBE, fontsize=9, va="bottom")
    ax1.set_xlim(-90.0, 90.0)
    ax1.set_ylim(DB_FLOOR, 2.0)
    ax1.set_xlabel("angle off boresight in the $\\phi=0$ plane [deg]")
    ax1.set_ylabel("normalised array factor [dB]")
    ax1.set_xticks(np.arange(-90, 91, 30))
    ax1.grid(True, color=GRID, lw=0.8)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # -- Polar cut ----------------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.plot(np.radians(ang), np.maximum(bore_dB, DB_FLOOR), color=BORE, lw=2.0)
    ax2.plot(np.radians(ang), np.maximum(scan_dB, DB_FLOOR), color=SCAN, lw=2.0, ls="--")
    ax2.set_theta_zero_location("N")                       # boresight at the top
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    ax2.set_ylim(DB_FLOOR, 0.0)
    ax2.set_yticks([-30, -20, -10, 0])
    ax2.set_title("principal-plane cut [dB]", fontsize=10, pad=12)

    fig.suptitle(f"Phased-array radiation pattern ({NX}x{NY} elements, $d=\\lambda/2$ at L1)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out = os.path.join(REPO_ROOT, "docs", "source", "images", "antenna_pattern.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
