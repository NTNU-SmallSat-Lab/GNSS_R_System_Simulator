"""
Golden-file regression test for the antenna gain pipeline.

Captures the output of RxAntenna.get_gain_dbi_for_direction over a fixed grid of inputs in
a synthesized antenna setup, and compares to a stored reference. Any change to the gain
pipeline (bilinear sampler, steering bracket, directivity scaling, future on-demand steering)
that drifts numerically will trip this test.

To accept a new reference (after an intentional change), run
    python tests/golden/regenerate_gains.py
and commit the updated tests/golden/reference_gains.npz.
"""

from pathlib import Path

import numpy as np
import pytest

from antenna_fixture import compute_gain_grid, make_test_antenna


GOLDEN_PATH = Path(__file__).parent / "golden" / "reference_gains.npz"


@pytest.mark.golden
def test_antenna_gain_matches_golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden reference {GOLDEN_PATH} does not exist. "
            "Run `python tests/golden/regenerate_gains.py` to create it."
        )

    reference = np.load(GOLDEN_PATH)
    ref_gains = reference["gains"]

    antenna = make_test_antenna()
    current = compute_gain_grid(antenna)

    # NaN pattern (e.g. behind-antenna or beyond-max-steering) must match
    np.testing.assert_array_equal(
        np.isnan(current), np.isnan(ref_gains),
        err_msg="NaN pattern differs from golden reference.",
    )

    # Finite gain values must match to ~math precision (post-bilinear refactor; pre any (c) change)
    finite = ~np.isnan(current)
    np.testing.assert_allclose(
        current[finite], ref_gains[finite],
        atol=1e-10, rtol=1e-10,
        err_msg="Gain values differ from golden reference. If the change was intentional, "
                "run tests/golden/regenerate_gains.py to update the baseline.",
    )
