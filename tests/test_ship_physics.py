"""Unit tests for ship_physics helpers (flat-plate PO RCS, 90-deg dihedral RCS)."""

import numpy as np

from functions.ship_physics import dihedral_rcs, flat_plate_rcs

WAVELENGTH_M = 0.19           # rough GPS L1 wavelength
EX = np.array([[1.0, 0.0, 0.0]])
EY = np.array([[0.0, 1.0, 0.0]])
EZ = np.array([[0.0, 0.0, 1.0]])


class TestFlatPlateRcs:
    def test_normal_incidence_backscatter_peak(self):
        # Plate at origin facing +z, square 5 m x 5 m. Tx and Rx co-located directly above
        # (monostatic, normal incidence) -> sigma = 4 pi A^2 / lambda^2.
        L = 5.0
        n_tx2s = -EZ           # light travels -z (down) into the plate
        n_s2rx = EZ            # scattered back upward
        sigma = flat_plate_rcs(
            n_plate=EZ, edge_1=EX, edge_2=EY,
            length_edge_1_m=L, length_edge_2_m=L,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        expected = 4.0 * np.pi * (L * L) ** 2 / WAVELENGTH_M**2
        np.testing.assert_allclose(sigma, [expected], rtol=1e-12)

    def test_edge_on_geometry_gives_zero(self):
        # Plate facing +z; source and receiver in the plate plane (looking edge-on).
        # Visibility: -n_tx2s . n_plate = +x . +z = 0  => not visible.
        n_tx2s = -EX           # light goes along +x into the plate edge
        n_s2rx = EX
        sigma = flat_plate_rcs(
            n_plate=EZ, edge_1=EX, edge_2=EY,
            length_edge_1_m=1.0, length_edge_2_m=1.0,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        np.testing.assert_array_equal(sigma, [0.0])

    def test_back_face_culling(self):
        # Plate's normal points +z but Tx is below the plate -> not visible.
        n_tx2s = EZ            # light comes from below, going up
        n_s2rx = -EZ
        sigma = flat_plate_rcs(
            n_plate=EZ, edge_1=EX, edge_2=EY,
            length_edge_1_m=2.0, length_edge_2_m=2.0,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        np.testing.assert_array_equal(sigma, [0.0])

    def test_oblique_incidence_drops_via_cos_alpha(self):
        # Plate facing +z. Monostatic but at 30 deg incidence (tilted source/receiver).
        # cos(alpha) where alpha = angle between plate normal and bisector should be cos(30 deg).
        theta = np.deg2rad(30.0)
        n_tx2s = np.array([[np.sin(theta), 0.0, -np.cos(theta)]])
        n_s2rx = -n_tx2s
        L = 1.0
        sigma = flat_plate_rcs(
            n_plate=EZ, edge_1=EX, edge_2=EY,
            length_edge_1_m=L, length_edge_2_m=L,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        # Closed form: 4 pi A^2/lambda^2 * cos^2(theta) * sinc^2(2 L sin(theta)/lambda) * sinc^2(0)
        peak = 4.0 * np.pi * (L * L) ** 2 / WAVELENGTH_M**2
        expected = peak * np.cos(theta) ** 2 * np.sinc(2.0 * L * np.sin(theta) / WAVELENGTH_M) ** 2
        np.testing.assert_allclose(sigma, [expected], rtol=1e-10)

    def test_vectorised_over_samples(self):
        # Two samples: one normal-incidence, one back-face. Returns are independent.
        n_plate = np.vstack([EZ, EZ])
        edge_1 = np.vstack([EX, EX])
        edge_2 = np.vstack([EY, EY])
        n_tx2s = np.vstack([-EZ, EZ])      # second sample is back-face
        n_s2rx = np.vstack([EZ, -EZ])
        sigma = flat_plate_rcs(
            n_plate=n_plate, edge_1=edge_1, edge_2=edge_2,
            length_edge_1_m=1.0, length_edge_2_m=1.0,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        assert sigma.shape == (2,)
        assert sigma[0] > 0.0
        assert sigma[1] == 0.0


class TestDihedralRcs:
    def test_perpendicular_monostatic_peak(self):
        # Dihedral with edge along x, legs +y and +z. Monostatic source/receiver directly along
        # the bisector of the two leg normals (i.e., at 45 deg between +y and +z).
        # n_tx2s points INTO the corner (negative components on both leg normals).
        L = 10.0
        H = 3.0
        n_tx2s = np.array([[0.0, -np.sqrt(0.5), -np.sqrt(0.5)]])      # into the corner
        n_s2rx = -n_tx2s                                              # monostatic backscatter
        sigma = dihedral_rcs(
            edge_hat=EX, leg_1_normal=EY, leg_2_normal=EZ,
            length_edge_m=L, leg_dimension_m=H,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        expected = 8.0 * np.pi * (L * H) ** 2 / WAVELENGTH_M**2
        np.testing.assert_allclose(sigma, [expected], rtol=1e-12)

    def test_outside_wedge_gives_zero(self):
        # Source on the closed side of the wedge (negative leg-1 normal direction).
        n_tx2s = np.array([[0.0, +np.sqrt(0.5), -np.sqrt(0.5)]])      # -n_tx2s = (0, -sqrt(.5), +sqrt(.5))
        n_s2rx = -n_tx2s
        sigma = dihedral_rcs(
            edge_hat=EX, leg_1_normal=EY, leg_2_normal=EZ,
            length_edge_m=10.0, leg_dimension_m=3.0,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        np.testing.assert_array_equal(sigma, [0.0])

    def test_edge_component_sinc_falloff(self):
        # Bistatic case: bisector tilted along the edge direction -> sinc^2 reduces the peak.
        # Choose n_tx2s and n_s2rx such that (n_tx2s - n_s2rx) . edge_hat is nontrivial.
        # Use a small tilt where sinc argument == 1 (first zero of sinc).
        L = 10.0
        H = 3.0
        # We want |u_edge| = L / lambda * |(n_tx2s - n_s2rx) . edge_hat| = 1.0
        # Pick n_tx2s with x-component eps, n_s2rx = -n_tx2s + correction; simpler: both
        # mostly into corner but n_s2rx tilted in +x so the difference has a -x component.
        target_u = 1.0
        sum_edge_target = target_u * WAVELENGTH_M / L      # required |(difference).edge| component
        # n_tx2s = (-eps, -sqrt(.5), -sqrt(.5));  n_s2rx = (+eps + sum_edge_target, +sqrt(.5), +sqrt(.5))
        # so that (n_tx2s - n_s2rx).x = -sum_edge_target (eps = 0).
        eps = 0.0
        n_tx2s = np.array([[-eps, -np.sqrt(0.5), -np.sqrt(0.5)]])
        n_s2rx = np.array([[+eps + sum_edge_target, +np.sqrt(0.5), +np.sqrt(0.5)]])
        # Normalise both (small renormalisation, second one needs care)
        n_tx2s /= np.linalg.norm(n_tx2s, axis=1, keepdims=True)
        n_s2rx /= np.linalg.norm(n_s2rx, axis=1, keepdims=True)
        sigma = dihedral_rcs(
            edge_hat=EX, leg_1_normal=EY, leg_2_normal=EZ,
            length_edge_m=L, leg_dimension_m=H,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        # The sinc factor should suppress; exact value depends on the renormalisation, but
        # it must be << peak.
        peak = 8.0 * np.pi * (L * H) ** 2 / WAVELENGTH_M**2
        assert 0.0 < sigma[0] < 0.5 * peak

    def test_double_bounce_preserves_edge_component(self):
        # Incoming ray with edge component c: the double bounce flips y and z and keeps x,
        # so the peak lies at (c, +a, +a), while exact retro (-c, +a, +a) falls into the sinc sidelobes.
        L = 10.0
        H = 3.0
        c = 0.1
        a = np.sqrt((1.0 - c**2) / 2.0)
        n_tx2s = np.array([[c, -a, -a]])
        n_double_bounce = np.array([[c, +a, +a]])
        n_retro = -n_tx2s
        peak = 8.0 * np.pi * (L * H) ** 2 / WAVELENGTH_M**2
        kwargs = dict(edge_hat=EX, leg_1_normal=EY, leg_2_normal=EZ, length_edge_m=L, leg_dimension_m=H, wavelength_m=WAVELENGTH_M)
        sigma_bounce = dihedral_rcs(n_tx2s=n_tx2s, n_s2rx=n_double_bounce, **kwargs)
        sigma_retro = dihedral_rcs(n_tx2s=n_tx2s, n_s2rx=n_retro, **kwargs)
        np.testing.assert_allclose(sigma_bounce, [peak], rtol=1e-12)
        expected_retro = peak * np.sinc(L / WAVELENGTH_M * 2.0 * c) ** 2   # u_edge = L / lambda * 2c
        np.testing.assert_allclose(sigma_retro, [expected_retro], rtol=1e-9)
        assert sigma_retro[0] < 1e-2 * peak

    def test_vectorised_over_samples(self):
        n_tx2s = np.vstack([
            [0.0, -np.sqrt(0.5), -np.sqrt(0.5)],      # in wedge
            [0.0, +np.sqrt(0.5), -np.sqrt(0.5)],      # out of wedge (positive y component means -n_tx2s.y < 0)
        ])
        n_s2rx = -n_tx2s
        sigma = dihedral_rcs(
            edge_hat=np.vstack([EX, EX]),
            leg_1_normal=np.vstack([EY, EY]),
            leg_2_normal=np.vstack([EZ, EZ]),
            length_edge_m=10.0, leg_dimension_m=3.0,
            n_tx2s=n_tx2s, n_s2rx=n_s2rx,
            wavelength_m=WAVELENGTH_M,
        )
        assert sigma.shape == (2,)
        assert sigma[0] > 0.0
        assert sigma[1] == 0.0
