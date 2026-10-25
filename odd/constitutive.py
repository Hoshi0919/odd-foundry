r"""Continuum 2D isotropic odd elasticity constitutive relations and strain-cycle thermodynamics."""

import numpy as np


class OddElasticMedium:
    r"""Represents a 2D isotropic medium with active odd elasticity.

    Constitutive relation in the (v, s, a) basis:
        sigma_v = K * epsilon_v
        sigma_s = mu * epsilon_s + A_o * epsilon_a
        sigma_a = -A_o * epsilon_s + mu * epsilon_a

    where:
        epsilon_v = eps_xx + eps_yy (volumetric strain)
        epsilon_s = eps_xx - eps_yy (deviatoric normal shear)
        epsilon_a = 2 * eps_xy     (deviatoric pure shear)
        K   = bulk modulus
        mu  = even shear modulus
        A_o = odd shear modulus (breaks Maxwell-Betti reciprocity)
    """

    def __init__(self, bulk_modulus: float = 10.0, shear_modulus: float = 5.0, odd_modulus: float = 2.0):
        if bulk_modulus <= 0:
            raise ValueError("Bulk modulus K must be strictly positive")
        if shear_modulus <= 0:
            raise ValueError("Shear modulus mu must be strictly positive")
        self.K = float(bulk_modulus)
        self.mu = float(shear_modulus)
        self.A_o = float(odd_modulus)

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Returns the 3x3 stiffness matrix in the (v, s, a) basis."""
        return np.array([
            [self.K, 0.0, 0.0],
            [0.0, self.mu, self.A_o],
            [0.0, -self.A_o, self.mu],
        ], dtype=float)

    @property
    def is_reciprocal(self) -> bool:
        """Returns True if the material obeys Green hyperelasticity (A_o == 0)."""
        return abs(self.A_o) < 1e-12

    def compute_stress(self, eps_xx: float, eps_yy: float, eps_xy: float) -> tuple[float, float, float]:
        """Calculates Cartesian stress tensor components (sigma_xx, sigma_yy, sigma_xy)."""
        eps_v = eps_xx + eps_yy
        eps_s = eps_xx - eps_yy
        eps_a = 2.0 * eps_xy

        sigma_v = self.K * eps_v
        sigma_s = self.mu * eps_s + self.A_o * eps_a
        sigma_a = -self.A_o * eps_s + self.mu * eps_a

        sigma_xx = sigma_v + 0.5 * sigma_s
        sigma_yy = sigma_v - 0.5 * sigma_s
        sigma_xy = 0.5 * sigma_a

        return sigma_xx, sigma_yy, sigma_xy

    def compute_cycle_work_analytical(self, amp_s: float, amp_a: float) -> float:
        r"""Computes work done ON the material per cycle for a CCW elliptical strain path:

        eps_s(t) = amp_s * cos(t)
        eps_a(t) = amp_a * sin(t)

        W = \oint \sigma : d\epsilon = -2 * pi * A_o * amp_s * amp_a.
        (A negative work on the medium implies positive mechanical work extracted).
        """
        return -2.0 * np.pi * self.A_o * amp_s * amp_a

    def compute_work_extracted_analytical(self, amp_s: float, amp_a: float) -> float:
        """Computes net mechanical work extracted from the odd engine per cycle."""
        return -self.compute_cycle_work_analytical(amp_s, amp_a)

    def compute_cycle_work_numerical(self, eps_s_series: np.ndarray, eps_a_series: np.ndarray) -> float:
        r"""Integrates work W = \oint (sigma_s * d(eps_s) + sigma_a * d(eps_a)) along a discrete closed trajectory."""
        n_pts = len(eps_s_series)
        if n_pts < 3:
            raise ValueError("Trajectory must have at least 3 points to form a closed cycle")

        sigma_s = self.mu * eps_s_series + self.A_o * eps_a_series
        sigma_a = -self.A_o * eps_s_series + self.mu * eps_a_series

        d_eps_s = np.diff(eps_s_series, append=eps_s_series[0])
        d_eps_a = np.diff(eps_a_series, append=eps_a_series[0])

        # Midpoint integration around the closed loop
        work = np.sum(0.5 * (sigma_s + np.roll(sigma_s, -1)) * d_eps_s +
                      0.5 * (sigma_a + np.roll(sigma_a, -1)) * d_eps_a)
        return float(work)

    def eigenvalues(self) -> np.ndarray:
        """Returns complex eigenvalues of the stiffness matrix."""
        return np.array([self.K, complex(self.mu, self.A_o), complex(self.mu, -self.A_o)])
