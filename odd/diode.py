r"""Non-reciprocal acoustic metamaterial diode and shock isolator."""

import numpy as np


class MechanicalDiode:
    r"""Simulates a 1D non-reciprocal mechanical waveguide chain with active odd coupling.

    The equation of motion for node j:
        m * \ddot{u}_j = k * (u_{j+1} - 2 u_j + u_{j-1}) - alpha_odd * (u_{j+1} - u_{j-1}) - gamma * \dot{u}_j

    The asymmetric coupling term -alpha_odd * (u_{j+1} - u_{j-1}) creates a directional
    convective flow of phonons:
        Effective left-coupling:  k + alpha_odd
        Effective right-coupling: k - alpha_odd

    This provides strong acoustic diode rectification: forward shockwaves propagate smoothly,
    while backward impact shockwaves are exponentially extinguished.
    """

    def __init__(
        self,
        n_sites: int = 12,
        mass: float = 1.0,
        spring_k: float = 30.0,
        odd_alpha: float = 8.0,
        damping: float = 1.0,
    ):
        self.n_sites = int(n_sites)
        self.m = float(mass)
        self.k = float(spring_k)
        self.alpha = float(odd_alpha)
        self.gamma = float(damping)

    def simulate_pulse(self, source_direction: str = 'forward', duration: float = 2.0, dt: float = 0.002) -> dict:
        """Simulates wave packet propagation across the lattice.

        source_direction:
            'forward': impulse injected at site 0, measured at site n_sites-1
            'backward': impulse injected at site n_sites-1, measured at site 0
        """
        n_steps = int(duration / dt)
        u = np.zeros(self.n_sites, dtype=float)
        v = np.zeros(self.n_sites, dtype=float)

        transmitted_energy = 0.0
        input_energy = 0.0

        for step in range(n_steps):
            t = step * dt
            if t < 0.2:
                impulse = (np.sin(np.pi * t / 0.2) ** 2) * np.sin(2.0 * np.pi * 4.0 * t) * 5.0
            else:
                impulse = 0.0

            acc = np.zeros(self.n_sites, dtype=float)

            for j in range(self.n_sites):
                u_left = u[j - 1] if j > 0 else 0.0
                u_right = u[j + 1] if j < self.n_sites - 1 else 0.0

                f_even = self.k * (u_right - 2.0 * u[j] + u_left)
                f_odd = -self.alpha * (u_right - u_left)
                f_damp = -self.gamma * v[j]

                f_drive = 0.0
                if source_direction == 'forward' and j == 0:
                    f_drive = impulse
                elif source_direction == 'backward' and j == self.n_sites - 1:
                    f_drive = impulse

                acc[j] = (f_even + f_odd + f_damp + f_drive) / self.m

            v += acc * dt
            u += v * dt

            if source_direction == 'forward':
                transmitted_energy += (v[-1] ** 2) * dt
                input_energy += (v[0] ** 2) * dt
            else:
                transmitted_energy += (v[0] ** 2) * dt
                input_energy += (v[-1] ** 2) * dt

        transmittance = float(transmitted_energy / (input_energy if input_energy > 1e-9 else 1e-9))
        return {
            'direction': source_direction,
            'input_energy': float(input_energy),
            'transmitted_energy': float(transmitted_energy),
            'transmittance': transmittance,
        }

    def compute_rectification(self) -> dict:
        """Measures forward vs backward transmission and computes the isolation rectification ratio in dB."""
        res_fwd = self.simulate_pulse('forward')
        res_bwd = self.simulate_pulse('backward')

        t_fwd = max(res_fwd['transmittance'], 1e-6)
        t_bwd = max(res_bwd['transmittance'], 1e-6)

        rectification_db = 10.0 * np.log10(t_fwd / t_bwd)
        return {
            'forward_transmittance': t_fwd,
            'backward_transmittance': t_bwd,
            'rectification_ratio_db': float(rectification_db),
            'is_diode_active': bool(rectification_db > 6.0),
        }
