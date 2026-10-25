"""Autonomous non-reciprocal traveling wave conveyor driven by active odd elasticity."""

import numpy as np


class WaveConveyor:
    """Simulates a solid odd-elastic substrate that self-propels cargo via chiral surface waves.

    Surface nodes undergo retrograde chiral oscillations:
        u_x(x, t) =  A_x * sin(omega * t - k_wave * x)
        u_y(x, t) =  A_y * cos(omega * t - k_wave * x)

    This phase shift ensures that when a node moves forward (v_x > 0),
    it is at the wave crest (u_y > 0) pushing up against the cargo with maximal normal force.
    When the node moves backward (v_x < 0), it is in the wave trough, disengaged from cargo.
    """

    def __init__(
        self,
        length: float = 20.0,
        n_segments: int = 50,
        wave_speed: float = 4.0,
        wave_length: float = 5.0,
        amp_x: float = 0.15,
        amp_y: float = 0.20,
        friction_coeff: float = 0.4,
        gravity: float = 9.81,
    ):
        self.length = float(length)
        self.n_segments = int(n_segments)
        self.dx = self.length / self.n_segments
        self.nodes_x = np.linspace(0, self.length, self.n_segments)
        self.c_wave = float(wave_speed)
        self.lambda_wave = float(wave_length)
        self.k_wave = 2.0 * np.pi / self.lambda_wave
        self.omega = self.c_wave * self.k_wave

        self.amp_x = float(amp_x)
        self.amp_y = float(amp_y)
        self.mu_f = float(friction_coeff)
        self.g = float(gravity)

        self.time = 0.0

    def get_surface_state(self, x: float, t: float) -> tuple[float, float, float, float]:
        """Returns (u_x, u_y, v_x, v_y) of the surface at position x and time t."""
        phase = self.omega * t - self.k_wave * x
        u_x = self.amp_x * np.sin(phase)
        u_y = self.amp_y * np.cos(phase)
        v_x = self.amp_x * self.omega * np.cos(phase)
        v_y = -self.amp_y * self.omega * np.sin(phase)
        return float(u_x), float(u_y), float(v_x), float(v_y)

    def simulate_cargo(self, x0: float = 1.0, v0: float = 0.0, duration: float = 4.0, dt: float = 0.005) -> dict:
        """Simulates cargo trajectory on the odd wave conveyor."""
        n_steps = int(duration / dt)
        t_arr = np.linspace(0, duration, n_steps)
        x_arr = np.zeros(n_steps)
        v_arr = np.zeros(n_steps)

        x = float(x0)
        v = float(v0)

        for step in range(n_steps):
            t = t_arr[step]
            x_arr[step] = x
            v_arr[step] = v

            # Contact dynamics
            _, u_y, v_surf_x, _ = self.get_surface_state(x, t)

            # Normal load is active during crest, zero in trough
            n_ratio = max(0.0, 1.0 + (u_y / self.amp_y))
            normal_force = self.g * n_ratio

            # Relative slip velocity
            v_rel = v_surf_x - v
            slip_dir = np.tanh(v_rel / 0.1)
            f_fric = self.mu_f * normal_force * slip_dir

            # Accelerate cargo
            acc = f_fric
            v += acc * dt
            x += v * dt

            # Wrap or clamp
            if x >= self.length:
                x = x % self.length

        mean_speed = float(np.mean(v_arr[int(n_steps * 0.5):]))
        return {
            'time': t_arr,
            'x': x_arr,
            'velocity': v_arr,
            'steady_state_speed': mean_speed,
            'transport_efficiency': mean_speed / self.c_wave,
        }
