"""Anti-jamming hopper dynamics utilizing odd-elastic transverse shear stresses."""

import numpy as np


class OddHopper:
    """Simulates granular discharge through a narrow bottleneck hopper with odd shear coupling.

    In classical hoppers (odd_coeff == 0), static Coulomb friction forces particles
    into rigid force-chain arches (Janssen effect). In an odd hopper (odd_coeff > 0),
    compressive normal stress induces an active transverse shear force:
        F_transverse = odd_coeff * F_n * t_hat

    This active shear overcomes static friction, breaking the collinear symmetry
    of the arch and restoring continuous fluidic discharge.
    """

    def __init__(
        self,
        n_particles: int = 20,
        hopper_width: float = 3.0,
        orifice_width: float = 0.80,
        hopper_height: float = 4.0,
        particle_radius: float = 0.25,
        odd_coeff: float = 0.40,
        gravity: float = 9.81,
    ):
        self.n_particles = int(n_particles)
        self.W_top = float(hopper_width)
        self.W_exit = float(orifice_width)
        self.H = float(hopper_height)
        self.r_p = float(particle_radius)
        self.odd_coeff = float(odd_coeff)
        self.g = float(gravity)

        np.random.seed(42)
        self.pos = np.zeros((self.n_particles, 2), dtype=float)
        self.vel = np.zeros((self.n_particles, 2), dtype=float)
        self.active = np.ones(self.n_particles, dtype=bool)

        for i in range(self.n_particles):
            self.pos[i] = [
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.8, 3.2),
            ]

        self.discharged_count = 0
        self.jammed_duration = 0.0

    def step(self, dt: float = 0.005) -> tuple[int, bool]:
        """Advances particle simulation by dt. Returns (discharged_this_step, is_currently_jammed)."""
        forces = np.zeros_like(self.pos)
        forces[:, 1] -= self.g * self.active

        # Funnel wall contact with friction and odd transverse kick
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            x, y = self.pos[i]

            if 0.0 <= y <= self.H:
                half_w = 0.5 * (self.W_exit + (self.W_top - self.W_exit) * (y / self.H))
                # Left wall
                if x - self.r_p < -half_w:
                    overlap = (-half_w) - (x - self.r_p)
                    fn = 800.0 * overlap
                    ft = -min(0.6 * fn, 30.0 * max(0.0, self.vel[i, 1]))
                    forces[i, 0] += fn
                    forces[i, 1] += ft - self.odd_coeff * fn
                # Right wall
                elif x + self.r_p > half_w:
                    overlap = (x + self.r_p) - half_w
                    fn = 800.0 * overlap
                    ft = -min(0.6 * fn, 30.0 * max(0.0, self.vel[i, 1]))
                    forces[i, 0] -= fn
                    forces[i, 1] += ft - self.odd_coeff * fn

        # Particle-particle contact with Coulomb friction + Odd transverse shear
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            for j in range(i + 1, self.n_particles):
                if not self.active[j]:
                    continue
                d_vec = self.pos[j] - self.pos[i]
                dist = np.linalg.norm(d_vec)
                min_dist = 2.0 * self.r_p

                if 0.0 < dist < min_dist:
                    overlap = min_dist - dist
                    n_hat = d_vec / dist
                    t_hat = np.array([-n_hat[1], n_hat[0]])

                    fn = 1000.0 * overlap

                    # Coulomb friction opposes relative tangential movement
                    v_rel = self.vel[j] - self.vel[i]
                    v_t = np.dot(v_rel, t_hat)
                    f_fric = -np.sign(v_t) * min(0.7 * fn, 20.0 * abs(v_t))

                    # Active odd shear force
                    f_odd = self.odd_coeff * fn

                    f_total = fn * n_hat + (f_fric + f_odd) * t_hat

                    forces[i] -= f_total
                    forces[j] += f_total

        # Viscous drag
        forces -= 1.0 * self.vel

        discharged = 0
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            self.vel[i] += forces[i] * dt
            self.pos[i] += self.vel[i] * dt

            if self.pos[i, 1] < 0.0:
                self.active[i] = False
                self.discharged_count += 1
                discharged += 1

        active_mask = self.active & (self.pos[:, 1] < 1.5) & (self.pos[:, 1] > 0.0)
        n_in_throat = int(np.sum(active_mask))

        if n_in_throat >= 2:
            mean_speed = np.mean(np.linalg.norm(self.vel[active_mask], axis=1))
            is_jammed = bool(mean_speed < 0.06)
        else:
            is_jammed = False

        if is_jammed:
            self.jammed_duration += dt
        else:
            self.jammed_duration = 0.0

        return discharged, is_jammed

    def simulate_discharge(self, max_time: float = 3.0, dt: float = 0.005) -> dict:
        """Simulates hopper until complete or max_time reached."""
        n_steps = int(max_time / dt)
        t_arr = np.linspace(0, max_time, n_steps)
        jammed_steps = 0

        for _ in range(n_steps):
            _, is_jammed = self.step(dt)
            if is_jammed:
                jammed_steps += 1

        discharge_rate = self.discharged_count / max_time

        return {
            'discharged_count': self.discharged_count,
            'discharge_rate': discharge_rate,
            'jammed_steps': jammed_steps,
            'has_continuous_flow': discharge_rate > 1.0,
        }
