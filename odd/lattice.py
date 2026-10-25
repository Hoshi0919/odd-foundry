"""Discrete active odd spring network lattice dynamics."""

import numpy as np


class OddLattice:
    """Discrete 2D mass-spring network with active non-reciprocal odd transverse forces.

    Each bond (i, j) exerts:
        F_{i <- j} = k * (d - l0) * r_hat + k_odd * (d - l0) * n_hat - gamma * (v_i - v_j)
        F_{j <- i} = -F_{i <- j}  (linear momentum conserved, angular momentum injected)
    """

    def __init__(self, n_nodes: int, node_mass: float = 1.0, substrate_damping: float = 0.1):
        self.n_nodes = n_nodes
        self.masses = np.full(n_nodes, float(node_mass), dtype=float)
        self.positions = np.zeros((n_nodes, 2), dtype=float)
        self.velocities = np.zeros((n_nodes, 2), dtype=float)
        self.forces = np.zeros((n_nodes, 2), dtype=float)
        self.fixed = np.zeros(n_nodes, dtype=bool)
        self.substrate_damping = float(substrate_damping)

        # Bonds: list of dicts: {'i': int, 'j': int, 'l0': float, 'k': float, 'k_odd': float, 'gamma': float}
        self.bonds = []

    def set_position(self, node_idx: int, x: float, y: float, fixed: bool = False):
        self.positions[node_idx] = [x, y]
        self.fixed[node_idx] = fixed

    def add_bond(self, i: int, j: int, l0: float | None = None, k: float = 50.0, k_odd: float = 15.0, gamma: float = 0.5):
        if l0 is None:
            l0 = float(np.linalg.norm(self.positions[i] - self.positions[j]))
        self.bonds.append({
            'i': int(i),
            'j': int(j),
            'l0': float(l0),
            'k': float(k),
            'k_odd': float(k_odd),
            'gamma': float(gamma),
        })

    def compute_forces(self) -> tuple[np.ndarray, float]:
        """Calculates total net forces on all nodes and the instantaneous active internal torque."""
        self.forces.fill(0.0)
        total_internal_torque = 0.0

        for b in self.bonds:
            i, j = b['i'], b['j']
            pos_i = self.positions[i]
            pos_j = self.positions[j]
            r_vec = pos_j - pos_i
            dist = np.linalg.norm(r_vec)
            if dist < 1e-12:
                continue

            r_hat = r_vec / dist
            # Rotate 90 degrees counter-clockwise for transverse direction
            n_hat = np.array([-r_hat[1], r_hat[0]])

            delta_l = dist - b['l0']
            v_rel = self.velocities[i] - self.velocities[j]

            # Longitudinal (even) + Transverse (odd) + Bond damping
            f_even = b['k'] * delta_l * r_hat
            f_odd = b['k_odd'] * delta_l * n_hat
            f_damp = -b['gamma'] * np.dot(v_rel, r_hat) * r_hat

            f_bond_on_i = f_even + f_odd + f_damp
            self.forces[i] += f_bond_on_i
            self.forces[j] -= f_bond_on_i

            # Bond internal torque: r_vec x f_odd = dist * k_odd * delta_l
            torque_bond = -b['k_odd'] * dist * delta_l
            total_internal_torque += torque_bond

        # Substrate viscous friction
        for i in range(self.n_nodes):
            if not self.fixed[i]:
                self.forces[i] -= self.substrate_damping * self.velocities[i]
            else:
                self.forces[i].fill(0.0)

        return self.forces, total_internal_torque

    def step(self, dt: float):
        """Semi-implicit Euler / Verlet step for numerical stability in active lattices."""
        self.compute_forces()
        for i in range(self.n_nodes):
            if not self.fixed[i]:
                acc = self.forces[i] / self.masses[i]
                self.velocities[i] += acc * dt
                self.positions[i] += self.velocities[i] * dt

    def total_kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.masses[:, None] * (self.velocities ** 2)))
