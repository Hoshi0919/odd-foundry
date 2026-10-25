"""Comprehensive test suite for Odd Foundry: active odd elasticity, non-reciprocity, and anti-jamming logistics."""

import pytest
import numpy as np

from odd.constitutive import OddElasticMedium
from odd.lattice import OddLattice
from odd.wave_conveyor import WaveConveyor
from odd.anti_jamming import OddHopper
from odd.strain_engine import StrainEngine
from odd.diode import MechanicalDiode
from odd.engine import OddFoundryEngine


class TestOddConstitutive:
    """Tests 2D isotropic continuum odd elasticity relations and cycle work thermodynamics."""

    def test_modulus_validation(self):
        with pytest.raises(ValueError, match="Bulk modulus"):
            OddElasticMedium(bulk_modulus=-1.0)
        with pytest.raises(ValueError, match="Shear modulus"):
            OddElasticMedium(shear_modulus=0.0)

    def test_stiffness_matrix_antisymmetry(self):
        medium = OddElasticMedium(bulk_modulus=12.0, shear_modulus=6.0, odd_modulus=2.5)
        mat = medium.stiffness_matrix
        assert mat[0, 0] == 12.0
        assert mat[1, 1] == 6.0
        assert mat[2, 2] == 6.0
        assert mat[1, 2] == 2.5
        assert mat[2, 1] == -2.5
        assert not medium.is_reciprocal

    def test_reciprocal_medium(self):
        medium = OddElasticMedium(bulk_modulus=10.0, shear_modulus=4.0, odd_modulus=0.0)
        assert medium.is_reciprocal
        mat = medium.stiffness_matrix
        np.testing.assert_array_equal(mat, mat.T)

    def test_stress_computation(self):
        medium = OddElasticMedium(bulk_modulus=10.0, shear_modulus=4.0, odd_modulus=3.0)
        # Pure volumetric strain (eps_xx=0.01, eps_yy=0.01, eps_xy=0)
        s_xx, s_yy, s_xy = medium.compute_stress(0.01, 0.01, 0.0)
        assert pytest.approx(s_xx, rel=1e-5) == 0.2
        assert pytest.approx(s_yy, rel=1e-5) == 0.2
        assert pytest.approx(s_xy, abs=1e-7) == 0.0

        # Normal shear strain (eps_xx=0.02, eps_yy=-0.02, eps_xy=0) -> eps_s = 0.04
        # sigma_s = 4.0 * 0.04 = 0.16; sigma_a = -3.0 * 0.04 = -0.12
        # sigma_xy = 0.5 * sigma_a = -0.06 (Odd cross-coupling generates shear stress from normal strain!)
        s_xx, s_yy, s_xy = medium.compute_stress(0.02, -0.02, 0.0)
        assert pytest.approx(s_xx, rel=1e-5) == 0.08
        assert pytest.approx(s_yy, rel=1e-5) == -0.08
        assert pytest.approx(s_xy, rel=1e-5) == -0.06

    def test_analytical_vs_numerical_cycle_work(self):
        medium = OddElasticMedium(bulk_modulus=10.0, shear_modulus=5.0, odd_modulus=4.0)
        amp_s = 0.05
        amp_a = 0.08
        analytical_w = medium.compute_cycle_work_analytical(amp_s, amp_a)

        # Numerical trajectory (CCW loop in strain space)
        theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        eps_s = amp_s * np.cos(theta)
        eps_a = amp_a * np.sin(theta)
        numerical_w = medium.compute_cycle_work_numerical(eps_s, eps_a)

        assert pytest.approx(numerical_w, rel=0.01) == analytical_w

    def test_work_extracted_positive(self):
        medium = OddElasticMedium(bulk_modulus=10.0, shear_modulus=5.0, odd_modulus=3.0)
        extracted = medium.compute_work_extracted_analytical(0.1, 0.1)
        assert extracted > 0.0
        assert pytest.approx(extracted, rel=1e-5) == 2.0 * np.pi * 3.0 * 0.01

    def test_zero_work_for_reciprocal_material(self):
        medium = OddElasticMedium(bulk_modulus=10.0, shear_modulus=5.0, odd_modulus=0.0)
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        eps_s = 0.1 * np.cos(theta)
        eps_a = 0.1 * np.sin(theta)
        w = medium.compute_cycle_work_numerical(eps_s, eps_a)
        assert pytest.approx(w, abs=1e-8) == 0.0

    def test_complex_eigenvalues(self):
        medium = OddElasticMedium(bulk_modulus=8.0, shear_modulus=3.0, odd_modulus=2.0)
        eigs = medium.eigenvalues()
        assert eigs[0] == 8.0
        assert eigs[1] == complex(3.0, 2.0)
        assert eigs[2] == complex(3.0, -2.0)


class TestOddLattice:
    """Tests discrete active odd spring network physics."""

    def test_linear_momentum_conservation(self):
        lattice = OddLattice(n_nodes=2, substrate_damping=0.0)
        lattice.set_position(0, 0.0, 0.0)
        lattice.set_position(1, 1.2, 0.0)  # Stretched bond (l0=1.0)
        lattice.add_bond(0, 1, l0=1.0, k=50.0, k_odd=20.0, gamma=0.0)

        forces, torque = lattice.compute_forces()
        # Linear forces must be equal and opposite
        np.testing.assert_allclose(forces[0] + forces[1], [0.0, 0.0], atol=1e-12)
        # Transverse force must be present on node 0
        assert abs(forces[0, 1]) > 0.0
        # Active internal torque must be non-zero
        assert abs(torque) > 0.0

    def test_internal_torque_generation(self):
        lattice = OddLattice(n_nodes=2, substrate_damping=0.0)
        lattice.set_position(0, 0.0, 0.0)
        lattice.set_position(1, 2.0, 0.0)  # length 2.0, l0 1.0 -> delta_l = 1.0
        lattice.add_bond(0, 1, l0=1.0, k=10.0, k_odd=5.0, gamma=0.0)

        _, torque = lattice.compute_forces()
        # Torque = -k_odd * dist * delta_l = -5.0 * 2.0 * 1.0 = -10.0
        assert pytest.approx(torque, rel=1e-4) == -10.0

    def test_lattice_step_execution(self):
        lattice = OddLattice(n_nodes=3, substrate_damping=0.5)
        lattice.set_position(0, 0.0, 0.0, fixed=True)
        lattice.set_position(1, 1.0, 0.0)
        lattice.set_position(2, 2.0, 0.0, fixed=True)
        lattice.add_bond(0, 1, l0=1.0, k=40.0, k_odd=10.0)
        lattice.add_bond(1, 2, l0=1.0, k=40.0, k_odd=10.0)

        lattice.positions[1, 1] += 0.2
        lattice.step(dt=0.01)
        assert lattice.total_kinetic_energy() > 0.0


class TestWaveConveyor:
    """Tests non-reciprocal retrograde traveling wave conveyor."""

    def test_surface_kinematics_phase_shift(self):
        conveyor = WaveConveyor(wave_speed=4.0, wave_length=4.0, amp_x=0.1, amp_y=0.2)
        ux, uy, vx, vy = conveyor.get_surface_state(0.0, 0.0)
        assert pytest.approx(ux, abs=1e-7) == 0.0
        assert pytest.approx(uy, rel=1e-5) == 0.2
        assert vx > 0.0  # Wave crest pushes forward!

    def test_cargo_unidirectional_transport(self):
        conveyor = WaveConveyor(length=10.0, wave_speed=3.0, wave_length=3.0, amp_x=0.15, amp_y=0.25)
        res = conveyor.simulate_cargo(x0=0.5, v0=0.0, duration=2.5)
        assert res['steady_state_speed'] > 0.2
        assert res['transport_efficiency'] > 0.0
        assert res['x'][-1] > res['x'][0]

    def test_conveyor_speed_scaling(self):
        c1 = WaveConveyor(wave_speed=2.0)
        c2 = WaveConveyor(wave_speed=4.0)
        res1 = c1.simulate_cargo(duration=2.0)
        res2 = c2.simulate_cargo(duration=2.0)
        assert res2['steady_state_speed'] > res1['steady_state_speed']


class TestAntiJammingHopper:
    """Tests anti-jamming granular flow with active odd shear."""

    def test_odd_hopper_discharge(self):
        hopper = OddHopper(n_particles=20, orifice_width=0.85, particle_radius=0.25, odd_coeff=0.4)
        res = hopper.simulate_discharge(max_time=2.0)
        assert res['discharged_count'] > 0
        assert res['discharge_rate'] > 0.0
        assert res['has_continuous_flow']

    def test_passive_vs_odd_jamming_contrast(self):
        hopper_odd = OddHopper(n_particles=20, orifice_width=0.80, particle_radius=0.25, odd_coeff=0.4)
        res_odd = hopper_odd.simulate_discharge(max_time=3.0)

        hopper_passive = OddHopper(n_particles=20, orifice_width=0.80, particle_radius=0.25, odd_coeff=0.0)
        res_passive = hopper_passive.simulate_discharge(max_time=3.0)

        # Active odd hopper discharges more particles because odd shear breaks static friction arches
        assert res_odd['discharged_count'] >= res_passive['discharged_count']


class TestStrainEngine:
    """Tests continuous mechanical work extraction from strain loops."""

    def test_power_and_work_positive(self):
        engine = StrainEngine(strain_amplitude=0.1, frequency_hz=25.0)
        assert engine.work_per_cycle > 0.0
        assert engine.average_power_output > 0.0

    def test_cargo_acceleration(self):
        engine = StrainEngine(strain_amplitude=0.12, frequency_hz=20.0)
        res = engine.boost_cargo(cargo_mass=1.0, entry_velocity=1.0, dwell_cycles=5.0)
        assert res['exit_velocity'] > res['entry_velocity']
        assert res['final_kinetic_energy'] > res['initial_kinetic_energy']
        assert res['work_transferred'] > 0.0
        assert res['boost_factor'] > 1.0

    def test_invalid_cargo_mass(self):
        engine = StrainEngine()
        with pytest.raises(ValueError, match="Cargo mass"):
            engine.boost_cargo(cargo_mass=-1.0)


class TestMechanicalDiode:
    """Tests non-reciprocal acoustic metamaterial diode and rectifying shock absorber."""

    def test_forward_backward_asymmetry(self):
        diode = MechanicalDiode(n_sites=12, odd_alpha=8.0, damping=1.0)
        res_fwd = diode.simulate_pulse('forward', duration=1.5)
        res_bwd = diode.simulate_pulse('backward', duration=1.5)

        assert res_fwd['transmittance'] > res_bwd['transmittance']

    def test_rectification_ratio(self):
        diode = MechanicalDiode(n_sites=12, odd_alpha=8.0, damping=1.0)
        metrics = diode.compute_rectification()
        assert metrics['rectification_ratio_db'] > 6.0
        assert metrics['is_diode_active']

    def test_diode_damping_dependence(self):
        d1 = MechanicalDiode(damping=0.5)
        d2 = MechanicalDiode(damping=2.0)
        r1 = d1.compute_rectification()
        r2 = d2.compute_rectification()
        assert r1['forward_transmittance'] > r2['forward_transmittance']


class TestOddFoundryEngine:
    """Tests integrated odd logistics factory pipeline."""

    def test_full_pipeline_audit(self):
        engine = OddFoundryEngine(odd_shear_modulus=3.0, hopper_odd_coeff=0.40, wave_speed=4.0)
        audit = engine.run_full_pipeline_audit()

        assert audit['pipeline_verified']
        assert audit['hopper']['discharged_count'] > 0
        assert audit['conveyor']['steady_state_speed'] > 0.2
        assert audit['strain_engine']['boost_factor'] > 1.0
        assert audit['diode']['rectification_ratio_db'] > 6.0
