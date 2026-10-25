"""Unified Odd Logistics Factory Engine: coordinating anti-jamming, wave conveyance, strain work, and diode isolation."""

from odd.constitutive import OddElasticMedium
from odd.wave_conveyor import WaveConveyor
from odd.anti_jamming import OddHopper
from odd.strain_engine import StrainEngine
from odd.diode import MechanicalDiode


class OddFoundryEngine:
    """Full-pipeline automated logistics system powered by active odd elasticity.

    Pipeline:
      [Odd Hopper] -> [Wave Conveyor] -> [Strain Engine Core] -> [Mechanical Diode] -> [Discharge]
    """

    def __init__(
        self,
        odd_shear_modulus: float = 3.0,
        hopper_odd_coeff: float = 0.40,
        wave_speed: float = 4.0,
        engine_freq: float = 20.0,
        diode_alpha: float = 8.0,
    ):
        self.medium = OddElasticMedium(bulk_modulus=10.0, shear_modulus=5.0, odd_modulus=odd_shear_modulus)
        self.hopper = OddHopper(n_particles=20, odd_coeff=hopper_odd_coeff)
        self.conveyor = WaveConveyor(length=20.0, wave_speed=wave_speed)
        self.engine = StrainEngine(medium=self.medium, frequency_hz=engine_freq)
        self.diode = MechanicalDiode(n_sites=12, odd_alpha=diode_alpha)

    def run_full_pipeline_audit(self) -> dict:
        """Runs a comprehensive verification across all 4 operational units."""
        # 1. Hopper discharge test
        hopper_res = self.hopper.simulate_discharge(max_time=2.0)

        # 2. Conveyor transport test
        conveyor_res = self.conveyor.simulate_cargo(duration=3.0)

        # 3. Engine work boost test
        cargo_entry_v = max(0.5, conveyor_res['steady_state_speed'])
        engine_res = self.engine.boost_cargo(cargo_mass=1.5, entry_velocity=cargo_entry_v, dwell_cycles=4.0)

        # 4. Diode acoustic isolation test
        diode_res = self.diode.compute_rectification()

        all_passed = bool(
            hopper_res['discharged_count'] > 0 and
            conveyor_res['steady_state_speed'] > 0.2 and
            engine_res['exit_velocity'] > engine_res['entry_velocity'] and
            diode_res['rectification_ratio_db'] > 6.0
        )

        return {
            'hopper': hopper_res,
            'conveyor': conveyor_res,
            'strain_engine': engine_res,
            'diode': diode_res,
            'pipeline_verified': all_passed,
        }
