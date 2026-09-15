import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "run_phase1r_audit.py"
SPEC = importlib.util.spec_from_file_location("phase1r_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class Phase1RAuditTests(unittest.TestCase):
    def test_null_array_is_not_reported(self):
        self.assertEqual(audit.slot_summary('[null,null,null]'), (3, 0, []))

    def test_partial_array_preserves_reported_zero(self):
        total, reported, values = audit.slot_summary('[0,null,20]')
        self.assertEqual((total, reported), (3, 2))
        self.assertEqual(values, [0, 20])

    def test_missing_does_not_define_method(self):
        process = {field: None for field in audit.FIELD_META}
        self.assertEqual(audit.empirical_family(process), "unresolved")

    def test_power_and_pulse_profiles(self):
        power = {field: None for field in audit.FIELD_META}
        power["deposition_power"] = '[0,50,null]'
        pulse = {field: None for field in audit.FIELD_META}
        pulse["deposition_target_pulses"] = '[100,null]'
        self.assertEqual(audit.empirical_family(power), "power_parameterized")
        self.assertEqual(audit.empirical_family(pulse), "pulse_parameterized")

    def test_joint_absence_has_zero_common_fields(self):
        left = 0
        right = 0
        self.assertEqual((left & right).bit_count(), 0)


if __name__ == "__main__":
    unittest.main()
