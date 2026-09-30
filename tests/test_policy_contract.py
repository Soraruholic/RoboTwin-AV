import unittest
from types import SimpleNamespace
import numpy as np
from active_view.policy import execute_chunk, validate_actions, validate_contract
from scripts.convert_av_to_lerobot import causal_indices


class PolicyTests(unittest.TestCase):
    def test_reject_missing_head_or_velocity_and_nonfinite(self):
        for actions in (np.zeros((2, 14)), np.zeros((2, 16)), np.zeros((0, 28)), np.full((2, 28), np.nan)):
            with self.assertRaises(ValueError):
                validate_actions(actions)

    def test_reject_wrong_rate(self):
        with self.assertRaises(ValueError):
            validate_contract(dict(action_dim=28, state_dim=16, control_hz=25,
                                   action_type='dense_joint_position_velocity'))

    def test_causal_frames_never_look_ahead(self):
        np.testing.assert_array_equal(causal_indices([0, 3, 10], 10), [0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            causal_indices([0, 3, 9], 10)

    def test_dense_execution_keeps_velocity_and_timing(self):
        calls = []
        av = SimpleNamespace(step=0)
        def step(action, steps, arm_velocity, observe):
            calls.append((action.copy(), arm_velocity.copy(), steps, observe))
            av.step += steps
        av.policy_step = step
        task = SimpleNamespace(active_view=av, check_success=lambda: False)
        actions = np.zeros((4, 28), dtype=np.float32)
        actions[:, 14:16] = [.1, -.2]
        actions[:, 16:] = .3
        execute_chunk(task, actions, 4, 3)
        self.assertEqual(av.step, 3)
        np.testing.assert_allclose(calls[0][0][-2:], [.1, -.2])
        np.testing.assert_allclose(calls[0][1], .3)


if __name__ == '__main__':
    unittest.main()
