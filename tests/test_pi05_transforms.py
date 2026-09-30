import ast
from pathlib import Path
import unittest

import numpy as np

from active_view.policy import CAMERAS
from policy.pi05.transforms import Inputs, Outputs


class TransformTests(unittest.TestCase):
    def test_config_enables_pi05_state_conditioning(self):
        source = Path(__file__).resolve().parents[1] / 'policy/pi05/config.py'
        calls = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'Pi0Config']
        self.assertEqual(len(calls), 1)
        options = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
        self.assertTrue(options['pi05'])
        self.assertTrue(options['discrete_state_input'])
        self.assertEqual(options['action_dim'], 32)

    def test_preserves_state_actions_and_camera_identity(self):
        data = dict(state=np.arange(16, dtype=np.float32),
                    actions=np.arange(56, dtype=np.float32).reshape(2, 28), prompt='handover block',
                    images={name: np.full((8, 12, 3), i, np.uint8) for i, name in enumerate(CAMERAS)})
        result = Inputs()(data)
        np.testing.assert_array_equal(result['state'], data['state'])
        np.testing.assert_array_equal(result['actions'], data['actions'])
        self.assertEqual(result['prompt'], data['prompt'])
        for i, name in enumerate(('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')):
            self.assertTrue(result['image_mask'][name])
            self.assertTrue(np.all(result['image'][name] == i))

    def test_chw_float_images(self):
        result = Inputs()(dict(state=np.zeros(16),
                               images={name: np.ones((3, 8, 12), np.float32) for name in CAMERAS}))
        for image in result['image'].values():
            self.assertEqual(image.shape, (8, 12, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue(np.all(image == 255))

    def test_only_output_padding_is_removed(self):
        actions = np.arange(96, dtype=np.float32).reshape(3, 32)
        np.testing.assert_array_equal(Outputs()(dict(actions=actions))['actions'], actions[:, :28])
        with self.assertRaises(ValueError):
            Outputs()(dict(actions=np.zeros((3, 14))))

    def test_invalid_state_or_image_rejected(self):
        images = {name: np.zeros((8, 12, 3), np.uint8) for name in CAMERAS}
        for state in (np.zeros(14), np.full(16, np.nan)):
            with self.assertRaises(ValueError):
                Inputs()(dict(state=state, images=images))
        images['head_camera'] = np.zeros((8, 12))
        with self.assertRaises(ValueError):
            Inputs()(dict(state=np.zeros(16), images=images))


if __name__ == '__main__':
    unittest.main()
