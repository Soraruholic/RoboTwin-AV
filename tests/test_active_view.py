from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

from active_view.head import HeadServo
from active_view.recorder import Recorder
from active_view.dataset import load_window


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "active_view/config.yml").read_text())


class ServoTests(unittest.TestCase):
    def test_limits_and_acceleration_under_reversals(self):
        cfg = config()
        servo = HeadServo(np.eye(4), .004, cfg)
        qs, vs = [], []
        for i in range(4000):
            if i % 700 == 0:
                servo.command([(-1)**(i//700) * 20, (-1)**(i//700) * 10])
            servo.advance()
            qs.append(servo.q.copy())
            vs.append(servo.velocity.copy())
        limits = np.asarray(cfg["limits_rad"])
        self.assertTrue(np.all(np.asarray(qs) >= limits[:, 0]))
        self.assertTrue(np.all(np.asarray(qs) <= limits[:, 1]))
        self.assertLessEqual(np.abs(vs).max(), cfg["max_speed_rad_s"] + 1e-7)
        self.assertLessEqual(np.abs(np.diff(vs, axis=0)).max(), cfg["max_accel_rad_s2"]*.004 + 1e-7)

    def test_latency_and_settling(self):
        cfg = config()
        servo = HeadServo(np.eye(4), .004, cfg)
        servo.command([.3, -.2])
        for _ in range(20):
            servo.advance()
        np.testing.assert_array_equal(servo.q, [0, 0])
        for _ in range(1800):
            servo.advance()
        self.assertTrue(servo.settled)
        np.testing.assert_allclose(servo.q, [.3, -.2], atol=1e-4)
        servo.command([.5, .1])
        self.assertFalse(servo.settled)

    def test_fixed_center_no_virtual_translation(self):
        nominal = np.eye(4)
        nominal[:3, 3] = [1, 2, 3]
        servo = HeadServo(nominal, .004, config())
        servo.command([.3, -.4])
        for _ in range(1000):
            servo.advance()
        np.testing.assert_allclose(servo.pose()[:3, 3], [1, 2, 3])
        rot = servo.pose()[:3, :3]
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)

    def test_visible_point_projects_to_optical_axis(self):
        nominal = np.eye(4)
        from active_view.head import rotation_z, rotation_y
        nominal[:3, :3] = rotation_z(1.57) @ rotation_y(.4)
        servo = HeadServo(nominal, .004, config())
        point = nominal[:3, :3] @ np.array([1, .2, -.1])
        servo.q = servo.aim(point)
        local = servo.pose()[:3, :3].T @ point
        np.testing.assert_allclose(local[1:], [0, 0], atol=1e-10)

    def test_invalid_command(self):
        servo = HeadServo(np.eye(4), .004, config())
        for value in [[np.nan, 0], [0], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                servo.command(value)

    def test_invalid_initial_configuration(self):
        for key, value in [('initial_rad', [float('nan'), 0]), ('limits_rad', [[-1, float('nan')], [-1, 1]]),
                           ('settling_s', -1), ('max_speed_rad_s', float('inf'))]:
            cfg = config()
            cfg[key] = value
            with self.assertRaises(ValueError):
                HeadServo(np.eye(4), .004, cfg)
        with self.assertRaises(ValueError):
            HeadServo(np.eye(4), 0, config())


class WindowTests(unittest.TestCase):
    def test_exact_commands_and_boundary_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            writer = Recorder(path, {"camera_order": ["head_camera"]})
            for index in range(20):
                writer.append("commands", {"action": np.full(16, index), "time_s": index*.004,
                                           "arm_velocity_target": np.zeros(12)})
            for step in [0, 10, 20]:
                writer.append("frames", {"command_offset": step, "step": step, "time_s": step*.004,
                                          "robot_state": np.zeros(14), "head_qpos": np.zeros(2),
                                          "rgb/head_camera": np.zeros((4, 4, 3), np.uint8)})
            writer.finish(True, "success", {})
            window = load_window(path, 1, 1)
            np.testing.assert_array_equal(window["action"][:, 0], np.arange(10, 20))
            with self.assertRaises(IndexError):
                load_window(path, 2, 1)
            with self.assertRaises(FileExistsError):
                Recorder(path, {})


class TeacherTests(unittest.TestCase):
    def make_teacher(self):
        from active_view.runtime import ActiveView
        teacher = object.__new__(ActiveView)
        teacher.mode = "phase"
        teacher.config = config()
        teacher.servo = HeadServo(np.eye(4), .004, teacher.config)
        teacher.focus_names = ["target"]
        teacher.primary = "target"
        teacher.ee_names = []
        teacher.visible = {"target": {}}
        teacher.last_seen = {}
        teacher.step = 400
        teacher.dt = .004
        teacher.phase_start = 0
        # Any attempted hidden-state query raises: gaze has only visible observations.
        teacher.task = None
        return teacher

    def test_unseen_target_scans_without_hidden_state(self):
        teacher = self.make_teacher()
        teacher.update_gaze()
        self.assertEqual(teacher.behavior, "search")
        np.testing.assert_allclose(teacher.servo.target, [-.45, 0])

    def test_unseen_goal_is_not_starved_by_visible_object(self):
        teacher = self.make_teacher()
        teacher.focus_names = ["target", "goal"]
        teacher.visible["target"] = {"head_camera": {"point": np.array([1, 0, 0]), "pixels": 100}}
        teacher.last_seen["target"] = (400, np.array([1, 0, 0]))
        teacher.update_gaze()
        self.assertEqual(teacher.behavior, "search")

    def test_visible_target_can_control_gaze(self):
        teacher = self.make_teacher()
        teacher.visible["target"] = {"head_camera": {"point": np.array([1, .3, 0]), "pixels": 100}}
        teacher.last_seen["target"] = (400, np.array([1, .3, 0]))
        teacher.update_gaze()
        self.assertEqual(teacher.behavior, "frame")
        self.assertGreater(teacher.servo.target[0], .2)

    def test_external_mode_cannot_be_overridden_by_teacher(self):
        teacher = self.make_teacher()
        teacher.mode = "external"
        teacher.servo.command([.2, .1])
        teacher.update_gaze()
        np.testing.assert_allclose(teacher.servo.target, [.2, .1])


class TaskRecipeTests(unittest.TestCase):
    def test_canonical_fifty_task_coverage(self):
        from active_view.task_recipes import TASK_ROLES
        root = Path(__file__).resolve().parents[1]
        canonical = yaml.safe_load((root / "env_cfg/eval/all_tasks.yml").read_text())["tasks"]
        self.assertEqual(len(canonical), 50)
        self.assertEqual(set(canonical), set(TASK_ROLES))

    def test_list_actors_expand_without_querying_hidden_poses(self):
        from active_view.task_recipes import resolve_actors
        actors = [object(), object()]
        subjects, goals = resolve_actors(SimpleNamespace(bread=actors, breadbasket=object()), "place_bread_basket")
        self.assertEqual(list(subjects), ["bread[0]", "bread[1]"])
        self.assertIs(subjects["bread[1]"], actors[1])
        self.assertEqual(list(goals), ["breadbasket"])

    def test_articulated_segmentation_registers_all_links(self):
        from active_view.runtime import ActiveView
        observer = object.__new__(ActiveView)
        observer.objects = {}
        articulation = SimpleNamespace(get_links=lambda: [SimpleNamespace(entity=SimpleNamespace(per_scene_id=i))
                                                         for i in [42, 77]])
        observer.register("door", SimpleNamespace(actor=articulation))
        self.assertEqual(observer.objects["door"], [42, 77])

    def test_visual_goal_can_be_a_native_entity(self):
        from active_view.runtime import ActiveView
        observer = object.__new__(ActiveView)
        observer.objects = {}
        observer.register("stamp_target", SimpleNamespace(per_scene_id=99))
        self.assertEqual(observer.objects["stamp_target"], [99])

    def test_simultaneous_grasps_are_combined_at_execution(self):
        from active_view.task_recipes import ExecutionGazeRecipe
        calls = []
        av = SimpleNamespace(mode="phase", last_seen={}, focus=lambda *args, **kwargs: calls.append((args, kwargs)))
        task = SimpleNamespace(active_view=av)
        recipe = ExecutionGazeRecipe(task, "pick_dual_bottles", {"bottle1": object(), "bottle2": object()}, {})
        recipe.before_move([("left", [SimpleNamespace(action="move", av_semantic=("bottle1", "grasp"))]),
                            ("right", [SimpleNamespace(action="move", av_semantic=("bottle2", "grasp"))])])
        self.assertEqual(calls[0][0][1], ["bottle1", "bottle2"])
        self.assertEqual(calls[0][1]["ee"], ["left", "right"])
        self.assertFalse(calls[0][1]["wait"])


if __name__ == "__main__":
    unittest.main()
