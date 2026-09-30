"""Observation-gated gaze expert and fixed-duration policy interface."""
import numpy as np
import sapien
from .head import HeadServo


class ObservationTimeout(RuntimeError):
    pass


class ActiveView:
    def __init__(self, task, config):
        self.task, self.config = task, dict(config)
        self.mode = config["mode"]
        if self.mode not in {"fixed", "tracking", "phase", "external"}:
            raise ValueError("unknown active-view mode")
        self.cameras = {name: camera for name, camera in
                        zip(task.cameras.static_camera_name, task.cameras.static_camera_list)
                        if name == "head_camera"}
        self.cameras.update(left_camera=task.cameras.left_camera, right_camera=task.cameras.right_camera)
        self.head = self.cameras["head_camera"]
        self.dt = task.scene.get_timestep()
        self.servo = HeadServo(self.head.entity.get_pose().to_transformation_matrix(), self.dt, config)
        self.sample_steps = int(config["sample_steps"])
        if self.sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        self.step = 0
        self.recorder = None
        self.phase = "initial"
        self.behavior = "stay"
        self.objects = {}
        self.visible = {}
        self.last_seen = {}
        self.focus_names = []
        self.ee_names = []
        self.primary = None
        self.phase_start = 0
        self.last_frame_step = -1
        self.frame = None
        self.apply_pose()

    def register(self, name, actor):
        # IDs identify segmentation pixels, never expose the hidden actor pose.
        entity = getattr(actor, "actor", actor)
        if hasattr(entity, "get_links"):
            self.objects[name] = [int(link.entity.per_scene_id) for link in entity.get_links()]
        else:
            self.objects[name] = [int(entity.per_scene_id)]
        if not self.objects[name]:
            raise ValueError(f"no render entities for {name}")

    def apply_pose(self):
        self.head.entity.set_pose(sapien.Pose(self.servo.pose()))

    def start(self, recorder):
        self.recorder = recorder
        self.capture()

    def command_vector(self):
        robot = self.task.robot
        left = [float(j.get_drive_target()[0]) for j in robot.left_arm_joints]
        right = [float(j.get_drive_target()[0]) for j in robot.right_arm_joints]
        grippers = robot.get_normal_real_gripper_val()
        return np.asarray(left + [grippers[0]] + right + [grippers[1]] + self.servo.target.tolist(), np.float32)

    def before_step(self):
        # Row j is the control applied during [j*dt, (j+1)*dt).
        if self.recorder:
            self.recorder.append("commands", {
                "step": np.int64(self.step), "time_s": self.step * self.dt,
                "action": self.command_vector(), "phase": self.phase,
                "behavior": self.behavior,
                "arm_velocity_target": np.asarray([
                    float(j.get_drive_velocity_target()[0]) for j in
                    self.task.robot.left_arm_joints + self.task.robot.right_arm_joints], np.float32),
                "head_effective_target": self.servo.effective_target.copy(),
            })
        self.servo.advance()
        self.apply_pose()

    def after_step(self):
        self.step += 1
        if self.step % self.sample_steps == 0:
            self.capture()

    def capture(self):
        if self.last_frame_step == self.step:
            return self.frame
        task = self.task
        task._update_render()
        camera_values = {}
        self.visible = {name: {} for name in self.objects}
        for name, camera in self.cameras.items():
            camera.take_picture()
            rgb = (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
            ids = camera.get_picture("Segmentation")[..., 1].astype(np.uint32)
            position = camera.get_picture("Position")
            world = camera.get_model_matrix()
            for object_name, entity_ids in self.objects.items():
                mask = np.isin(ids, entity_ids) & (position[..., 3] < 1)
                count = int(mask.sum())
                if count >= self.config["min_visible_pixels"]:
                    # Surface centroid from current visible depth pixels. No hidden pose.
                    center = np.median(position[mask, :3], axis=0)
                    point = (world @ np.r_[center, 1])[:3]
                    self.visible[object_name][name] = {"point": point, "pixels": count}
                    self.last_seen[object_name] = (self.step, point.copy())
            camera_values[f"rgb/{name}"] = rgb
            camera_values[f"intrinsics/{name}"] = camera.get_intrinsic_matrix()
            camera_values[f"extrinsics_cv/{name}"] = camera.get_extrinsic_matrix()
            if self.config.get("save_segmentation", False):
                camera_values[f"actor_id/{name}"] = ids
        robot = task.robot
        left_state = list(robot.get_left_arm_jointState())
        right_state = list(robot.get_right_arm_jointState())
        # Upstream gripper state is command-valued: save physical finger qpos separately.
        finger_qpos = []
        for entity, joints in ((robot.left_entity, robot.left_gripper), (robot.right_entity, robot.right_gripper)):
            qpos = entity.get_qpos()
            active = entity.get_active_joints()
            indices = {joint.name: i for i, joint in enumerate(active)}
            finger_qpos.extend(float(qpos[indices[j[0].name]]) for j in joints)
        flags = np.array([[int(camera in self.visible[obj]) for camera in self.cameras]
                          for obj in self.objects], dtype=np.uint8)
        values = dict(camera_values, step=np.int64(self.step), time_s=self.step * self.dt,
                      command_offset=np.int64(self.step),
                      robot_state=np.asarray(left_state + right_state, np.float32),
                      finger_qpos=np.asarray(finger_qpos, np.float32),
                      head_qpos=self.servo.q.copy(), head_velocity=self.servo.velocity.copy(),
                      head_target=self.servo.target.copy(), head_settled=np.uint8(self.servo.settled),
                      phase=self.phase, behavior=self.behavior, visible=flags)
        if self.recorder:
            self.recorder.append("frames", values)
        self.frame, self.last_frame_step = values, self.step
        self.update_gaze()
        return values

    def update_gaze(self):
        if self.mode in {"fixed", "external"}:
            self.behavior = "stay" if self.mode == "fixed" else "policy"
            return
        names = [self.primary] if self.mode == "tracking" and self.primary else self.focus_names
        points = []
        for name in names:
            if self.visible.get(name):
                views = self.visible[name]
                selected = views.get("head_camera", next(iter(views.values())))
                points.append(selected["point"])
            elif name in self.last_seen and (self.step - self.last_seen[name][0]) * self.dt < 0.4:
                points.append(self.last_seen[name][1])
        if self.mode == "phase" and any(name not in self.last_seen for name in names):
            # A visible known object must not prevent scanning for an unseen goal.
            points = []
        if points:
            if self.mode == "phase":
                for side in self.ee_names:
                    points.append(np.asarray(self.task.get_arm_pose(side)[:3]))
            aim = self.servo.aim(np.mean(points, axis=0))
            if np.max(np.abs(aim - self.servo.q)) > self.config["deadband_rad"]:
                self.servo.command(aim)
                self.behavior = "track" if self.mode == "tracking" else "frame"
            else:
                self.behavior = "stay"
        elif names:
            # Deterministic route independent of hidden object identity/position.
            waypoints = self.config["scan_rad"]
            index = int((self.step - self.phase_start) * self.dt / self.config["scan_dwell_s"]) % len(waypoints)
            self.servo.command(waypoints[index])
            self.behavior = "search"
        else:
            self.behavior = "stay"

    def focus(self, phase, names, primary=None, ee=(), wait=True):
        self.phase = phase
        self.focus_names = list(names)
        self.primary = primary or (names[0] if names else None)
        self.ee_names = list(ee)
        self.phase_start = self.step
        self.capture()
        self.update_gaze()
        if not wait or self.mode == "external":
            return
        limit = int(self.config["observation_timeout_s"] / self.dt)
        for _ in range(limit):
            known = all(name in self.last_seen for name in names)
            stable = self.servo.settled or all(
                any(key != "head_camera" for key in self.visible.get(name, {})) for name in names)
            if known and stable:
                return
            self.task._step_physics()
        raise ObservationTimeout(f"{phase}: could not obtain stable visible evidence for {names}")

    def policy_step(self, action, steps=None, arm_velocity=None, observe=True):
        if self.mode != "external":
            raise RuntimeError("policy_step requires mode=external, so teacher cannot override policy")
        action = np.asarray(action, dtype=float)
        robot = self.task.robot
        nleft, nright = len(robot.left_arm_joints), len(robot.right_arm_joints)
        if action.shape != (nleft + nright + 4,) or not np.isfinite(action).all():
            raise ValueError("expected finite native qpos+gripper+pan/tilt action")
        steps = self.sample_steps if steps is None else int(steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        arm_velocity = np.zeros(nleft + nright) if arm_velocity is None else np.asarray(arm_velocity, dtype=float)
        if arm_velocity.shape != (nleft + nright,) or not np.isfinite(arm_velocity).all():
            raise ValueError("invalid arm velocity target")
        self.servo.command(action[-2:])
        for _ in range(steps):
            robot.set_arm_joints(action[:nleft], arm_velocity[:nleft], "left")
            robot.set_gripper(action[nleft], "left", 0)
            robot.set_arm_joints(action[nleft+1:nleft+nright+1], arm_velocity[nleft:], "right")
            robot.set_gripper(action[nleft+nright+1], "right", 0)
            self.task._step_physics()
        if not observe:
            return None
        observation = self.task.get_obs()
        # Whitelist camera fields too: enabling upstream diagnostic segmentation
        # or depth must never change the deployed model's information boundary.
        camera_fields = ("rgb", "intrinsic_cv", "extrinsic_cv", "cam2world_gl")
        return {"observation": {name: {key: observation["observation"][name][key]
                                       for key in camera_fields} for name in self.cameras},
                "robot_state": np.asarray(robot.get_left_arm_jointState() + robot.get_right_arm_jointState()),
                "head_qpos": self.servo.q.copy(), "head_velocity": self.servo.velocity.copy(),
                "time_s": self.step * self.dt}

    def metadata(self):
        return {"config": self.config, "physics_dt_s": self.dt,
                "camera_order": list(self.cameras), "object_order": list(self.objects),
                "object_entity_ids": self.objects,
                "head_nominal_pose": self.servo.nominal.tolist(),
                "action_semantics": "dense applied left joint targets, left gripper [0,1], right joint targets, right gripper [0,1], pan/tilt target radians",
                "velocity_note": "arm_velocity_target records planner feedforward; zero-velocity policy execution is a different controller and needs separate validation",
                "state_note": "robot_state contains measured arm qpos but command-valued grippers; physical finger_qpos is separate",
                "visibility_note": "instance surface pixels threshold, not a semantic readability or occlusion-ratio label",
                "teacher_note": "privileged visible segmentation/depth and scripted phase; no hidden actor pose for gaze; manipulation remains privileged expert"}
