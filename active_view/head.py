"""Bounded two-axis camera servo. Angles are radians; time is seconds."""
from collections import deque
import numpy as np


def rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def rotation_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])


class HeadServo:
    """Kinematic mount, not a dynamic neck or a calibrated real motor model.

    pan rotates about world-up through pivot; positive tilt looks down about
    the panned camera's left axis. Nominal orientation is the upstream mount.
    The pivot-to-optical-center offset rotates with the mount; it is not an
    independently controlled translation. Defaults to zero offset.
    """

    def __init__(self, nominal_pose, dt, config):
        self.nominal = np.asarray(nominal_pose, dtype=float)
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.limits = np.asarray(config["limits_rad"], dtype=float)
        self.max_speed = float(config["max_speed_rad_s"])
        self.max_accel = float(config["max_accel_rad_s2"])
        self.delay_steps = int(np.ceil(config["latency_s"] / dt))
        self.offset = np.asarray(config.get("pivot_offset_world_m", [0, 0, 0]), dtype=float)
        self.q = np.asarray(config.get("initial_rad", [0, 0]), dtype=float).copy()
        if (self.nominal.shape != (4, 4) or not np.isfinite(self.nominal).all()
                or self.offset.shape != (3,) or not np.isfinite(self.offset).all()
                or self.q.shape != (2,) or not np.isfinite(self.q).all()):
            raise ValueError("invalid camera pose or initial head state")
        if (self.limits.shape != (2, 2) or not np.isfinite(self.limits).all()
                or np.any(self.limits[:, 0] >= self.limits[:, 1])):
            raise ValueError("invalid pan/tilt limits")
        if (not np.isfinite([self.max_speed, self.max_accel, config['settling_s']]).all()
                or self.max_speed <= 0 or self.max_accel <= 0 or self.delay_steps < 0
                or config['latency_s'] < 0 or config['settling_s'] < 0):
            raise ValueError("invalid servo timing")
        if np.any(self.q < self.limits[:, 0]) or np.any(self.q > self.limits[:, 1]):
            raise ValueError("initial head pose outside limits")
        self.velocity = np.zeros(2)
        self.target = self.q.copy()
        self.effective_target = self.q.copy()
        self.queue = deque()
        self.tick = 0
        self.stable_ticks = 0
        self.settle_ticks = int(np.ceil(config["settling_s"] / dt))

    def command(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("head target must be finite pan/tilt[2]")
        target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        if not np.allclose(target, self.target, atol=1e-8, rtol=0):
            self.target = target.copy()
            self.stable_ticks = 0
            self.queue.append((self.tick + self.delay_steps, target.copy()))

    def advance(self):
        while self.queue and self.queue[0][0] <= self.tick:
            _, self.effective_target = self.queue.popleft()
        error = self.effective_target - self.q
        # Damped servo with braking-distance speed bound; acceleration is bounded.
        speed = np.minimum(self.max_speed, np.sqrt(2 * self.max_accel * np.abs(error)))
        desired = np.sign(error) * np.minimum(5 * np.abs(error), speed)
        dv = np.clip(desired - self.velocity, -self.max_accel * self.dt, self.max_accel * self.dt)
        self.velocity += dv
        proposed = self.q + self.velocity * self.dt
        self.q = np.clip(proposed, self.limits[:, 0], self.limits[:, 1])
        self.velocity[proposed != self.q] = 0
        self.tick += 1
        stable = np.max(np.abs(self.target - self.q)) < 0.02 and np.max(np.abs(self.velocity)) < 0.04
        self.stable_ticks = self.stable_ticks + 1 if stable else 0

    @property
    def settled(self):
        return self.stable_ticks >= self.settle_ticks

    def pose(self):
        pan, tilt = self.q
        rz = rotation_z(pan)
        rot = rz @ self.nominal[:3, :3] @ rotation_y(tilt)
        delta = rot @ self.nominal[:3, :3].T
        out = np.eye(4)
        out[:3, :3] = rot
        out[:3, 3] = self.nominal[:3, 3] - self.offset + delta @ self.offset
        return out

    def aim(self, point):
        # Convert a visible point to pan/tilt using the calibrated mount.
        vector = np.asarray(point) - self.pose()[:3, 3]
        forward = self.nominal[:3, 0]
        pan = np.arctan2(vector[1], vector[0]) - np.arctan2(forward[1], forward[0])
        pan = (pan + np.pi) % (2 * np.pi) - np.pi
        local = self.nominal[:3, :3].T @ rotation_z(-pan) @ vector
        tilt = np.arctan2(-local[2], local[0])
        return np.clip([pan, tilt], self.limits[:, 0], self.limits[:, 1])
