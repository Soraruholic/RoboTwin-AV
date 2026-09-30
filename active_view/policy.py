"""Observation and dense-action contract shared by policy adapters."""
import numpy as np

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
ACTION_DIM = 28
STATE_DIM = 16
CONTROL_HZ = 250


def observation(frame, prompt):
    state = np.concatenate((frame['robot_state'], frame['head_qpos'])).astype(np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError('Expected a finite 16-dimensional observation state')
    return {'state': state, 'images': {name: frame['rgb/' + name] for name in CAMERAS},
            'prompt': prompt}


def validate_actions(actions):
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or not len(value) or not np.isfinite(value).all():
        raise ValueError('Policy must return a finite [horizon, 28] array')
    if np.any(value[:, [6, 13]] < 0) or np.any(value[:, [6, 13]] > 1):
        raise ValueError('Gripper commands must be in [0, 1]')
    return value


def validate_contract(metadata):
    expected = dict(action_dim=ACTION_DIM, state_dim=STATE_DIM, control_hz=CONTROL_HZ,
                    action_type='dense_joint_position_velocity')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Policy contract mismatch: {key} must be {value!r}')


def execute_chunk(task, actions, count, max_steps):
    av = task.active_view
    actions = validate_actions(actions)
    if count < 1 or count > len(actions):
        raise ValueError('execute_steps must fit the returned horizon')
    for action in actions[:count]:
        av.policy_step(action[:16], steps=1, arm_velocity=action[16:], observe=False)
        if task.check_success():
            return True
        if av.step >= max_steps:
            break
    return False
