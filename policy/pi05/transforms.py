"""Map three-camera observations to OpenPI without changing joint conventions."""
import dataclasses

import numpy as np

from active_view.policy import ACTION_DIM, CAMERAS, STATE_DIM


@dataclasses.dataclass(frozen=True)
class Inputs:
    def __call__(self, data):
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError('Expected a finite 16-dimensional state')
        images = {}
        for name, target in zip(CAMERAS, ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')):
            image = np.asarray(data['images'][name])
            if image.ndim != 3:
                raise ValueError(f'{name}: expected a three-dimensional RGB image')
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.transpose(1, 2, 0)
            if image.shape[-1] != 3 or not np.isfinite(image).all():
                raise ValueError(f'{name}: invalid RGB image')
            if np.issubdtype(image.dtype, np.floating):
                image = (image * 255).clip(0, 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError(f'{name}: expected uint8 or floating RGB image')
            images[target] = image
        result = dict(state=state, image=images,
                      image_mask={key: np.True_ for key in images})
        for key in ('actions', 'prompt'):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class Outputs:
    def __call__(self, data):
        actions = np.asarray(data['actions'])
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError('Expected at least 28 output action channels')
        return {'actions': actions[..., :ACTION_DIM]}
