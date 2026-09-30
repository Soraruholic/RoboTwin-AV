"""OpenPI configuration using native dense RoboTwin-AV controls."""
import dataclasses
import os

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config, weight_loaders

from active_view.policy import CAMERAS
from policy.pi05.transforms import Inputs, Outputs


@dataclasses.dataclass(frozen=True)
class AVDataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(inputs=[transforms.RepackTransform({
            'images': {name: 'observation.images.' + name for name in CAMERAS},
            'state': 'observation.state', 'actions': 'action', 'prompt': 'prompt',
        })])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
            data_transforms=transforms.Group(inputs=[Inputs()], outputs=[Outputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=('action',), prompt_from_task=True,
        )


def register():
    name = 'pi05_robotwin_av'
    cfg = config.TrainConfig(
        name=name, model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50,
                                             discrete_state_input=True),
        data=AVDataConfig(repo_id=os.environ.get('AV_REPO_ID', 'local/robotwin_av'),
                          base_config=config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get('AV_BASE_PARAMS', 'gs://openpi-assets/checkpoints/pi05_base/params')),
        batch_size=32, num_train_steps=30000,
    )
    config._CONFIGS_DICT[name] = cfg
    return cfg
