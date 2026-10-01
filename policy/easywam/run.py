"""Prepare text embeddings or fine-tune using the upstream EasyWAM trainer."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = 'a8ec082360e101d22601af4fdf02a82eab506819'


def use_upstream(path):
    path = Path(path).resolve()
    if not (path / 'src/model/easywam_unified.py').is_file():
        raise FileNotFoundError(f'Not an EasyWAM checkout: {path}')
    # Upstream's data/model/utils must precede RoboTwin's packages.
    sys.path[:0] = [str(path / 'src'), str(path), str(ROOT)]
    return path


def config(args, overrides):
    upstream = use_upstream(args.easywam_root)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from utils.config_resolvers import register_default_resolvers
    from active_view.policy import CAMERAS, validate_contract
    register_default_resolvers()
    data = args.data.resolve()
    contract = json.loads((data / 'av_contract.json').read_text())
    validate_contract(contract)
    if contract['split'] != 'train':
        raise ValueError('Fine-tuning requires the training split')
    with initialize_config_dir(version_base='1.3', config_dir=str(upstream / 'configs')):
        cfg = compose(config_name='train', overrides=['data=robotwin', 'model=easywam_unified_wan22_lora'])
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(__file__).with_name('finetune.yaml')))
    cfg.output_dir = str(args.output.resolve())
    cfg.model._target_ = 'policy.easywam.model.create_model'
    cfg.model.base_checkpoint = str(args.base_checkpoint.resolve()) if args.base_checkpoint else None
    cfg.model.backbone.model_id = str(args.backbone.resolve())
    cfg.model.backbone.tokenizer_model_id = str(args.backbone.resolve() / 'google/umt5-xxl')
    info = json.loads((data / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        raise ValueError('Use policy/easywam/convert.py to create LeRobot v3.0 data')
    for split in ('train', 'val'):
        node = cfg.data[split]
        node.dataset_dirs = [str(data)]
        node.pretrained_norm_stats = None
        node.text_embedding_cache_dir = str(data / 'text_embeds')
        node.shape_meta.images = [dict(key=name, raw_shape=[3, *info['features']['observation.images.' + name]['shape'][:2]],
                                      shape=[3, 240, 320]) for name in CAMERAS]
        node.shape_meta.action = [dict(key='default', raw_shape=28, shape=28)]
        node.shape_meta.state = [dict(key='default', raw_shape=16, shape=16)]
        node.processor.action_output_dim = 28
        node.processor.proprio_output_dim = 16
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if args.command == 'train' and int(info['total_episodes'] * (1 - cfg.data.train.val_set_proportion)) < 1:
        raise ValueError('Empty training split; use more episodes or set data.train.val_set_proportion=0')
    # Check dimensions and temporal sampling before allocating the model.
    if cfg.data.train.processor.action_output_dim != 28 or cfg.data.train.processor.proprio_output_dim != 16:
        raise ValueError('AV requires 28-D actions and 16-D state')
    horizon = cfg.data.train.num_frames - 1
    if horizon < 1 or horizon % (4 * cfg.data.train.action_video_freq_ratio):
        raise ValueError('Action horizon must span a multiple of four video transitions')
    cfg.av_tested_upstream_revision = UPSTREAM_REVISION
    cfg.output_dir = str(Path(cfg.output_dir).resolve())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['text', 'train'])
    p.add_argument('--easywam-root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args, overrides = p.parse_known_args()
    cfg = config(args, overrides)
    import torch
    import numpy as np
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if args.command == 'text':
        from scripts.precompute_text_embeds import main as precompute
        precompute.__wrapped__(cfg)
    else:
        from runtime import run_training
        run_training(cfg)


if __name__ == '__main__':
    main()
