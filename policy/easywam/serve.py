"""Serve an upstream EasyWAM AV checkpoint using the OpenPI client protocol."""
import argparse
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy.easywam.run import use_upstream


class EasyWAMPolicy:
    def __init__(self, args):
        use_upstream(args.easywam_root)
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from data.lerobot.utils.normalizer import load_dataset_stats_from_json
        from experiments.prompt_context_cache import PromptContextCache
        from model.helpers.inference import configure_model_execution
        from utils.logging_config import setup_logging
        setup_logging(log_level=logging.INFO)
        cfg = OmegaConf.load(args.run_dir / 'config.yaml')
        if args.backbone:
            cfg.model.backbone.model_id = str(args.backbone.resolve())
            cfg.model.backbone.tokenizer_model_id = str(args.backbone.resolve() / 'google/umt5-xxl')
        if args.base_checkpoint:
            cfg.model.base_checkpoint = str(args.base_checkpoint.resolve())
        cfg.model.backbone.load_text_encoder = True
        cfg.model.backbone.use_gradient_checkpointing = False
        cfg.model.backbone.dit_config.use_gradient_checkpointing = False
        torch.manual_seed(cfg.seed)
        self.model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
        self.model.load_checkpoint(str(args.checkpoint), merge_lora=True)
        self.model = configure_model_execution(self.model, vae_micro_batch_size=1,
                                               inference_cross_kv_reuse=True).eval()
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(args.run_dir / 'dataset_stats.json')))
        self.cache = PromptContextCache(self.model)
        self.horizon = int(cfg.data.train.num_frames) - 1
        self.video_frames = self.horizon // int(cfg.data.train.action_video_freq_ratio) + 1
        from data.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop
        video_size = cfg.data.train.video_size
        self.resize = ResizeSmallestSideAspectPreserving({'img_h': video_size[0], 'img_w': video_size[1]})
        self.crop = CenterCrop({'img_h': video_size[0], 'img_w': video_size[1]})
        self.steps = args.inference_steps

    def infer(self, obs):
        import numpy as np
        import torch
        from torchvision.transforms import functional as F
        from data.lerobot.prompts import DEFAULT_PROMPT
        from active_view.policy import CAMERAS, validate_actions
        images = []
        for name, size in zip(CAMERAS, ([256, 320], [128, 160], [128, 160])):
            image = np.asarray(obs['images'][name])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f'{name} must be HWC uint8 RGB')
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.
            # Match the two resize operations in upstream RobotVideoDataset.
            tensor = F.resize(tensor, [240, 320], antialias=True)
            images.append(F.resize(tensor, size, antialias=True))
        image = torch.cat((images[0], torch.cat(images[1:], dim=-1)), dim=-2)
        image = self.crop(self.resize(image))
        image = (image * 2 - 1).unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)
        state = np.array(obs['state'], dtype=np.float32, copy=True)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 16-D state')
        batch = self.processor.normalizer.forward({'state': {'default': torch.from_numpy(state).unsqueeze(0)}})
        context, mask = self.cache.get(DEFAULT_PROMPT.format(task=str(obs['prompt'])))
        with torch.inference_mode():
            result = self.model.infer_action(prompt=None, context=context, context_mask=mask,
                input_image=image, proprio=batch['state']['default'], action_horizon=self.horizon,
                num_video_frames=self.video_frames, num_inference_steps=self.steps, text_cfg_scale=1.)
        actions = self.processor.normalizer.normalizers['action']['default'].backward(
            result['action'].detach().float().cpu()).numpy()
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0., 1.)
        return {'actions': validate_actions(actions)}


async def serve(policy, host, port):
    from openpi_client import msgpack_numpy
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.exceptions import ConnectionClosed
    metadata = dict(policy='easywam', action_type='dense_joint_position_velocity',
                    action_dim=28, state_dim=16, control_hz=250)

    async def handler(socket):
        packer = msgpack_numpy.Packer()
        await socket.send(packer.pack(metadata))
        try:
            async for message in socket:
                result = policy.infer(msgpack_numpy.unpackb(message))
                await socket.send(packer.pack(result))
        except ConnectionClosed:
            pass
        except Exception:
            logging.exception('EasyWAM inference failed')
            await socket.close(code=1011, reason='Inference failed; see server log')

    async with websocket_serve(handler, host, port, compression=None, max_size=None):
        print(f'EasyWAM ready on {host}:{port}', flush=True)
        await asyncio.Future()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--easywam-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--backbone', type=Path)
    p.add_argument('--base-checkpoint', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--inference-steps', type=int, default=10)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8001)
    args = p.parse_args()
    if args.inference_steps < 1:
        p.error('inference-steps must be positive')
    asyncio.run(serve(EasyWAMPolicy(args), args.host, args.port))


if __name__ == '__main__':
    main()
