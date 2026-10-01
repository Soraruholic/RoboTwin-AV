"""Initialize the upstream model with AV-sized state/action projections."""
def create_model(base_checkpoint=None, lora=None, **kwargs):
    import torch
    from runtime import create_easywam_unified
    from model.component.lora import inject_video_dit_lora

    if base_checkpoint:
        kwargs['skip_dit_load_from_pretrain'] = True
    model = create_easywam_unified(lora=None, **kwargs)
    if base_checkpoint:
        payload = torch.load(base_checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if payload.get('backbone_name', 'wan22') != 'wan22':
            raise ValueError('Expected a Wan2.2 EasyWAM checkpoint')
        # Transfer the trained video backbone; AV uses a different action space.
        video = {key.removeprefix('video_dit.'): value for key, value in payload['dit'].items()
                 if key.startswith('video_dit.')}
        model.video_dit.load_state_dict(video, strict=True)
        print(f'Loaded upstream EasyWAM video backbone: {base_checkpoint}', flush=True)
    if lora is not None:
        inject_video_dit_lora(model.video_dit, lora)
    return model
