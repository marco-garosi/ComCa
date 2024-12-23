import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler


def load_diffusion_pipeline(model_name, scheduler, compile=False):
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
    if compile:
        pipeline.unet = torch.compile(pipeline.unet, mode='reduce-overhead', fullgraph=True)
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    if scheduler == 'dpm_ms':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    return pipeline, distributed_state
