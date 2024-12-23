import torch
from tqdm import tqdm


def generate_images_batch(
        pipeline, distributed_state,
        seeds,
        metadata_store,
        num_inference_steps=50,
        batch_size=1
):
    images_store = {}
    prompts = [m['prompt'] for m in metadata_store.values()]
    negative_prompts = [m['negatives_prompt'] for m in metadata_store.values()]

    ACCELERATORS = distributed_state.num_processes
    BS = batch_size * ACCELERATORS
    ids = torch.arange(len(metadata_store))
    for idx_start in tqdm(range(0, len(prompts), BS), 'Generation'):
        idx_end = min(idx_start + BS, len(prompts))

        inputs = {
            'prompts': prompts[idx_start:idx_end],
            'ids': ids[idx_start:idx_end],
            'seeds': seeds[idx_start:idx_end],
            'negative_prompts': negative_prompts[idx_start:idx_end],
        }

        # Split work across all GPUs
        with distributed_state.split_between_processes(inputs) as input:
            # Get the input
            prompt = input['prompts']
            ids_for_device = input['ids']
            seeds_for_device = input['seeds']

            # Set the seed for the generator
            generator = [torch.Generator(distributed_state.device).manual_seed(seed.item()) for seed in seeds_for_device]
            
            # Generate and store
            with torch.no_grad():
                images = pipeline(prompt=prompt, generator=generator, num_inference_steps=num_inference_steps).images
            for idx, attr_idx in enumerate(ids_for_device):
                # images_store[attr_idx.item()] = images[idx]
                images_store[idx] = images[idx]
    
    return images_store
