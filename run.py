import pprint
from typing import List, Optional, Dict, Tuple, Callable
import pyrallis
import torch
from PIL import Image

from config import RunConfig
from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline as AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    pprint.pprint(token_indices)
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def get_indices_to_alter_new(stable, prompt: str, tokens_str: str) -> Tuple[List[int], Dict[int, str]]:
    """
    Parses the prompt to get a map of token indices to words,
    and identifies the indices to alter based on user input.
    Returns the list of indices to alter and the full token map.
    """
    token_ids = stable.tokenizer(prompt)['input_ids']
    token_idx_to_word = {
        idx: stable.tokenizer.decode(t)
        for idx, t in enumerate(token_ids)
        if 0 < idx < len(token_ids) - 1
    }
    pprint.pprint(token_idx_to_word)

    # The web UI passes a number, which we convert to a string.
    if isinstance(tokens_str, (int, float)):
        tokens_str = str(int(tokens_str))

    try:
        # Handle comma-separated strings for indices
        token_indices = [int(i.strip()) for i in tokens_str.split(",") if i.strip()]
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse token indices '{tokens_str}'. Error: {e}. Defaulting to empty list.")
        token_indices = []

    print(f"Altering tokens: {[token_idx_to_word.get(i, '<UNK>') for i in token_indices]}")
    return token_indices, token_idx_to_word


def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    token_indices = get_indices_to_alter(stable,
                                         config.prompt) if config.token_indices is None else config.token_indices

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')


# TODO add image guidance
def run_on_prompt_and_image(prompt: List[str],
                            model: AttendAndExcitePipeline,
                            controller: AttentionStore,
                            token_indices: List[int],
                            init_image: Image.Image,
                            init_image_guidance_scale: float,
                            seed: torch.Generator,
                            config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    init_image=init_image,
                    init_image_guidance_scale=init_image_guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image


def run_on_prompt_and_masked_image(prompt: List[str],
                                   model: AttendAndExcitePipeline,
                                   controller: AttentionStore,
                                   token_indices: List[int],
                                   init_image: Image.Image,
                                   init_image_guidance_scale: float,
                                   mask_image: str,
                                   seed: torch.Generator,
                                   config: RunConfig,
                                   normal_prompt,
                                   detailed_prompt,
                                   img_prompt=None,
                                   abnormal_img=None,
                                   clip_loss=None,
                                   callback: Optional[Callable] = None,
                                   callback_steps: Optional[int] = 1) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs, image_latents = model(prompt=prompt,
                                   attention_store=controller,
                                   indices_to_alter=token_indices,
                                   attention_res=config.attention_res,
                                   guidance_scale=config.guidance_scale,
                                   init_image=init_image,
                                   init_image_guidance_scale=init_image_guidance_scale,
                                   mask_image=mask_image,
                                   generator=seed,
                                   num_inference_steps=config.n_inference_steps,
                                   max_iter_to_alter=config.max_iter_to_alter,
                                   run_standard_sd=config.run_standard_sd,
                                   thresholds=config.thresholds,
                                   scale_factor=config.scale_factor,
                                   scale_range=config.scale_range,
                                   smooth_attentions=config.smooth_attentions,
                                   sigma=config.sigma,
                                   kernel_size=config.kernel_size,
                                   sd_2_1=config.sd_2_1,
                                   img_prompt=img_prompt,
                                   normal_prompt=normal_prompt,
                                   abnormal_img=abnormal_img,
                                   detailed_prompt=detailed_prompt,
                                   clip_loss=clip_loss,
                                   callback=callback,
                                   callback_steps=callback_steps)
    # image = outputs.images[0]
    # return image, image_latents
    return outputs[0], image_latents


if __name__ == '__main__':
    main()
