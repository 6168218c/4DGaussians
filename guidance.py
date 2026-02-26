from multiprocessing.dummy import Process
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Literal
import inspect
import numpy as np
from dataclasses import dataclass
from types import MethodType

import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm, trange
from contextlib import nullcontext
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
# import threading
import torch.multiprocessing as mp
import torchvision.transforms.functional as vF
import os

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    BaseOutput
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers import FluxPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
    
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
    
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@dataclass
class FlowMatchTransportOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """
    prev_transport_sample: torch.FloatTensor
    prev_sample: torch.FloatTensor
    
def set_step_index(self, index: int):
        self._step_index = index

def transport_step(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    source_model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    transport_sample: torch.FloatTensor,
    sample: torch.FloatTensor,
    source_sample: torch.FloatTensor,
    lambda_drift: float = 1.0,
    generator: Optional[torch.Generator] = None,
    per_token_timesteps: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[FlowMatchTransportOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        s_churn (`float`):
        s_tmin  (`float`):
        s_tmax  (`float`):
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        per_token_timesteps (`torch.Tensor`, *optional*):
            The timesteps for each token in the sample.
        return_dict (`bool`):
            Whether or not to return a
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

    Returns:
        [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`,
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
            otherwise a tuple is returned where the first element is the sample tensor.
    """

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    if per_token_timesteps is not None:
        per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

        sigmas = self.sigmas[:, None, None]
        lower_mask = sigmas < per_token_sigmas[None] - 1e-6
        lower_sigmas = lower_mask * sigmas
        lower_sigmas, _ = lower_sigmas.max(dim=0)

        current_sigma = per_token_sigmas[..., None]
        next_sigma = lower_sigmas[..., None]
        dt = next_sigma - current_sigma
    else:
        sigma_idx = self.step_index
        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]

        current_sigma = sigma
        next_sigma = sigma_next
        dt = next_sigma - current_sigma

    prev_sample = sample + dt * model_output
    prev_transport_sample = (transport_sample + dt * (model_output - source_model_output)
       + 0.5 * lambda_drift * (current_sigma ** 3) * dt * (sample + (1 - current_sigma) * model_output - source_sample - (1 - current_sigma) * source_model_output) 
    )

    # upon completion increase step index by one
    self._step_index += 1
    if per_token_timesteps is None:
        # Cast sample back to model compatible dtype
        prev_transport_sample = prev_transport_sample.to(model_output.dtype)
        prev_sample = prev_sample.to(model_output.dtype)

    if not return_dict:
        return (prev_transport_sample, prev_sample)

    return FlowMatchTransportOutput(prev_transport_sample=prev_transport_sample, prev_sample=prev_sample)

class FluxEditPipeline(FluxPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.prompt_cache = {}
        self.scheduler.set_step_index = MethodType(set_step_index, self.scheduler)
        self.scheduler.transport_step = MethodType(transport_step, self.scheduler)
    
    def prepare_image_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        image_latents,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    
    @torch.no_grad()
    # Brought from https://github.com/huggingface/diffusers/blob/4c723d8ec318b5fd266afaf14ba37afdefc967df/examples/community/pipeline_flux_rf_inversion.py#L407
    def encode_source_image(self, 
                            image, 
                            height, 
                            width,
                            dtype,
                            device, 
                            resize_mode="default", crops_coords=None):
        image = self.image_processor.preprocess(
            image=image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        resized = self.image_processor.postprocess(image=image, output_type="pil")

        if max(image.shape[-2:]) > self.vae.config["sample_size"] * 1.5:
            logger.warning(
                "Your input images far exceed the default resolution of the underlying diffusion model. "
                "The output images may contain severe artifacts! "
                "Consider down-sampling the input using the `height` and `width` parameters"
            )
        image = image.to(dtype)

        x0 = self.vae.encode(image.to(device)).latent_dist.sample()
        x0 = (x0 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        x0 = x0.to(dtype)
        return x0, resized
    
    def prepare_prompts(
        self,
        num_images_per_prompt: int,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        source_prompt: Union[str, List[str]] = None,
        source_prompt_2: Optional[Union[str, List[str]]] = None,
        source_negative_prompt: Union[str, List[str]] = None,
        source_negative_prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        true_cfg_scale: float = 1.0,
        source_true_cfg_scale: float = 1.0,
        lora_scale: Optional[float] = None,
        max_sequence_length: int = 512,
    ):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        dtype = self.text_encoder.dtype
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        source_has_neg_prompt = source_negative_prompt is not None or (
            source_negative_prompt_embeds is not None and source_negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        source_do_true_cfg = source_true_cfg_scale > 1 and source_has_neg_prompt
        
        if self.prompt_cache.get("prompt", None) == prompt and self.prompt_cache.get("prompt_2", None) == prompt_2:
            prompt_embeds = self.prompt_cache["prompt_embeds"]
            pooled_prompt_embeds = self.prompt_cache["pooled_prompt_embeds"]
        else:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            self.prompt_cache["prompt"] = prompt
            self.prompt_cache["prompt_2"] = prompt_2
            self.prompt_cache["prompt_embeds"] = prompt_embeds
            self.prompt_cache["pooled_prompt_embeds"] = pooled_prompt_embeds
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(batch_size * num_images_per_prompt, -1)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
            
        if self.prompt_cache.get("source_prompt", None) == source_prompt and self.prompt_cache.get("source_prompt_2", None) == source_prompt_2:
            source_prompt_embeds = self.prompt_cache["source_prompt_embeds"]
            source_pooled_prompt_embeds = self.prompt_cache["source_pooled_prompt_embeds"]
        else:    
            (
                source_prompt_embeds,
                source_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=source_prompt,
                prompt_2=source_prompt_2,
                prompt_embeds=source_prompt_embeds,
                pooled_prompt_embeds=source_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            self.prompt_cache["source_prompt"] = source_prompt
            self.prompt_cache["source_prompt_2"] = source_prompt_2
            self.prompt_cache["source_prompt_embeds"] = source_prompt_embeds
            self.prompt_cache["source_pooled_prompt_embeds"] = source_pooled_prompt_embeds
        seq_len = source_prompt_embeds.shape[1]
        source_prompt_embeds = source_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
        source_pooled_prompt_embeds = source_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(batch_size * num_images_per_prompt, -1)
        source_text_ids = torch.zeros(source_prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
            
        if do_true_cfg:
            if self.prompt_cache.get("negative_prompt", None) == negative_prompt and self.prompt_cache.get("negative_prompt_2", None) == negative_prompt_2:
                negative_prompt_embeds = self.prompt_cache["negative_prompt_embeds"]
                negative_pooled_prompt_embeds = self.prompt_cache["negative_pooled_prompt_embeds"]
            else:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )
                self.prompt_cache["negative_prompt"] = negative_prompt
                self.prompt_cache["negative_prompt_2"] = negative_prompt_2
                self.prompt_cache["negative_prompt_embeds"] = negative_prompt_embeds
                self.prompt_cache["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(batch_size * num_images_per_prompt, -1)
            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            negative_text_ids = None
                
        if source_do_true_cfg:
            if self.prompt_cache.get("source_negative_prompt", None) == source_negative_prompt and self.prompt_cache.get("source_negative_prompt_2", None) == source_negative_prompt_2:
                source_negative_prompt_embeds = self.prompt_cache["source_negative_prompt_embeds"]
                source_negative_pooled_prompt_embeds = self.prompt_cache["source_negative_pooled_prompt_embeds"]
            else:
                (
                    source_negative_prompt_embeds,
                    source_negative_pooled_prompt_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=source_negative_prompt,
                    prompt_2=source_negative_prompt_2,
                    prompt_embeds=source_negative_prompt_embeds,
                    pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )
                self.prompt_cache["source_negative_prompt"] = source_negative_prompt
                self.prompt_cache["source_negative_prompt_2"] = source_negative_prompt_2
                self.prompt_cache["source_negative_prompt_embeds"] = source_negative_prompt_embeds
                self.prompt_cache["source_negative_pooled_prompt_embeds"] = source_negative_pooled_prompt_embeds
            seq_len = source_negative_prompt_embeds.shape[1]
            source_negative_prompt_embeds = source_negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
            source_negative_pooled_prompt_embeds = source_negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(batch_size * num_images_per_prompt, -1)
            source_negative_text_ids = torch.zeros(source_negative_prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        else:
            source_negative_prompt_embeds = None
            source_negative_pooled_prompt_embeds = None
            source_negative_text_ids = None
            
                            
        return (
            prompt_embeds,
            pooled_prompt_embeds,
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            source_negative_prompt_embeds,
            source_negative_pooled_prompt_embeds,
            text_ids,
            source_text_ids,
            negative_text_ids,
            source_negative_text_ids,
        )
    
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        source_image: Optional[PipelineImageInput] = None,
        source_prompt: Union[str, List[str]] = None,
        source_prompt_2: Optional[Union[str, List[str]]] = None,
        source_negative_prompt: Union[str, List[str]] = None,
        source_negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        source_true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        num_skipped_initial_steps: int = 0,
        num_transport_steps: int = 1,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        source_guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        source_latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        step_partitioning_pass: bool = False,
        display_progress: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if image is not None:
            height, width = self.image_processor.preprocess(image).shape[-2:]
        if source_image is not None:
            height, width = self.image_processor.preprocess(source_image).shape[-2:]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        if source_prompt is None:
            logger.warning("UserWarning: source_prompt is None, setting to empty string.")
            source_prompt = ""

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        source_has_neg_prompt = source_negative_prompt is not None or (
            source_negative_prompt_embeds is not None and source_negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        source_do_true_cfg = source_true_cfg_scale > 1 and source_has_neg_prompt
        # (
        #     prompt_embeds,
        #     pooled_prompt_embeds,
        #     text_ids,
        # ) = self.encode_prompt(
        #     prompt=prompt,
        #     prompt_2=prompt_2,
        #     prompt_embeds=prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     device=device,
        #     num_images_per_prompt=num_images_per_prompt,
        #     max_sequence_length=max_sequence_length,
        #     lora_scale=lora_scale,
        # )
        # (
        #     source_prompt_embeds,
        #     source_pooled_prompt_embeds,
        #     source_text_ids,
        # ) = self.encode_prompt(
        #     prompt=source_prompt,
        #     prompt_2=source_prompt_2,
        #     prompt_embeds=source_prompt_embeds,
        #     pooled_prompt_embeds=source_pooled_prompt_embeds,
        #     device=device,
        #     num_images_per_prompt=num_images_per_prompt,
        #     max_sequence_length=max_sequence_length,
        #     lora_scale=lora_scale,
        # )
        # if do_true_cfg:
        #     (
        #         negative_prompt_embeds,
        #         negative_pooled_prompt_embeds,
        #         negative_text_ids,
        #     ) = self.encode_prompt(
        #         prompt=negative_prompt,
        #         prompt_2=negative_prompt_2,
        #         prompt_embeds=negative_prompt_embeds,
        #         pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #         device=device,
        #         num_images_per_prompt=num_images_per_prompt,
        #         max_sequence_length=max_sequence_length,
        #         lora_scale=lora_scale,
        #     )
        # if source_do_true_cfg:
        #     (
        #         source_negative_prompt_embeds,
        #         source_negative_pooled_prompt_embeds,
        #         source_negative_text_ids,
        #     ) = self.encode_prompt(
        #         prompt=source_negative_prompt,
        #         prompt_2=source_negative_prompt_2,
        #         prompt_embeds=source_negative_prompt_embeds,
        #         pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
        #         device=device,
        #         num_images_per_prompt=num_images_per_prompt,
        #         max_sequence_length=max_sequence_length,
        #         lora_scale=lora_scale,
        #     )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            source_negative_prompt_embeds,
            source_negative_pooled_prompt_embeds,
            text_ids,
            source_text_ids,
            negative_text_ids,
            source_negative_text_ids,
        ) = self.prepare_prompts(
            num_images_per_prompt=num_images_per_prompt,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            source_prompt=source_prompt,
            source_prompt_2=source_prompt_2,
            source_negative_prompt=source_negative_prompt,
            source_negative_prompt_2=source_negative_prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            source_prompt_embeds=source_prompt_embeds,
            source_pooled_prompt_embeds=source_pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            source_negative_prompt_embeds=source_negative_prompt_embeds,
            source_negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            true_cfg_scale=true_cfg_scale,
            source_true_cfg_scale=source_true_cfg_scale,
            lora_scale=lora_scale,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        if image is not None:
            has_init_latents = True
            image_latents, _ = self.encode_source_image(
                image,
                height,
                width,
                prompt_embeds.dtype,
                device,
            )
            latents, latent_image_ids = self.prepare_image_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                image_latents,
            )
        else:
            has_init_latents = False
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            
        if source_image is not None:
            source_image_latents, _ = self.encode_source_image(
                source_image,
                height,
                width,
                prompt_embeds.dtype,
                device,
            )
            source_latents, source_latent_image_ids = self.prepare_image_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                source_image_latents,
            )
        else:
            source_latents, source_latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=source_latents,
            )
        source_clean_latents = source_latents.clone()
        if has_init_latents:
            transport_latents = latents - source_latents
        else:
            transport_latents = torch.zeros_like(latents)
        # sanity check
        assert source_latents.shape == latents.shape, "Source latents and target latents must have the same shape"

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
            source_guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
            source_guidance = source_guidance.expand(latents.shape[0])
        else:
            guidance = None
            source_guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
            
        self.scheduler.set_step_index(num_skipped_initial_steps)
        timesteps = timesteps[num_skipped_initial_steps:]
        timesteps = timesteps[:num_transport_steps]
        
        init_noise = randn_tensor(shape=source_clean_latents.shape, generator=generator, device=device, dtype=source_clean_latents.dtype)

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps - num_skipped_initial_steps) if display_progress else nullcontext() as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                noise = randn_tensor(shape=source_clean_latents.shape, generator=generator, device=device, dtype=source_clean_latents.dtype)
                source_latents = self.scheduler.scale_noise(
                    source_clean_latents, timestep / 1000, noise
                )
                # source_latents = self.scheduler.scale_noise(
                #     source_clean_latents, timestep / 1000, init_noise
                # )
                latents = source_latents + transport_latents
                
                with self.transformer.cache_context("source_cond"):
                    self._joint_attention_kwargs = None
                    source_noise_pred = self.transformer(
                        hidden_states=source_latents,
                        timestep=timestep / 1000,
                        guidance=source_guidance,
                        pooled_projections=source_pooled_prompt_embeds,
                        encoder_hidden_states=source_prompt_embeds,
                        txt_ids=source_text_ids,
                        img_ids=source_latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                
                with self.transformer.cache_context("cond"):
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    
                if source_do_true_cfg:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = None
                    with self.transformer.cache_context("source_uncond"):
                        source_neg_noise_pred = self.transformer(
                            hidden_states=source_latents,
                            timestep=timestep / 1000,
                            guidance=source_guidance,
                            pooled_projections=source_negative_pooled_prompt_embeds,
                            encoder_hidden_states=source_negative_prompt_embeds,
                            txt_ids=source_negative_text_ids,
                            img_ids=source_latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    source_noise_pred = source_neg_noise_pred + source_true_cfg_scale * (source_noise_pred - source_neg_noise_pred)

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs = None

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                transport_latents, latents = self.scheduler.transport_step(
                    noise_pred, source_noise_pred, t, transport_latents, latents, source_latents, return_dict=False
                )

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
                        transport_latents = transport_latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if progress_bar is not None and (i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latents = source_clean_latents + transport_latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
    
def create_guidance_pipeline(device="cuda"):
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors", device=device
    )
    pipeline = FluxEditPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    )
    
    return pipeline.to(device)

@dataclass
class StepRequest:
    new_transport_steps: Optional[int] = None
    edited_images: Optional[torch.Tensor] = None
    ordered_edit_ids: Optional[List[int]] = None
    
class IndexWrappedDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]
    
class EditedImageCache:
    @dataclass
    class QueueItem:
        command: Literal["set", "clear"]
        idxs: Optional[List[int]] = None
        images: Optional[torch.Tensor] = None
    def __init__(self, queue):
        self.queue = queue
        
    def setup(self, length, height, width, device, master_dataset: Optional[data.Dataset] = None):
        self.edited = torch.zeros(length, dtype=torch.bool, device=device)
        self.is_master = master_dataset is not None
        if self.is_master:
            self.dataset = master_dataset
            # self.image_cache = torch.zeros((length, 3, height, width), dtype=torch.float32, device=device)
        
    def get(self, idxs):
        if self.is_master:
            while not torch.all(self.edited[idxs]):
                item: EditedImageCache.QueueItem = self.queue.get(block=True)
                if item.command == "set":
                    images = item.images.to(torch.float32).cpu()
                    # self.image_cache[item.idxs] = item.images.to(self.image_cache.dtype).to(self.image_cache.device)
                    self.dataset.set_image(item.idxs, images)
                    self.edited[item.idxs] = True
                elif item.command == "clear":
                    self.edited = torch.zeros_like(self.edited)
            images = [self.dataset[idx].original_image for idx in idxs]
            assert all(img is not None for img in images), f"Some images are still None for idxs {idxs}"
            return images
        else:
            raise NotImplementedError("Only master cache can get images")
    
    def is_present(self, idx):
        return self.edited[idx]
    
    def set(self, idxs, images):
        self.edited[idxs] = True
        self.queue.put(EditedImageCache.QueueItem(command="set", idxs=idxs, images=images))
            
    def clear(self):
        self.edited = torch.zeros_like(self.edited)
        self.queue.put(EditedImageCache.QueueItem(command="clear"))
        
class EditController:
    def __init__(self, request_queues: List[mp.Queue], response_queues: List[mp.Queue], edited_cache: EditedImageCache, step_sizes: Optional[List[int]] = None):
        self.request_queues = request_queues
        self.response_queues = response_queues
        self.edited_cache = edited_cache
        self.step_sizes = step_sizes
        self.current_step = -1
        
    def step(self, new_transport_steps: Optional[int], edited_images: Optional[torch.Tensor], ordered_edit_ids: Optional[List[int]]):
        edit_ids_per_queue = []
        pre_edit_images_per_queue = []
        num_workers = len(self.request_queues)
        for start_idx in range(0, num_workers):
            edit_ids_per_queue.append(ordered_edit_ids[start_idx::num_workers])
            if edited_images is not None:
                pre_edit_images_per_queue.append(edited_images[start_idx::num_workers])
            else:
                pre_edit_images_per_queue.append(None)
        for q, edit_ids, pre_edit_images in zip(self.request_queues, edit_ids_per_queue, pre_edit_images_per_queue):
            q.put(StepRequest(new_transport_steps=new_transport_steps, edited_images=pre_edit_images, ordered_edit_ids=edit_ids))
        
        for response_queue in self.response_queues:
            response_queue.get(block=True)
            
        self.current_step += 1
    
def encode_source_image_batch(pipeline, batch_data, device):
    real_batch_size = len(batch_data)
    batch_source_image = pipeline.image_processor.preprocess(batch_data).to(device)
    height, width = batch_source_image.shape[-2], batch_source_image.shape[-1]
    batch_source_image_latents, _ = pipeline.encode_source_image(
        batch_source_image,
        height,
        width,
        pipeline.dtype,
        device,
    )
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    batch_source_latents, batch_source_latent_image_ids = pipeline.prepare_image_latents(
        real_batch_size,
        num_channels_latents,
        height,
        width,
        pipeline.dtype,
        device,
        batch_source_image_latents,
    )
    return batch_source_latents, batch_source_latent_image_ids

def guidance_worker_fn(rank:int, source_latents_path: str, request_queue: mp.Queue, response_queue: mp.Queue, edited_cache_queue: mp.Queue, image_height: int, image_width: int,
                       prompt: str, source_prompt: str, num_inference_steps: int, initial_skip_steps: int, device: torch.device, 
                       guidance_scale: float = 3.5, source_guidance_scale: float = 1.5):
    logger.info(f"Guidance worker rank {rank} starting on device {device}")

    output_root_dir = os.path.dirname(source_latents_path)
    pipeline = create_guidance_pipeline(device=device)
    source_latents = torch.load(source_latents_path, map_location=device)
    edited_cache = EditedImageCache(edited_cache_queue)
    edited_cache.setup(len(source_latents), None, None, device)
    memory_source_latents = {k: v for k, v in enumerate(source_latents)}
    memory_previous_edited_image = {}
    edited_cache.clear()
    n_skipped_steps = initial_skip_steps
    n_transport_steps = 0
    edit_epoch = -1
    toedit_list = []
    # DATASET_LEN = len(source_images)
    
    with torch.no_grad():
        def generate(frame_idxs: List[int]) -> List[Image.Image]:
            if len(memory_previous_edited_image) == 0:
                previous_images = None
            else:
                previous_images = [memory_previous_edited_image[i] for i in frame_idxs]
            source_latents = torch.stack([memory_source_latents[i] for i in frame_idxs], dim=0)
            images = pipeline(
                prompt=prompt, 
                source_prompt=source_prompt,
                num_inference_steps=num_inference_steps,
                num_skipped_initial_steps=n_skipped_steps, 
                num_transport_steps=n_transport_steps,
                guidance_scale=guidance_scale,
                source_guidance_scale=source_guidance_scale,
                num_images_per_prompt=len(frame_idxs),
                image=previous_images,
                source_latents=source_latents,
                height=image_height,
                width=image_width,
                display_progress=False,
                output_type="pt"
            ).images
            return images
            
        while True:
            if not toedit_list:
                request: StepRequest = request_queue.get(block=True)
                generate_ids = []
                if request is None:
                    break
            else:
                EDIT_BATCH_SIZE = 8
                generate_ids = toedit_list[:EDIT_BATCH_SIZE]
                toedit_list = toedit_list[EDIT_BATCH_SIZE:]
                if request_queue.empty():
                    request = None
                else:
                    request = request_queue.get(block=False)
            
            if request is not None:
                edited_cache.clear()
                n_skipped_steps = n_skipped_steps + n_transport_steps
                n_transport_steps = request.new_transport_steps or n_transport_steps
                toedit_list = request.ordered_edit_ids or []
                
                if request.edited_images is not None:
                    memory_previous_edited_image = {k: v for k, v in zip(toedit_list, request.edited_images)}
                else:
                    memory_previous_edited_image = {}
                edit_epoch = edit_epoch + 1
                response_queue.put(edit_epoch)
            else:
                if edit_epoch < 0:
                    logger.error("Received generate command before any step command. This should not happen.")
                    continue
                frame_ids = [i for i in generate_ids if not edited_cache.is_present(i)]
                images = generate(frame_ids)
                edited_cache.set(frame_ids, images)
    
def setup_guidance_worker_and_dataset(dataset, output_root_dir, source_image_producer, prompt: str, source_prompt: str, num_inference_steps: int, initial_skip_steps: int, main_device: torch.device, available_devices: list[torch.device],
                                            guidance_scale: float = 3.5, source_guidance_scale: float = 1.5):
    cache_path = os.path.join(output_root_dir, "source_latents.pt")
    edit_cache_queue = mp.Queue()
    edit_cache = EditedImageCache(edit_cache_queue)
    if not os.path.exists(cache_path):
        logger.info("Encoding source images and caching latents...")
        source_latents = []
        source_images = source_image_producer()
        ENCODE_BATCH_SIZE = 32
        pipeline = create_guidance_pipeline().to(main_device)
        for batch in tqdm(range(0, len(source_images), ENCODE_BATCH_SIZE), desc="Encoding source images"):
            batch_data = source_images[batch:batch + ENCODE_BATCH_SIZE].to(main_device)
            batch_source_latents, _ = encode_source_image_batch(pipeline, batch_data, main_device)
            source_latents.append(batch_source_latents.cpu())
        source_latents = torch.cat(source_latents, dim=0)
        torch.save(source_latents, cache_path)
        del pipeline
        del source_latents
        torch.cuda.empty_cache()
    
    workers = []
    request_queues = []
    response_queues = []
    for rank, device in enumerate(available_devices):
        request_queue = mp.Queue()
        response_queue = mp.Queue()
        request_queues.append(request_queue)
        response_queues.append(response_queue)
        worker = mp.Process(target=guidance_worker_fn,
                        args=(rank, cache_path, request_queue, response_queue, edit_cache_queue, dataset.override_h, dataset.override_w,
                                prompt, source_prompt, num_inference_steps, initial_skip_steps, device, guidance_scale, source_guidance_scale))
        worker.start()
        workers.append(worker)  
    index_dataset = IndexWrappedDataset(dataset)
    edit_cache.setup(len(dataset), dataset.override_h, dataset.override_w, main_device, master_dataset=dataset)
    
    step_sizes = [4] * ((num_inference_steps - initial_skip_steps) // 4)
    controller = EditController(request_queues, response_queues, edit_cache, step_sizes)
    return workers, controller, edit_cache, index_dataset

if __name__ == "__main__":
    from torch.utils import data
    class VideoFrameDataset(data.Dataset):
        def __init__(self, video_reader, transform=None):
            self.video_reader = video_reader
            self.transform = transform

        def __len__(self):
            return self.video_reader.count_frames()

        def __getitem__(self, idx):
            frame = self.video_reader.get_data(idx)
            if self.transform:
                frame = self.transform(frame)
            return frame
        
    import os
    import imageio.v2 as imageio
    import torchvision.transforms as T
    
    reader = imageio.get_reader(".cache/test_000.mp4")
    dataset = VideoFrameDataset(reader, transform=T.ToTensor())
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    pipeline = create_guidance_pipeline()
    # pipeline = FluxEditPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    # ).to("cuda")
    # pipeline.enable_vae_slicing()

    os.makedirs("outputs", exist_ok=True)
    for batch_idx, frames in enumerate(dataloader):
        transport_step_per_iter = 9
        images = None
        for iter_index, skipped_steps in enumerate(range(14, 50, transport_step_per_iter)):
            frames = frames.to("cuda")
            images = pipeline(
                prompt="A realistic medium shot of a woman wearing a light yellow shirt in an apron slicing meat in a cluttered kitchen at night.", 
                source_prompt="A realistic medium shot of a man in an apron slicing meat in a cluttered kitchen at night.",
                num_inference_steps=50,
                num_skipped_initial_steps=skipped_steps, 
                num_transport_steps=transport_step_per_iter,
                guidance_scale=3.5,
                source_guidance_scale=1.5,
                num_images_per_prompt=len(frames),
                image=images,
                source_image=frames,
                output_type="pil"
            ).images

            for i, img in enumerate(images):
                img.save(f"outputs/flux_edit_iter{iter_index}_{batch_idx * len(frames) + i}.png")