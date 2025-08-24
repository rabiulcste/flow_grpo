# Bagel Pipeline with Logprob Support for Flow-GRPO
# Properly adapted for Bagel's unified multimodal architecture
#
# CRITICAL: Use Flow-GRPO's existing SDE implementation
# BAGEL: v_t = x_1 - x_0 (raw velocity prediction from data to noise)
# Flow-GRPO: Uses sde_step_with_logprob for SDE-based RL training
# This pipeline converts BAGEL's velocity predictions to Flow-GRPO's expected format
#
# MODIFICATIONS vs ORIGINAL BAGEL CODE:
# - BagelPipeline class: MODIFIED - adapted for Flow-GRPO training using existing SDE

# - All other methods marked with "ORIGINAL BAGEL LOGIC" or "MODIFIED FOR FLOW-GRPO"

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import math
from diffusers.utils.torch_utils import randn_tensor
import logging
from PIL import Image
from tqdm import tqdm
from bagel.modeling.bagel.qwen2_navit import NaiveCache
from bagel.data.data_utils import patchify


logger = logging.getLogger(__name__)


def sde_step_with_logprob_packed(
    scheduler,
    model_output_packed: torch.FloatTensor,  # Packed velocity from Bagel
    timestep: Union[float, torch.FloatTensor],
    sample_packed: torch.FloatTensor,  # Packed latents
    batch_size: int,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Bagel-native SDE step computation that works directly on packed format.
    This avoids unpacking/repacking and is mathematically equivalent to standard SDE.
    
    Args:
        scheduler: FlowMatchEulerDiscreteScheduler instance
        model_output_packed: Velocity in Bagel's packed format (total_patches, channels * patch_size^2)
        timestep: Current timestep(s)
        sample_packed: Current latents in Bagel's packed format  
        batch_size: Number of samples in the batch
        noise_level: SDE noise level (default 0.7)
        prev_sample: Optional previous sample (packed format)
        generator: Random number generator
        
    Returns:
        prev_sample_packed: Next sample in packed format
        log_prob: Log probability per sample (batch_size,)
        prev_sample_mean_packed: Mean prediction in packed format
        std_dev_t: Standard deviation tensor
    """
    # Convert to fp32 for numerical stability (same as original SDE)
    model_output_packed = model_output_packed.float()
    sample_packed = sample_packed.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()
    
    # Get scheduler coefficients (same as standard SDE)
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    
    # For packed format: sigma shape should broadcast properly
    sigma = scheduler.sigmas[step_index].view(-1, 1)  # (batch_size, 1)
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, 1)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma
    
    # Compute noise level (same math as standard SDE)
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    
    # Patches per sample (for proper reshaping)
    patches_per_sample = sample_packed.shape[0] // batch_size
    channels_times_patch_sq = sample_packed.shape[1]
    
    # Reshape for batch-wise operations: (batch_size, patches_per_sample, channels * patch_size^2)
    sample_reshaped = sample_packed.view(batch_size, patches_per_sample, channels_times_patch_sq)
    model_output_reshaped = model_output_packed.view(batch_size, patches_per_sample, channels_times_patch_sq)
    
    # Broadcast coefficients to match packed format
    sigma_broadcast = sigma.unsqueeze(-1)  # (batch_size, 1, 1)
    dt_broadcast = dt.unsqueeze(-1)       # (batch_size, 1, 1)
    std_dev_t_broadcast = std_dev_t.unsqueeze(-1)  # (batch_size, 1, 1)
    
    # SDE update (exactly same math as standard, but on packed tensors)
    prev_sample_mean_reshaped = (
        sample_reshaped * (1 + std_dev_t_broadcast**2 / (2 * sigma_broadcast) * dt_broadcast) +
        model_output_reshaped * (1 + std_dev_t_broadcast**2 * (1 - sigma_broadcast) / (2 * sigma_broadcast)) * dt_broadcast
    )
    
    if prev_sample is None:
        # Generate noise in packed format (same distribution as standard)
        variance_noise = randn_tensor(
            (batch_size, patches_per_sample, channels_times_patch_sq),
            generator=generator,
            device=model_output_packed.device,
            dtype=model_output_packed.dtype,
        )
        prev_sample_reshaped = prev_sample_mean_reshaped + std_dev_t_broadcast * torch.sqrt(-1 * dt_broadcast) * variance_noise
    else:
        prev_sample_reshaped = prev_sample.view(batch_size, patches_per_sample, channels_times_patch_sq)
    
    # Compute log probability (same math as standard, different reduction)
    noise_variance = (std_dev_t_broadcast * torch.sqrt(-1 * dt_broadcast))**2
    log_prob_per_element = (
        -((prev_sample_reshaped.detach() - prev_sample_mean_reshaped) ** 2) / (2 * noise_variance)
        - torch.log(std_dev_t_broadcast * torch.sqrt(-1 * dt_broadcast))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    
    # Reduce over spatial dimensions (patches and channels), keep batch dimension
    # This is mathematically equivalent to .mean(dim=(1,2,3)) on unpacked format
    log_prob = log_prob_per_element.mean(dim=(1, 2))  # (batch_size,)
    
    # Flatten back to packed format for return
    prev_sample_packed = prev_sample_reshaped.view(-1, channels_times_patch_sq)
    prev_sample_mean_packed = prev_sample_mean_reshaped.view(-1, channels_times_patch_sq)
    
    return prev_sample_packed, log_prob, prev_sample_mean_packed, std_dev_t.squeeze()


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input

def patchify(latent, patch_size):
    """Simple patchify function for Bagel format."""
    c, h, w = latent.shape
    patches = latent.reshape(c, h // patch_size, patch_size, w // patch_size, patch_size)
    patches = torch.einsum("chpwq->hwpqc", patches)
    patches = patches.reshape(-1, c * patch_size * patch_size)
    return patches

class BagelPipeline:
    """
    MODIFIED: Bagel pipeline adapted for Flow-GRPO training.
    
    ORIGINAL BAGEL: Standard inference pipeline
    MODIFICATIONS: 
    - Returns velocity predictions instead of noise
    - Integrates with Bagel's native methods directly
    - Handles log probability computation for RL training
    - Returns dictionary format expected by training script
    """
    
    def __init__(self, bagel_model, vae, tokenizer, scheduler, new_token_ids):
        # ORIGINAL BAGEL: Simple initialization like gen_images_mp.py
        self.bagel_model = bagel_model
        self.vae = vae
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.new_token_ids = new_token_ids  # Pass in like original Bagel
        


    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 77,
        noise_level: float = 0.7,
    ):
        """
        CORRECTED: Handle multiple prompts with num_images_per_prompt efficiently.
        
        Example: 10 prompts × 6 images per prompt = 60 total samples
        We create a flattened batch where each prompt is repeated num_images_per_prompt times.
        """
        # Set default resolution for Bagel (1024x1024)
        height = height or 1024
        width = width or 1024
        
        if prompt is None:
            raise ValueError("prompt must be provided")
        
        # 1. Handle prompt input - convert to list for uniform processing
        if isinstance(prompt, str):
            prompts_list = [prompt]
        elif isinstance(prompt, list):
            prompts_list = prompt
        else:
            raise ValueError("prompt must be a string or list of strings")
        
        # 2. Create expanded prompt list for batch processing
        # Example: ["prompt1", "prompt2"] with num_images_per_prompt=3
        # becomes: ["prompt1", "prompt1", "prompt1", "prompt2", "prompt2", "prompt2"]
        expanded_prompts = []
        for prompt_text in prompts_list:
            expanded_prompts.extend([prompt_text] * num_images_per_prompt)
        
        total_batch_size = len(expanded_prompts)
        print(f"Processing {len(prompts_list)} prompts × {num_images_per_prompt} images = {total_batch_size} total samples")
        
        # 3. Generate all samples in one batch using Bagel's native pattern
        return self._generate_batch(
            prompts=expanded_prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type,
            noise_level=noise_level,
        )
    
    def _generate_batch(
        self,
        prompts: List[str],
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        output_type: str,
        noise_level: float,
    ):
        """
        Generate images for a batch of prompts using Bagel's native pattern.
        
        Args:
            prompts: List of prompts (already expanded with repetitions)
                    e.g., ["cat", "cat", "dog", "dog"] for 2 prompts × 2 images each
        """
        batch_size = len(prompts)
        self.bagel_model.eval()
        
        # BAGEL NATIVE: Setup for the full batch
        past_key_values = NaiveCache(self.bagel_model.config.llm_config.num_hidden_layers)
        newlens = [0] * batch_size
        new_rope = [0] * batch_size
        
        # BAGEL NATIVE: Prepare prompts (now we can handle different prompts in batch)
        generation_input, newlens, new_rope = self.bagel_model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=prompts,  # Use the expanded prompt list
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        
        # BAGEL NATIVE: Update text cache
        device = next(self.bagel_model.parameters()).device
        generation_input = move_generation_input_to_device(generation_input, device)
        
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                past_key_values = self.bagel_model.forward_cache_update_text(past_key_values, **generation_input)
    
        
        # BAGEL NATIVE: Prepare VAE latent for the full batch (this REPLACES generation_input!)
        generation_input = self.bagel_model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            image_sizes=[(height, width)] * batch_size,
            new_token_ids=self.new_token_ids,
        )
        
        generation_input = move_generation_input_to_device(generation_input, device)
        
        
        # BAGEL NATIVE: CFG setup for the full batch  
        cfg_past_key_values = NaiveCache(self.bagel_model.config.llm_config.num_hidden_layers)
        # CFG should start with empty cache but correct positioning
        cfg_newlens = [0] * batch_size
        cfg_new_rope = [0] * batch_size
        
        generation_input_cfg = self.bagel_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_newlens,
            curr_rope=cfg_new_rope, 
            image_sizes=[(height, width)] * batch_size,
        )
        generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)
        
        # CFG should use same sequence structure but empty KV cache
        # Calculate proper CFG indexes that match the current sequence layout
        h, w = height // self.bagel_model.latent_downsample, width // self.bagel_model.latent_downsample
        num_image_tokens = h * w
        
        # For each sample in batch: start_token + image_tokens + end_token = num_image_tokens + 2
        cfg_seqlens_per_sample = num_image_tokens + 2
        cfg_total_seqlens = cfg_seqlens_per_sample * batch_size
        
        # Create CFG query indexes that span the same range as main generation
        # but assume no KV cache (start from 0)
        cfg_query_indexes = []
        for i in range(batch_size):
            start_idx = i * cfg_seqlens_per_sample
            # start_of_image token
            cfg_query_indexes.append(start_idx)
            # image tokens  
            cfg_query_indexes.extend(range(start_idx + 1, start_idx + 1 + num_image_tokens))
            # end_of_image token
            cfg_query_indexes.append(start_idx + 1 + num_image_tokens)
        
        # Override the CFG parameters with corrected ones
        generation_input_cfg["cfg_packed_query_indexes"] = torch.tensor(cfg_query_indexes, dtype=torch.long, device=device)
        generation_input_cfg["cfg_key_values_lens"] = torch.zeros(batch_size, dtype=torch.int, device=device)
        generation_input_cfg["cfg_packed_key_value_indexes"] = torch.tensor([], dtype=torch.long, device=device)
                
        # BAGEL NATIVE: Follow exact inference pattern with correct batch size
        with torch.no_grad():
            # Initialize exactly like Bagel's generate_image
            x_t = generation_input['packed_init_noises']  # Stay in packed format like Bagel
                
            # Setup scheduler with Bagel's timestep range and get timesteps
            timesteps = self._setup_bagel_compatible_scheduler(num_inference_steps, device)
            dts = timesteps[:-1] - timesteps[1:]  # Bagel's dt calculation (not used currently)
            
            # Storage for GRPO (optimized memory usage)
            all_latents_packed = []
            all_log_probs = []
            
            # Store initial state in packed format (Bagel-native)
            h, w = height // self.bagel_model.latent_downsample, width // self.bagel_model.latent_downsample
            all_latents_packed.append(x_t.detach().clone())  # Detach to save memory
            
            # Denoising loop - FULLY Bagel-native (no unpacking/repacking!)
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising loop"):
                    
                    # Bagel's timestep preparation - Use x_t.shape[0] like original Bagel
                    timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)  # Match original Bagel exactly
                    
                    # CRITICAL FIX: Match original generate_image parameter mapping EXACTLY
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        v_t = self.bagel_model._forward_flow(
                        x_t=x_t,  # packed_init_noises -> x_t (correct)
                        timestep=timestep,  # Created correctly above
                        # All parameters now come from the unified generation_input
                        packed_vae_token_indexes=generation_input['packed_vae_token_indexes'],
                        packed_vae_position_ids=generation_input['packed_vae_position_ids'],
                        # Sequence structure parameters
                        packed_indexes=generation_input['packed_indexes'],
                        packed_position_ids=generation_input['packed_position_ids'],
                        packed_seqlens=generation_input['packed_seqlens'],
                        # Text-related parameters
                        packed_text_ids=generation_input['packed_text_ids'],
                        packed_text_indexes=generation_input['packed_text_indexes'],
                        # KV cache parameters
                        key_values_lens=generation_input['key_values_lens'],
                        past_key_values=past_key_values,
                        packed_key_value_indexes=generation_input['packed_key_value_indexes'],
                        # MISSING: All CFG parameters from official Bagel
                        cfg_renorm_min=0.0,  # Default from gen_images_mp.py
                        cfg_renorm_type="global",  # Default
                        cfg_text_scale=guidance_scale,  # Use guidance_scale parameter
                        cfg_text_packed_position_ids=generation_input_cfg.get("cfg_packed_position_ids"),
                        cfg_text_packed_query_indexes=generation_input_cfg.get("cfg_packed_query_indexes"),
                        cfg_text_key_values_lens=generation_input_cfg.get("cfg_key_values_lens"),
                        cfg_text_past_key_values=cfg_past_key_values,
                        cfg_text_packed_key_value_indexes=generation_input_cfg.get("cfg_packed_key_value_indexes"),
                        # cfg_img parameters (MISSING from our implementation!)
                        cfg_img_scale=1.0,  # Default for image CFG
                        cfg_img_packed_position_ids=generation_input_cfg.get("cfg_packed_position_ids"),
                        cfg_img_packed_query_indexes=generation_input_cfg.get("cfg_packed_query_indexes"),
                        cfg_img_key_values_lens=generation_input_cfg.get("cfg_key_values_lens"),
                        cfg_img_past_key_values=cfg_past_key_values,
                        cfg_img_packed_key_value_indexes=generation_input_cfg.get("cfg_packed_key_value_indexes"),
                        cfg_type="parallel",  # Default CFG type
                    )
                    
                    # BAGEL-NATIVE SDE: Use packed SDE computation directly (NO format conversion!)
                    # Use Bagel's timestep value, not Flow GRPO scheduler's timestep
                    # Convert Bagel's scalar timestep to batch tensor for Flow GRPO SDE compatibility
                    timestep_tensor = torch.tensor([t] * batch_size, device=device)
                    x_t, log_prob, _, _ = sde_step_with_logprob_packed(
                        self.scheduler,  # Use pipeline's configured scheduler
                        v_t,  # velocity in packed format (Bagel-native!)
                        timestep_tensor,  # Batch tensor of timesteps
                        x_t,  # current state in packed format (Bagel-native!)
                        batch_size,  # Use the correct full batch size
                        noise_level=noise_level,
                        generator=None,
                    )
                    
                    # Store for GRPO (optimized memory usage)
                    all_latents_packed.append(x_t.detach().clone())  # Detach to save memory
                    all_log_probs.append(log_prob.detach())  # Detach to save memory
                
        # Convert packed latents to standard format for GRPO compatibility (only when needed)
        # Ensure latents are in float32 when leaving autocast context (like original Bagel)
        latents = []
        for latent_packed in all_latents_packed:
            latent_standard = self._unpack_velocity_from_bagel_format(latent_packed, batch_size, h, w)
            latents.append(latent_standard.float())  # Ensure float32 for VAE compatibility
        log_probs = all_log_probs
        
        # Stack latents to match SD3 format: (batch_size, num_steps + 1, channels, height, width)
        latents = torch.stack(latents, dim=1)
        
        # Stack log_probs to match SD3 format: (batch_size, num_steps)
        log_probs = torch.stack(log_probs, dim=1)
        
        # Convert to output format
        if output_type == "latent":
            images = torch.stack(latents)
        else:
            # BAGEL NATIVE: Decode ONLY FINAL latents for reward computation (like Flow-GRPO)
            # latents shape: (batch_size, num_steps + 1, channels, height, width)
            # We want the final timestep for ALL samples, not all timesteps for the last sample
            final_latents = latents[:, -1]  # Get the final latents after all denoising steps for ALL samples
            
            images = []
            # Process each sample separately (Bagel processes one by one)
            for sample_idx in range(final_latents.shape[0]):
                single_latent = final_latents[sample_idx:sample_idx+1]  # Keep batch dim
                
                # line 91 - 94 in gen_images_mp.py
                single_latent = single_latent.reshape(1, height//16, width//16, 2, 2, 16)
                single_latent = torch.einsum("nhwpqc->nchpwq", single_latent)
                single_latent = single_latent.reshape(1, 16, height//8, width//8)
                
                # Decode with VAE (exactly like line 94 in gen_images_mp.py)
                image = self.vae.decode(single_latent.to(device))
                
                if output_type == "pil":
                    # line 95 in gen_images_mp.py
                    tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                    tmpimage = Image.fromarray(tmpimage)
                    images.append(tmpimage)
                else:
                    images.append(image)
            
            if output_type != "pil":
                images = torch.cat(images, dim=0)
        
        # Return format for batch processing
        return {
            "images": images,
            "latents": latents,
            "log_probs": log_probs,
            "past_key_values": past_key_values  # Store cache for efficient training (Bagel naming)
        }
    
    
    def _unpack_velocity_from_bagel_format(self, packed_velocity, batch_size, h, w):
        """
        Convert packed velocity back to standard format using Bagel's unpacking pattern.
        
        This uses the same einsum pattern found in Bagel's generation scripts.
        """
        latent_patch_size = self.bagel_model.latent_patch_size
        latent_channel = self.bagel_model.latent_channel
        
        # Split packed velocity by sample
        patches_per_sample = h * w
        split_velocities = packed_velocity.split([patches_per_sample] * batch_size, dim=0)
        
        unpacked_velocities = []
        for velocity in split_velocities:
            # Reshape from (patches, channels) to (h, w, p, p, c) format
            velocity = velocity.reshape(h, w, latent_patch_size, latent_patch_size, latent_channel)
            
            # Use Bagel's native unpacking einsum pattern
            velocity = torch.einsum("hwpqc->chpwq", velocity)
            velocity = velocity.reshape(latent_channel, h * latent_patch_size, w * latent_patch_size)
            
            unpacked_velocities.append(velocity)
        
        # Stack to get (batch, channels, height, width)
        return torch.stack(unpacked_velocities, dim=0)
    
    def _create_bagel_timesteps(self, num_timesteps, device):
        """
        Create timesteps using Bagel's native timestep generation logic.
        
        This reuses the exact same logic from Bagel's generate_image method.
        """
        timesteps = torch.linspace(1, 0, num_timesteps, device=device)
        timestep_shift = self.bagel_model.timestep_shift
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        return timesteps
    
    def _setup_bagel_compatible_scheduler(self, num_timesteps, device):
        """
        Configure the pipeline's scheduler to be compatible with Bagel's timestep range.
        
        This ensures scheduler.index_for_timestep() works correctly with Bagel's values.
        """
        # Generate Bagel's timestep schedule
        timesteps = self._create_bagel_timesteps(num_timesteps, device)
        # Don't remove last timestep - keep all timesteps for proper training
        
        # Configure scheduler to use Bagel's timesteps
        self.scheduler.num_train_timesteps = num_timesteps
        self.scheduler.set_timesteps(num_timesteps, device=device)
        # Override scheduler's timesteps with Bagel's timestep schedule AFTER set_timesteps
        self.scheduler.timesteps = timesteps.clone()
        
        return timesteps
    



def compute_log_prob_bagel_native(bagel_model, sample, timestep_idx, config, tokenizer, new_token_ids, scheduler=None, policy_train=False):
    """
    OPTIMIZED: Compute log probability using Bagel's packed format throughout.
    Eliminates redundant unpacking/repacking operations.
    """

    device = next(bagel_model.parameters()).device        
 
    current_latents = sample["latents"][:, timestep_idx] 
    batch_size = current_latents.shape[0]
    
    # Simple Bagel setup (like gen_images_mp.py)
    past_key_values = NaiveCache(bagel_model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size
    h, w = current_latents.shape[2], current_latents.shape[3]
    resolution = h * 8
    
    # Text embeddings (simple)
    generation_input, newlens, new_rope = bagel_model.prepare_prompts(
        curr_kvlens=newlens, curr_rope=new_rope, 
        prompts=sample["prompts"], tokenizer=tokenizer, new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    
    # Add policy_train to generation_input so it gets passed via **kwargs
    generation_input['policy_train'] = policy_train
    
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        past_key_values = bagel_model.forward_cache_update_text(past_key_values, **generation_input)
    
    # VAE inputs (simple)
    generation_input = bagel_model.prepare_vae_latent(
        curr_kvlens=newlens, curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * batch_size, new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    
    # Add policy_train to VAE generation_input as well
    generation_input['policy_train'] = policy_train
    
    # OPTIMIZATION: Convert to packed format once and stay in packed format
    current_latents_packed = []
    for i in range(batch_size):
        single_latent = current_latents[i]  # (C, H, W)
        packed_single = patchify(single_latent, bagel_model.latent_patch_size)
        current_latents_packed.append(packed_single)
    current_latents_packed = torch.cat(current_latents_packed, dim=0).to(torch.float32).to(device)
    
    # Get velocity in packed format
    timestep_tensor = torch.tensor([sample["timesteps"][0, timestep_idx]] * current_latents_packed.shape[0], device=device).to(torch.float32)
    

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        velocity_pred_packed = bagel_model._forward_flow(
            x_t=current_latents_packed, 
            timestep=timestep_tensor, 
            # Pass individual parameters explicitly like original Bagel
            packed_vae_token_indexes=generation_input['packed_vae_token_indexes'],
            packed_vae_position_ids=generation_input['packed_vae_position_ids'],
            packed_text_ids=generation_input['packed_text_ids'],
            packed_text_indexes=generation_input['packed_text_indexes'],
            packed_indexes=generation_input['packed_indexes'],
            packed_position_ids=generation_input['packed_position_ids'],
            packed_seqlens=generation_input['packed_seqlens'],
            key_values_lens=generation_input['key_values_lens'],
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input['packed_key_value_indexes'],
            # Add missing CFG parameters to match original Bagel signature
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            cfg_text_scale=1.0,  # No CFG for training
            cfg_text_packed_position_ids=None,
            cfg_text_packed_query_indexes=None,
            cfg_text_key_values_lens=None,
            cfg_text_past_key_values=None,
            cfg_text_packed_key_value_indexes=None,
            cfg_img_scale=1.0,
            cfg_img_packed_position_ids=None,
            cfg_img_packed_query_indexes=None,
            cfg_img_key_values_lens=None,
            cfg_img_past_key_values=None,
            cfg_img_packed_key_value_indexes=None,
            cfg_type="parallel",
            policy_train=policy_train,
        )
        
    # OPTIMIZATION: Use packed SDE step directly (no unpacking needed)
    prev_sample_packed, log_prob, prev_sample_mean_packed, std_dev_t = sde_step_with_logprob_packed(
        scheduler,
        velocity_pred_packed,  # Already in packed format
        sample["timesteps"][:, timestep_idx],
        current_latents_packed,  # Already in packed format
        batch_size,
        prev_sample=sample["next_latents"][:, timestep_idx] if "next_latents" in sample else None,
        noise_level=config.sample.noise_level,
    )
    
    # Convert back to standard format only for return (if needed by training loop)
    # This is the only conversion needed
    # Calculate the correct number of patches per sample based on the actual packed tensor size
    patches_per_sample = prev_sample_packed.shape[0] // batch_size
    split_prev_samples = prev_sample_packed.split([patches_per_sample] * batch_size, dim=0)
    split_prev_means = prev_sample_mean_packed.split([patches_per_sample] * batch_size, dim=0)
    
    prev_sample_standard = []
    prev_sample_mean_standard = []
    
    for prev_sample_single, prev_mean_single in zip(split_prev_samples, split_prev_means):
        # Calculate the actual patch grid dimensions based on Bagel's logic
        # patches_per_sample = h_patches * w_patches where h_patches = h // patch_size, w_patches = w // patch_size
        h_patches = h // bagel_model.latent_patch_size
        w_patches = w // bagel_model.latent_patch_size
        
        # Unpack single sample using Bagel's exact unpacking pattern
        # From Bagel's gen_images_mp.py: latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        prev_sample_unpacked = prev_sample_single.reshape(1, h_patches, w_patches, bagel_model.latent_patch_size, bagel_model.latent_patch_size, bagel_model.latent_channel)
        # From Bagel: torch.einsum("nhwpqc->nchpwq", latent)
        prev_sample_unpacked = torch.einsum("nhwpqc->nchpwq", prev_sample_unpacked)
        # From Bagel: latent.reshape(1, 16, resolution//8, resolution//8)
        prev_sample_unpacked = prev_sample_unpacked.reshape(1, bagel_model.latent_channel, h, w)
        prev_sample_unpacked = prev_sample_unpacked.squeeze(0)  # Remove batch dimension
        
        prev_mean_unpacked = prev_mean_single.reshape(1, h_patches, w_patches, bagel_model.latent_patch_size, bagel_model.latent_patch_size, bagel_model.latent_channel)
        prev_mean_unpacked = torch.einsum("nhwpqc->nchpwq", prev_mean_unpacked)
        prev_mean_unpacked = prev_mean_unpacked.reshape(1, bagel_model.latent_channel, h, w)
        prev_mean_unpacked = prev_mean_unpacked.squeeze(0)  # Remove batch dimension
        
        prev_sample_standard.append(prev_sample_unpacked)
        prev_sample_mean_standard.append(prev_mean_unpacked)
    
    prev_sample = torch.stack(prev_sample_standard, dim=0)
    prev_sample_mean = torch.stack(prev_sample_mean_standard, dim=0)
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
    



