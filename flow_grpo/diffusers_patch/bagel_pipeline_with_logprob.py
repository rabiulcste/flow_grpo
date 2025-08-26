# Bagel Pipeline with Logprob Support for Flow-GRPO
# Properly adapted for Bagel's unified multimodal architecture
#
# CRITICAL: Use BAGEL's native ODE step for sample collection, add noise for RL
# BAGEL: v_t = x_1 - x_0 (raw velocity prediction from data to noise)
# Flow-GRPO: Needs SDE for RL training (noise + log probability)
# This pipeline uses BAGEL's ODE step + noise injection for RL compatibility

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


def bagel_sde_step_with_logprob_packed(
    v_t_packed: torch.FloatTensor,  # BAGEL's velocity prediction in packed format
    x_t_packed: torch.FloatTensor,  # Current latents in packed format
    dt: float,  # BAGEL's timestep difference
    noise_level: float = 0.7,
    batch_size: int = 1,
    generator: Optional[torch.Generator] = None,
):
    """
    Fundamentally correct BAGEL-native SDE: ODE + noise for RL exploration.
    
    Args:
        v_t_packed: BAGEL's velocity prediction (packed format)
        x_t_packed: Current latents (packed format)
        dt: BAGEL's timestep difference
        noise_level: Noise level for RL exploration (default 0.7, like SD3)
        batch_size: Number of samples in batch
        generator: Random number generator
        
    Returns:
        x_next_packed: Next latents in packed format
        log_prob: Log probability per sample (batch_size,)
    """
    # 1. BAGEL's native ODE step (velocity-based)
    x_next_mean_packed = x_t_packed - v_t_packed * dt
    
    # 2. Add noise for RL exploration
    if noise_level > 0:
        noise_scale = noise_level * torch.sqrt(torch.abs(dt))
        noise_packed = randn_tensor(
            x_t_packed.shape,
            generator=generator,
            device=x_t_packed.device,
            dtype=x_t_packed.dtype,
        )
        x_next_packed = x_next_mean_packed + noise_scale * noise_packed
    else:
        x_next_packed = x_next_mean_packed
    
    # 3. Compute log probability for RL training
    if noise_level > 0:
        log_prob = (
            -((x_next_packed.detach() - x_next_mean_packed) ** 2) / (2 * noise_scale ** 2)
            - torch.log(noise_scale)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=x_t_packed.device)))
        )
        # Average over spatial dimensions, keep batch dimension
        log_prob = log_prob.mean(dim=1)  # (total_patches,)
        
        # Aggregate log_prob across patches to get per-sample log probability
        patches_per_sample = x_t_packed.shape[0] // batch_size
        log_prob_per_sample = []
        for i in range(batch_size):
            start_idx = i * patches_per_sample
            end_idx = (i + 1) * patches_per_sample
            sample_log_prob = log_prob[start_idx:end_idx].mean()  # Average across patches
            log_prob_per_sample.append(sample_log_prob)
        log_prob = torch.stack(log_prob_per_sample)  # (batch_size,)
    else:
        log_prob = torch.zeros(batch_size, device=x_t_packed.device)
    
    return x_next_packed, log_prob


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
        
        # total_batch_size = len(expanded_prompts)
        # print(f"Processing {len(prompts_list)} prompts × {num_images_per_prompt} images = {total_batch_size} total samples")
        
        # 3. Generate all samples in one batch using Bagel's native pattern
        return self._generate_batch(
            prompts=expanded_prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type,
            noise_level=noise_level,
            generator=generator,
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
        generator: Optional[torch.Generator] = None,
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
                
            # Setup timesteps (BAGEL native)
            timesteps = self._setup_bagel_compatible_scheduler(num_inference_steps, device)
            dts = timesteps[:-1] - timesteps[1:]  # BAGEL's dt calculation
            timesteps = timesteps[:-1]  # Remove last timestep like original BAGEL
            
            # Storage for GRPO (optimized memory usage)
            all_latents_packed = []
            all_log_probs = []
            
            # Store initial state in packed format (Bagel-native)
            h, w = height // self.bagel_model.latent_downsample, width // self.bagel_model.latent_downsample
            all_latents_packed.append(x_t.detach().clone())  # Detach to save memory
            
            # Denoising loop - FULLY Bagel-native (no unpacking/repacking!)
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False, position=1, desc="Denoising loop"):
                    
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
                    
                    # BAGEL-native SDE step (fundamentally correct)
                    dt = dts[i] if i < len(dts) else timesteps[i]  # Use BAGEL's native dt
                    x_t, log_prob = bagel_sde_step_with_logprob_packed(
                        v_t_packed=v_t,  # BAGEL's velocity prediction
                        x_t_packed=x_t,  # Current latents in packed format
                        dt=dt,  # BAGEL's timestep difference
                        noise_level=noise_level,
                        batch_size=batch_size,
                        generator=generator,
                    )
                    
                    # Store for GRPO (optimized memory usage)
                    all_latents_packed.append(x_t.detach().clone())  # Detach to save memory
                    all_log_probs.append(log_prob.detach())  # Detach to save memory
                
        # Keep latents in packed format for efficiency (Bagel native format)
        # Only unpack when needed for VAE decoding
        latents_packed = all_latents_packed  # Keep in packed format
        log_probs = all_log_probs
        
        # Stack packed latents: (num_steps + 1, total_packed_tokens, features)
        latents_packed = torch.stack(latents_packed, dim=0)
        
        # Stack log_probs to match SD3 format: (batch_size, num_steps)
        log_probs = torch.stack(log_probs, dim=1)
        
        # Convert to output format
        if output_type == "latent":
            # Return packed latents for latent output
            images = latents_packed
        else:
            # BAGEL NATIVE: Decode ONLY FINAL latents for reward computation (like Flow-GRPO)
            # latents_packed shape: (num_steps + 1, total_packed_tokens, features)
            # We want the final timestep for ALL samples
            final_latents_packed = latents_packed[-1]  # Get the final packed latents
            
            # Use EXACT same unpacking logic as original Bagel
            unpacked_latent = final_latents_packed.split((generation_input['packed_seqlens'] - 2).tolist())
            
            images = []
            # Process each sample exactly like original Bagel
            for latent in unpacked_latent:
                # Original Bagel's exact unpacking logic:
                latent = latent.reshape(1, h, w, self.bagel_model.latent_patch_size, self.bagel_model.latent_patch_size, self.bagel_model.latent_channel)
                latent = torch.einsum("nhwpqc->nchpwq", latent)
                latent = latent.reshape(1, self.bagel_model.latent_channel, h * self.bagel_model.latent_patch_size, w * self.bagel_model.latent_patch_size)
                
                # Decode with VAE (exactly like original Bagel)
                image = self.vae.decode(latent.to(device))
                
                if output_type == "pil":
                    # Original Bagel's exact conversion logic
                    tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                    tmpimage = Image.fromarray(tmpimage)
                    images.append(tmpimage)
                else:
                    images.append(image)
            
            if output_type != "pil":
                images = torch.cat(images, dim=0)
        
        # Return format for Flow-GRPO (match training script expectations)
        return {
            "images": images,
            "latents": latents_packed,  # Return packed latents for efficiency
            "log_probs": log_probs,
            "past_key_values": past_key_values,  # Store cache for efficient training
        }
    
    

    
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

def compute_log_prob_bagel_native(bagel_model, sample, timestep_idx, config, tokenizer, new_token_ids, policy_train=False):
    """
    BAGEL NATIVE: Compute log probability using Bagel's packed format throughout.
    Returns packed format for maximum efficiency: log_prob, prev_sample_mean_packed, std_dev_t.
    """

    device = next(bagel_model.parameters()).device        
    
    prompts = tokenizer.batch_decode(sample["prompts_ids"], skip_special_tokens=True)

    # Get packed latents directly from sample (no conversion needed!)
    current_latents_packed = sample["latents"][timestep_idx]  # Already in packed format
    batch_size = len(prompts)  # Get batch size from prompts
    
    # BAGEL NATIVE: Follow the exact same pattern as generate_image
    # 1. Setup like original Bagel
    past_key_values = NaiveCache(bagel_model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size
    
    # 2. Prepare prompts exactly like original Bagel
    generation_input, newlens, new_rope = bagel_model.prepare_prompts(
        curr_kvlens=newlens, curr_rope=new_rope, 
        prompts=prompts, tokenizer=tokenizer, new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    generation_input["policy_train"] = policy_train
    
    # 3. Update text cache exactly like original Bagel
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        past_key_values = bagel_model.forward_cache_update_text(past_key_values, **generation_input)
    
    # 4. Prepare VAE latent exactly like original Bagel
    resolution = config.resolution
    generation_input = bagel_model.prepare_vae_latent(
        curr_kvlens=newlens, curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * batch_size, new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    
    # 5. Use packed latents directly (no conversion needed!)
    x_t = current_latents_packed.to(torch.float32).to(device)
    
    # 7. Create timestep tensor exactly like original Bagel
    timestep = torch.tensor([sample["timesteps"][0, timestep_idx]] * x_t.shape[0], device=device).to(torch.float32)
    
    # 8. Call _forward_flow exactly like original Bagel
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        velocity_pred_packed = bagel_model._forward_flow(
            x_t=x_t,
            timestep=timestep,
            packed_vae_token_indexes=generation_input['packed_vae_token_indexes'],
            packed_vae_position_ids=generation_input['packed_vae_position_ids'],
            packed_text_ids=generation_input['packed_text_ids'],
            packed_text_indexes=generation_input['packed_text_indexes'],
            packed_position_ids=generation_input['packed_position_ids'],
            packed_indexes=generation_input['packed_indexes'],
            packed_seqlens=generation_input['packed_seqlens'],
            key_values_lens=generation_input['key_values_lens'],
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input['packed_key_value_indexes'],
            policy_train=policy_train,
        )
        
    # 9. BAGEL-native SDE step (exactly like original Bagel)
    # Use the exact same dt calculation as original Bagel
    timesteps = sample["timesteps"][0]  # Get the full timestep sequence
    dts = timesteps[:-1] - timesteps[1:]  # Original Bagel's dt calculation
    dt = dts[timestep_idx] if timestep_idx < len(dts) else timesteps[timestep_idx]  # Use dts[i] like original Bagel
    
    prev_sample_packed, log_prob = bagel_sde_step_with_logprob_packed(
        v_t_packed=velocity_pred_packed,  # BAGEL's velocity prediction
        x_t_packed=x_t,  # Use x_t (current noisy state)
        dt=dt,  # BAGEL's timestep difference
        noise_level=config.sample.noise_level,
        batch_size=batch_size,
        generator=None,
    )
    
    # 10. Compute mean prediction for training (keep in packed format)
    prev_sample_mean_packed = x_t - velocity_pred_packed * dt
    
    # Compute std_dev_t for training (match SD3 format)
    std_dev_t = config.sample.noise_level * torch.sqrt(torch.abs(dt))
    
    # 11. Return packed format for maximum efficiency (no unpacking needed)
    return log_prob, prev_sample_mean_packed, std_dev_t
    



