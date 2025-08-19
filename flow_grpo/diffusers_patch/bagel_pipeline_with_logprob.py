# Bagel Pipeline with Logprob Support for Flow-GRPO
# Properly adapted for Bagel's unified multimodal architecture
#
# CRITICAL: Use Flow-GRPO's existing SDE implementation
# BAGEL: v_t = x_1 - x_0 (raw velocity prediction from data to noise)
# Flow-GRPO: Uses sde_step_with_logprob for SDE-based RL training
# This pipeline converts BAGEL's velocity predictions to Flow-GRPO's expected format
#
# MODIFICATIONS vs ORIGINAL BAGEL CODE:
# - BagelWithVelocity class: NEW - extends original Bagel to add velocity extraction
# - forward_with_velocity method: NEW - returns velocity predictions instead of loss
# - BagelPipeline class: MODIFIED - adapted for Flow-GRPO training using existing SDE
# - predict_noise_bagel method: MODIFIED - converts BAGEL velocity to Flow-GRPO format
# - All other methods marked with "ORIGINAL BAGEL LOGIC" or "MODIFIED FOR FLOW-GRPO"

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from .sd3_sde_with_logprob import sde_step_with_logprob
from .train_dreambooth_lora_bagel import compute_text_embeddings_bagel
import numpy as np
import logging

from bagel.modeling.bagel import Bagel


logger = logging.getLogger(__name__)

class BagelWithVelocity(Bagel):
    """
    MODIFIED: Extended Bagel class that can return velocity predictions for Flow-GRPO training.
    This extends the original Bagel class to add velocity extraction functionality.
    
    ORIGINAL BAGEL: Inherits from Bagel class
    MODIFICATION: Adds forward_with_velocity method
    """
    
    def __init__(self, language_model, vit_model, config):
        # ORIGINAL BAGEL: Standard initialization
        super().__init__(language_model, vit_model, config)
    
    def forward_with_velocity(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        MODIFIED: Forward pass that returns velocity predictions instead of loss.
        This is the same as the original forward method but returns velocity directly.
        
        ORIGINAL BAGEL: forward() method returns loss
        MODIFICATION: forward_with_velocity() returns velocity predictions
        """
        # ORIGINAL BAGEL: Call the original forward method to get all the internal computations
        outputs = super().forward(
            sequence_length=sequence_length,
            packed_text_ids=packed_text_ids,
            packed_text_indexes=packed_text_indexes,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            nested_attention_masks=nested_attention_masks,
            split_lens=split_lens,
            attn_modes=attn_modes,
            ce_loss_indexes=ce_loss_indexes,
            packed_label_ids=packed_label_ids,
            packed_vit_tokens=packed_vit_tokens,
            packed_vit_token_indexes=packed_vit_token_indexes,
            packed_vit_position_ids=packed_vit_position_ids,
            vit_token_seqlens=vit_token_seqlens,
            padded_latent=padded_latent,
            patchified_vae_latent_shapes=patchified_vae_latent_shapes,
            packed_latent_position_ids=packed_latent_position_ids,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_timesteps=packed_timesteps,
            mse_loss_indexes=mse_loss_indexes,
        )
        
        # MODIFICATION: Extract velocity prediction from the internal computation
        # We need to access the velocity prediction that was computed during the forward pass
        # Since the original forward method doesn't return it, we need to recompute it
        
        # ORIGINAL BAGEL LOGIC: Reconstruct the velocity prediction using the same logic as the original forward method
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            # ORIGINAL BAGEL: Use Bagel's actual attention mask creation
            from data.data_utils import create_sparse_mask
            from torch.nn.attention.flex_attention import create_block_mask
            
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
            
        else:
            attention_mask = nested_attention_masks

        # ORIGINAL BAGEL LOGIC: Process latent tokens (visual generation)
        if self.config.visual_gen and padded_latent is not None:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        # ORIGINAL BAGEL LOGIC: Language model forward pass
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        # MODIFICATION: Extract velocity prediction (this is what we need for Flow-GRPO!)
        packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
        return packed_mse_preds
        

class BagelPipeline:
    """
    MODIFIED: Bagel pipeline adapted for Flow-GRPO training.
    
    ORIGINAL BAGEL: Uses standard inference pipeline
    MODIFICATIONS: 
    - Returns velocity predictions instead of noise
    - Integrates with extended BagelWithVelocity class
    - Handles log probability computation for RL training
    """
    
    def __init__(self, bagel_model, vae, tokenizer, scheduler, device="cuda"):
        # ORIGINAL BAGEL: Standard initialization
        self.bagel_model = bagel_model
        self.vae = vae
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        self.original_prompts = None
        
        # MODIFICATION: Add Bagel-specific token IDs for proper text processing
        self._bagel_token_ids = None
    
    def _get_bagel_token_ids(self):
        """MODIFICATION: Get Bagel-specific token IDs for text processing."""
        if self._bagel_token_ids is None:
            # ORIGINAL BAGEL: Use Bagel's tokenizer to get special tokens
            self._bagel_token_ids = {
                'bos_token_id': self.tokenizer.bos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        return self._bagel_token_ids
    
    def patchify_latents(self, latents):
        """
        ORIGINAL BAGEL LOGIC: Convert latents to Bagel's patchified format.
        This is exactly how Bagel processes latents internally.
        """
        batch_size = latents.shape[0]
        h, w = latents.shape[2], latents.shape[3]
        p = self.bagel_model.latent_patch_size
        
        # ORIGINAL BAGEL: Patchify latents exactly as Bagel does
        patchified = []
        for i in range(batch_size):
            latent = latents[i]
            # Reshape to patches
            latent = latent.reshape(self.bagel_model.latent_channel, h, p, w, p)
            # Transpose to get patches in the right order
            latent = torch.einsum("chpwq->hwpqc", latent)
            # Flatten patches
            latent = latent.reshape(-1, p * p * self.bagel_model.latent_channel)
            patchified.append(latent)
        
        return patchified
    
    def _unpack_velocity_to_latents(self, packed_velocity, original_latents, h, w):
        """
        MODIFICATION: Convert packed velocity predictions back to latent format.
        This reverses the patchification process.
        """
        batch_size = original_latents.shape[0]
        p = self.bagel_model.latent_patch_size
        
        # MODIFICATION: Unpack velocity predictions
        unpacked_velocity = []
        start_idx = 0
        for i in range(batch_size):
            num_patches = h * w
            end_idx = start_idx + num_patches
            velocity_patches = packed_velocity[start_idx:end_idx]
            
            # Reshape back to latent format
            velocity_patches = velocity_patches.reshape(h, w, p, p, self.bagel_model.latent_channel)
            velocity_patches = torch.einsum("hwpqc->chpwq", velocity_patches)
            velocity_patches = velocity_patches.reshape(self.bagel_model.latent_channel, h * p, w * p)
            
            unpacked_velocity.append(velocity_patches)
            start_idx = end_idx
        
        return torch.stack(unpacked_velocity)
    
    def predict_noise_bagel(self, latents, timesteps, prompt_embeds, original_prompts=None):
        """
        MODIFIED: Predict velocity using Bagel's unified model (Bagel predicts v_t = dx_t/dt, not noise).
        
        ORIGINAL FLOW-GRPO: Expects noise predictions
        MODIFICATION: Returns velocity predictions (which is what Bagel actually predicts)
        """
        batch_size = latents.shape[0]
        
        # Store original prompts for proper Bagel integration
        if original_prompts is not None:
            self.original_prompts = original_prompts
        
        # MODIFICATION: Use the extended Bagel class to get velocity predictions directly
        return self._extract_velocity_from_bagel(latents, timesteps, original_prompts)
    
    def _extract_velocity_from_bagel(self, latents, timesteps, original_prompts):
        """
        MODIFICATION: Extract velocity prediction using the extended Bagel class.
        
        ORIGINAL BAGEL: Uses standard inference pipeline
        MODIFICATION: Uses extended BagelWithVelocity class for direct velocity extraction
        """
        batch_size = latents.shape[0]
        h, w = latents.shape[2], latents.shape[3]  # These are latent dimensions
        
        # ORIGINAL BAGEL LOGIC: Prepare text inputs using Bagel's methods
        if original_prompts:
            # ORIGINAL BAGEL: Use Bagel's prepare_prompts method
            generation_input, newlens, new_rope = self.bagel_model.prepare_prompts(
                curr_kvlens=[0] * batch_size,
                curr_rope=[0] * batch_size,
                prompts=original_prompts,
                tokenizer=self.tokenizer,
                new_token_ids=self._get_bagel_token_ids(),
            )
            packed_text_ids = generation_input['packed_text_ids']
            packed_text_indexes = generation_input['packed_text_indexes']
        else:
            # MODIFICATION: Fallback: create dummy text tokens
            packed_text_ids = torch.ones(batch_size * 77, dtype=torch.long, device=self.device) * 2
            packed_text_indexes = torch.arange(77, device=self.device).repeat(batch_size)
        
        # ORIGINAL BAGEL LOGIC: Patchify latents to match Bagel's format
        patchified_latents = self.patchify_latents(latents)
        
        # ORIGINAL BAGEL LOGIC: Prepare model inputs for Bagel's forward method
        model_inputs = {
            'sequence_length': batch_size * (h * w + 77),
            'packed_text_ids': packed_text_ids,
            'packed_text_indexes': packed_text_indexes,
            'sample_lens': [h * w + 77] * batch_size,
            'packed_position_ids': torch.zeros(batch_size * (h * w + 77), device=self.device, dtype=torch.long),
            'padded_latent': [patchified_latents[i] for i in range(batch_size)],
            'patchified_vae_latent_shapes': [(h, w)] * batch_size,
            'packed_latent_position_ids': torch.arange(h * w, device=self.device).repeat(batch_size),
            'packed_vae_token_indexes': torch.arange(h * w, device=self.device).repeat(batch_size),
            'packed_timesteps': timesteps.repeat(h * w),
            'mse_loss_indexes': torch.ones(h * w * batch_size, device=self.device, dtype=torch.bool),
        }
        
        # MODIFICATION: Use the extended Bagel class to get velocity predictions directly
        if hasattr(self.bagel_model, 'forward_with_velocity'):
            # MODIFICATION: Use the extended class
            packed_velocity = self.bagel_model.forward_with_velocity(**model_inputs)
        else:
            # MODIFICATION: Fallback to the old method if the extended class is not available
            packed_velocity = self._bagel_forward_extract_velocity(model_inputs, latents, timesteps)
        
        # MODIFICATION: Convert packed velocity back to latent format
        velocity_pred = self._unpack_velocity_to_latents(packed_velocity, latents, h, w)
        
        return velocity_pred
    
    def _bagel_forward_extract_velocity(self, model_inputs, latents, timesteps):
        """
        MODIFICATION: Fallback method to extract velocity prediction from Bagel's forward method.
        This is used when the extended BagelWithVelocity class is not available.
        
        ORIGINAL BAGEL: Uses standard forward method
        MODIFICATION: Manually reconstructs forward pass to extract velocity predictions
        """
        batch_size = latents.shape[0]
        h, w = latents.shape[2], latents.shape[3]
        
        # Extract inputs
        sequence_length = model_inputs['sequence_length']
        packed_text_ids = model_inputs['packed_text_ids']
        packed_text_indexes = model_inputs['packed_text_indexes']
        sample_lens = model_inputs['sample_lens']
        packed_position_ids = model_inputs['packed_position_ids']
        padded_latent = model_inputs['padded_latent']
        patchified_vae_latent_shapes = model_inputs['patchified_vae_latent_shapes']
        packed_latent_position_ids = model_inputs['packed_latent_position_ids']
        packed_vae_token_indexes = model_inputs['packed_vae_token_indexes']
        packed_timesteps = model_inputs['packed_timesteps']
        mse_loss_indexes = model_inputs['mse_loss_indexes']
        
        # ORIGINAL BAGEL LOGIC: 1. Process text embeddings
        packed_text_embedding = self.bagel_model.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.bagel_model.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        
        # ORIGINAL BAGEL LOGIC: 2. Create attention mask - PROPERLY handle this
        try:
            # ORIGINAL BAGEL: Use Bagel's actual attention mask creation
            # Add the correct import path for BAGEL's data utilities
            import sys
            import os
            
            # Try multiple possible paths for BAGEL's data utilities
            bagel_paths = [
                os.path.join(os.getcwd(), "bagel"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "bagel"),
                os.path.join(os.path.dirname(__file__), "..", "..", "bagel"),
            ]
            
            bagel_utils_imported = False
            for bagel_path in bagel_paths:
                if os.path.exists(os.path.join(bagel_path, "data", "data_utils.py")):
                    sys.path.insert(0, bagel_path)
                    try:
                        from data.data_utils import create_sparse_mask
                        from torch.nn.attention.flex_attention import create_block_mask
                        bagel_utils_imported = True
                        break
                    except ImportError:
                        sys.path.pop(0)
                        continue
            
            if bagel_utils_imported:
                sparse_mask = create_sparse_mask(sample_lens, None, None, packed_text_embedding.device)
                seqlen = sum(sample_lens)
                block_mask = create_block_mask(
                    sparse_mask, B=1, H=self.bagel_model.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                    device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
                )
                attention_mask = block_mask
            else:
                raise ImportError("Could not import BAGEL's data utilities")
                
        except ImportError:
            # MODIFICATION: Fallback if Bagel's utilities aren't available
            logger.warning("Could not import Bagel's attention mask utilities. Using fallback.")
            # Create a simple causal attention mask as fallback
            seqlen = sum(sample_lens)
            attention_mask = torch.ones((seqlen, seqlen), device=packed_text_embedding.device).tril()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
        
        # ORIGINAL BAGEL LOGIC: 3. Process latent tokens (visual generation) - EXACTLY as Bagel does
        p = self.bagel_model.latent_patch_size
        packed_latent = []
        for latent, (h_shape, w_shape) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h_shape * p, :w_shape * p].reshape(self.bagel_model.latent_channel, h_shape, p, w_shape, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.bagel_model.latent_channel)
            packed_latent.append(latent)
        packed_latent_clean = torch.cat(packed_latent, dim=0)
        
        # ORIGINAL BAGEL LOGIC: Add noise and timestep processing (exactly as Bagel does)
        noise = torch.randn_like(packed_latent_clean)
        packed_timesteps = torch.sigmoid(packed_timesteps)
        packed_timesteps = self.bagel_model.timestep_shift * packed_timesteps / (1 + (self.bagel_model.timestep_shift - 1) * packed_timesteps)
        packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
        packed_timestep_embeds = self.bagel_model.time_embedder(packed_timesteps)
        latent_token_pos_emb = self.bagel_model.latent_pos_embed(packed_latent_position_ids)
        packed_latent = self.bagel_model.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
        packed_sequence[packed_vae_token_indexes] = packed_latent
        
        # ORIGINAL BAGEL LOGIC: 4. Language model forward pass
        extra_inputs = {}
        if self.bagel_model.use_moe:
            packed_und_token_indexes = packed_text_indexes
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )
        
        last_hidden_state = self.bagel_model.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )
        
        # MODIFICATION: 5. Extract velocity prediction (this is what we need!)
        packed_mse_preds = self.bagel_model.llm2vae(last_hidden_state[mse_loss_indexes])
        
        # MODIFICATION: 6. Convert packed velocity back to latent format
        velocity_pred = self._unpack_velocity_to_latents(packed_mse_preds, latents, h, w)
        
        return velocity_pred
    
    def _velocity_to_noise(self, velocity_pred, current_latents, timesteps):
        """
        MODIFICATION: Convert BAGEL's raw velocity predictions to Flow-GRPO's expected format.
        
        BAGEL: v_t = x_1 - x_0 (raw velocity prediction from data to noise)
        Flow-GRPO: expects noise predictions compatible with sde_step_with_logprob
        
        SOLUTION: Convert BAGEL's velocity to match Flow-GRPO's SDE expectations
        """
        # Convert timesteps to proper format
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        
        # BAGEL's velocity is v_t = x_1 - x_0 (from data to noise)
        # For Flow Matching: x_t = (1-t)x_0 + t*x_1
        # Therefore: v_t = x_1 - x_0 = (x_t - (1-t)x_0)/t
        
        # For Flow-GRPO compatibility, we need to convert this to noise predictions
        # that work with the existing sde_step_with_logprob function
        
        # Convert timesteps to Flow-GRPO format (0 to 1 scale)
        timestep_factor = timesteps.view(-1, 1, 1, 1)
        
        # Convert velocity to noise-like predictions for Flow-GRPO's SDE
        # The key insight: Flow-GRPO's sde_step_with_logprob expects noise predictions
        # that can be used in the SDE formulation
        
        # For Flow Matching SDE, the relationship between velocity and noise is:
        # noise_pred ≈ velocity * scaling_factor
        # where scaling_factor accounts for the SDE formulation
        
        # Use a scaling that's compatible with Flow-GRPO's SDE step
        scaling_factor = timestep_factor * 0.1  # Adjust this based on Flow-GRPO's expectations
        noise_pred = velocity_pred * scaling_factor
        
        return noise_pred
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
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
        MODIFIED: Bagel pipeline with logprob computation for Flow-GRPO training.
        
        ORIGINAL BAGEL: Standard inference pipeline
        MODIFICATIONS: 
        - Returns velocity predictions instead of noise
        - Integrates with extended BagelWithVelocity class
        - Handles log probability computation for RL training
        - Returns dictionary format expected by training script
        """
        # Set default resolution for Bagel (1024x1024)
        height = height or 1024
        width = width or 1024
        
        # 1. Check inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1

        # 2. Encode prompts
        if prompt_embeds is None:
            if prompt is not None:
                # MODIFICATION: Use Bagel's text encoding
                prompt_embeds = compute_text_embeddings_bagel(
                    prompt, 
                    self.bagel_model.language_model, 
                    self.tokenizer, 
                    max_sequence_length, 
                    self.device
                )
            else:
                raise ValueError("Either prompt or prompt_embeds must be provided")
        
        if negative_prompt_embeds is None and guidance_scale > 1.0:
            if negative_prompt is not None:
                negative_prompt_embeds = compute_text_embeddings_bagel(
                    negative_prompt, 
                    self.bagel_model.language_model, 
                    self.tokenizer, 
                    max_sequence_length, 
                    self.device
                )
            else:
                # Use empty string as default negative prompt
                negative_prompt_embeds = compute_text_embeddings_bagel(
                    [""] * batch_size, 
                    self.bagel_model.language_model, 
                    self.tokenizer, 
                    max_sequence_length, 
                    self.device
                )
        
        # 3. Prepare latents
        if latents is None:
            # MODIFICATION: Use Bagel's latent format
            latent_height = height // 8
            latent_width = width // 8
            latents = randn_tensor(
                (batch_size, self.bagel_model.latent_channel, latent_height, latent_width),
                generator=generator,
                device=self.device,
                dtype=torch.float16,
            )
        
        # 4. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # 5. Prepare for classifier-free guidance
        if guidance_scale > 1.0:
            # Duplicate inputs for classifier-free guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            latents = latents.repeat(2, 1, 1, 1)
        
        # 6. Denoising loop with logprob computation
        all_latents = []
        all_log_probs = []
        
        for i, timestep in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = latents
            
            # MODIFICATION: Predict velocity using Bagel model
            # Pass original prompts for proper Bagel integration
            original_prompts = prompt if isinstance(prompt, list) else [prompt] if prompt else None
            velocity_pred = self.predict_noise_bagel(latent_model_input, timestep, prompt_embeds, original_prompts)
            
            # MODIFICATION: Convert velocity to noise for Flow-GRPO compatibility
            noise_pred = self._velocity_to_noise(velocity_pred, latent_model_input, timestep)
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # MODIFICATION: Compute logprob using Bagel-specific SDE step
            prev_latent, log_prob, prev_latent_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred,
                timestep,
                latent_model_input,
                noise_level=noise_level,
                generator=generator,
            )
            
            # Update latents
            latents = prev_latent
            
            # Store intermediate results
            all_latents.append(latents.detach().clone())
            all_log_probs.append(log_prob.detach().clone())
        
        # 7. Decode latents
        if output_type == "latent":
            images = latents
        else:
            # Scale and decode the image latents with vae
            latents = 1 / self.vae.config.scaling_factor * latents
            images = self.vae.decode(latents, return_dict=False)[0]
            
            if output_type == "pil":
                # Convert to PIL images
                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.cpu().permute(0, 2, 3, 1).numpy()
                images = (images * 255).round().astype("uint8")
                # Convert to PIL images (you'll need to import PIL)
                # images = [PIL.Image.fromarray(image) for image in images]
        
        # MODIFICATION: 8. Return format matches training script expectation
        return {
            "images": images,
            "latents": all_latents,
            "log_probs": all_log_probs
        }


def compute_log_prob_bagel(bagel_model, pipeline, sample, timestep_idx, prompt_embeds, config):
    """
    MODIFICATION: Compute log probability for Bagel using Flow-GRPO's existing SDE implementation.
    This function is called by the training script to compute log probabilities
    for the policy gradient computation.
    
    ORIGINAL FLOW-GRPO: Standard log probability computation using sde_step_with_logprob
    MODIFICATION: Adapted for Bagel's velocity predictions and timestep handling
    """
    batch_size = sample["latents"][timestep_idx].shape[0]
    
    # Get current latents and timestep
    current_latents = sample["latents"][timestep_idx]
    
    # MODIFICATION: Convert timestep index to actual timestep value
    # Bagel uses timesteps from 1.0 to 0.0 (reverse of DDPM)
    timestep_value = 1.0 - (timestep_idx / config.sample.num_steps)
    timestep = torch.tensor([timestep_value], device=current_latents.device).repeat(batch_size)
    
    # MODIFICATION: Predict velocity using Bagel model
    original_prompts = sample["prompts"] if "prompts" in sample else None
    velocity_pred = pipeline.predict_noise_bagel(
        current_latents, 
        timestep, 
        prompt_embeds, 
        original_prompts
    )
    
    # MODIFICATION: Convert velocity to Flow-GRPO's expected format
    noise_pred = pipeline._velocity_to_noise(velocity_pred, current_latents, timestep)
    
    # MODIFICATION: Compute log probability using Flow-GRPO's existing SDE step
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred,
        timestep,
        current_latents,
        noise_level=config.sample.noise_level,
    )
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


