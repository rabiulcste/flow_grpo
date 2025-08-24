#!/usr/bin/env python
"""
Bagel Training Script for Flow-GRPO

This script provides a clean implementation of Bagel training with Flow-GRPO,
using Bagel's native methods and Flow-GRPO's policy optimization.

CRITICAL FIX (v1.1): LoRA is now applied to bagel_model BEFORE creating the pipeline,
ensuring both training and sampling use the updated policy. This is simpler and cleaner
than modifying the pipeline after creation.
"""

from collections import defaultdict
import os
import datetime
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
import torch
import wandb
from functools import partial
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import Dataset, DataLoader
from flow_grpo.ema import EMAModuleWrapper
import numpy as np

from flow_grpo.diffusers_patch.bagel_pipeline_with_logprob import BagelPipeline, compute_log_prob_bagel_native

import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

tqdm = partial(tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/bagel.py", "Training configuration.")

logger = get_logger(__name__)

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Calculate zero standard deviation ratio for reward statistics."""
    prompt_to_rewards = defaultdict(list)
    for prompt, reward in zip(prompts, gathered_rewards['avg']):
        prompt_to_rewards[prompt].append(reward)
    
    zero_std_count = 0
    total_rewards = []
    for prompt, rewards in prompt_to_rewards.items():
        if len(rewards) > 1:
            std = np.std(rewards)
            if std == 0:
                zero_std_count += 1
        total_rewards.extend(rewards)
    
    zero_std_ratio = zero_std_count / len(prompt_to_rewards) if prompt_to_rewards else 0
    reward_std_mean = np.std(total_rewards) if total_rewards else 0
    
    return zero_std_ratio, reward_std_mean

class TextPromptDataset(Dataset):
    """Dataset for text prompts from .txt files."""
    
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(self.prompts) == 0:
            raise ValueError(f"No valid prompts found in {self.file_path}")
        
        logger.info(f"Loaded {len(self.prompts)} prompts from {self.file_path}")
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    """Dataset for Geneval prompts from JSONL files."""
    
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f if line.strip()]
            self.prompts = [item['prompt'] for item in self.metadatas if 'prompt' in item]
        
        if len(self.prompts) == 0:
            raise ValueError(f"No valid prompts found in {self.file_path}")
        
        logger.info(f"Loaded {len(self.prompts)} prompts from {self.file_path}")
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

def load_bagel_model(config):
    """Load Bagel model for training."""
    # Import Bagel components
    from bagel.modeling.bagel import Bagel, BagelConfig, Qwen2ForCausalLM, Qwen2Config, SiglipVisionModel, SiglipVisionConfig
    from bagel.modeling.autoencoder import load_ae
    from bagel.modeling.qwen2 import Qwen2Tokenizer
    from bagel.data.data_utils import add_special_tokens
    
    # Load model path from config
    model_path = os.environ["SAVE_DIR"] + "/" + config.pretrained.model
    logger.info(f"Loading BAGEL model from: {model_path}")
    
    # Load configurations
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # Create language model
    language_model = Qwen2ForCausalLM(llm_config)

    # Setup for visual generation only (no visual understanding for now)
    visual_und = False
    visual_gen = True
    
    vit_model = None
    vit_config = None
    vae_model = None
    vae_config = None

    if visual_gen:
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Create BagelConfig
    bagel_config = BagelConfig(
        visual_gen=visual_gen,
        visual_und=visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if visual_und else None,
        vae_config=vae_config if visual_gen else None,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        interpolate_pos=False,
        timestep_shift=config.bagel_timestep_shift,
    )
    
    # Setup tokenizer exactly like official Bagel training (pretrain_unified_navit.py)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    logger.info(f"Number of new tokens added: {num_new_tokens}")
    logger.info(f"Language model config vocab_size: {language_model.config.vocab_size}")
    logger.info(f"Language model embedding weight shape: {language_model.model.embed_tokens.weight.shape}")
    logger.info(f"Tokenizer vocab size after adding special tokens: {len(tokenizer)}")

    
    # Create Bagel model FIRST (before resizing)
    bagel_model = Bagel(language_model, vit_model if visual_und else None, bagel_config)
    
    # Load weights FIRST using Bagel's native approach
    from safetensors.torch import load_file
    model_state_dict_path = os.path.join(model_path, "ema.safetensors")
    
    logger.info(f"Loading model weights from: {model_state_dict_path}")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    
    msg = bagel_model.load_state_dict(model_state_dict, strict=False)
    logger.info(f"Model loading message: {msg}")
    
    # Resize embeddings AFTER loading weights (if needed)
    if num_new_tokens > 0:
        bagel_model.language_model.resize_token_embeddings(len(tokenizer))
        bagel_model.config.llm_config.vocab_size = len(tokenizer)
        bagel_model.language_model.config.vocab_size = len(tokenizer)
    
    logger.info(f"Final tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Final model config vocab size: {bagel_model.language_model.config.vocab_size}")
    logger.info(f"Final embedding shape: {bagel_model.language_model.model.embed_tokens.weight.shape}")
    
    del model_state_dict
    
    logger.info("Bagel model loaded and verified successfully")

    return bagel_model, vae_model, tokenizer, new_token_ids
        

def create_bagel_pipeline(bagel_model, vae, tokenizer, new_token_ids, config):
    """Create Bagel pipeline for training."""

    logger.info("Creating Bagel pipeline...")
            
    # Create proper scheduler for Flow Matching (consistent with pipeline)        
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=config.sample.num_steps,  # Match Bagel's inference steps; this is the number of diffusion steps
    )
    
    # Create Bagel pipeline with proper scheduler - FLOW-GRPO style
    pipeline = BagelPipeline(
        bagel_model=bagel_model,
        vae=vae,
        tokenizer=tokenizer,
        scheduler=scheduler,
        new_token_ids=new_token_ids,  # Pass like original Bagel
    )
    
    # Configure scheduler with Bagel's native timestep range for consistency
    # This ensures compute_log_prob and pipeline use the same timestep interpretations
    pipeline._setup_bagel_compatible_scheduler(config.sample.num_steps, bagel_model.device)
    
    logger.info("Bagel pipeline created successfully")
    return pipeline
        
   

def setup_lora_on_bagel_model(config, bagel_model, accelerator):
    """Setup LoRA on the entire Bagel model after loading weights."""
    logger.info("Setting up LoRA configuration for entire Bagel model...")
    
    # Configure LoRA for the entire Bagel model (language + visual components)
    lora_config = LoraConfig(
        r=config.train.lora_r,
        lora_alpha=config.train.lora_alpha,
        target_modules=[
            # Language model attention and MLP modules
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
            "gate_proj", "up_proj", "down_proj",     # MLP projections
            # Visual components - target individual Linear layers
            "vae2llm", "llm2vae",                    # VAE-LLM connectors
            # "connector",                            # ViT components
        ],
        lora_dropout=config.train.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to the entire Bagel model
    bagel_model = get_peft_model(bagel_model, lora_config)
    bagel_model.print_trainable_parameters()
    
    logger.info("LoRA setup completed successfully")
    return bagel_model, lora_config
    


def collect_samples_bagel(
    pipeline,
    train_iter,
    config,
    accelerator,
    reward_fn,
    stat_tracker,
    epoch,
    global_step,
):
    """Collect samples for Bagel training using Flow-GRPO's proper pattern."""
    
    # Generate samples for the entire epoch
    samples = []
    prompts = []
    
    for i in tqdm(
        range(config.sample.num_batches_per_epoch),
        desc=f"Epoch {epoch}: sampling",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):

        # Get prompts from dataloader iterator
        prompts_batch, prompt_metadata = next(train_iter)
        
        # Generate images using Bagel's native pipeline
        with torch.no_grad():
            # CRITICAL: Memory cleanup and CUDA health check
            torch.cuda.empty_cache()  # Clear memory cache
            
            # VERIFICATION: Ensure we're using the updated model for sampling
            if i == 0 and epoch == 0 and accelerator.is_main_process:
                logger.info(f"🔍 Sampling from model with {sum(p.numel() for p in pipeline.bagel_model.parameters() if p.requires_grad)} trainable parameters")
            
            negative_prompts = [""] * len(prompts_batch) if config.train.cfg else None

            outputs = pipeline(
                prompt=prompts_batch,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                negative_prompt=negative_prompts if config.train.cfg else None,
                num_images_per_prompt=config.sample.num_image_per_prompt,
                output_type="pt",
                noise_level=config.sample.noise_level,
            )
        
        images = outputs["images"]
        latents = outputs["latents"]
        log_probs = outputs["log_probs"]
        
        # Validate pipeline output immediately
        expected_image_count = len(prompts_batch) * config.sample.num_image_per_prompt
        actual_image_count = images.shape[0] if hasattr(images, 'shape') else len(images)
        if actual_image_count != expected_image_count:
            raise RuntimeError(
                f"Bagel pipeline failed to generate correct number of images in batch {i}: "
                f"Expected {expected_image_count} images "
                f"(prompts={len(prompts_batch)} * images_per_prompt={config.sample.num_image_per_prompt}), "
                f"but pipeline generated {actual_image_count} images. "
                f"Images shape: {images.shape if hasattr(images, 'shape') else type(images)}. "
                f"This indicates a bug in the Bagel pipeline's num_images_per_prompt handling."
            )
        
        # Compute rewards (create metadata matching number of images - Flow-GRPO pattern)
        metadata = []
        for prompt in prompts_batch:
            for _ in range(config.sample.num_image_per_prompt):
                metadata.append({"prompt": prompt})
        rewards, reward_metadata = reward_fn(images, prompts_batch, metadata, only_strict=True)
        
        # Validate reward dimensions immediately after reward function call
        expected_reward_count = len(prompts_batch) * config.sample.num_image_per_prompt
        if 'avg' not in rewards:
            raise RuntimeError(
                f"Reward function returned invalid structure: missing 'avg' key. "
                f"Available keys: {list(rewards.keys())}"
            )
        actual_reward_count = len(rewards['avg'])
        if actual_reward_count != expected_reward_count:
            raise RuntimeError(
                f"Reward function dimension mismatch in batch {i}: "
                f"Expected {expected_reward_count} rewards "
                f"(prompts={len(prompts_batch)} * images_per_prompt={config.sample.num_image_per_prompt}), "
                f"but reward_fn returned {actual_reward_count} rewards. "
                f"Images shape: {images.shape if hasattr(images, 'shape') else type(images)}, "
                f"Metadata length: {len(metadata)}, "
                f"Reward keys: {list(rewards.keys())}"
            )
        
        # Convert rewards to tensors exactly like Flow-GRPO
        rewards_tensor = {
            key: torch.as_tensor(value, device=accelerator.device).float()
            for key, value in rewards.items()
        }
        
        # Get timesteps from Bagel's scheduler (use all timesteps)
        timesteps = pipeline.scheduler.timesteps.repeat(len(prompts_batch) * config.sample.num_image_per_prompt, 1)
        
        # Store results exactly like Flow-GRPO
        samples.append({
            "latents": latents,
            "log_probs": log_probs,
            "timesteps": timesteps,
            # "past_key_values": outputs["past_key_values"],  # Store cache for efficient training (Bagel naming)
            "prompts": prompts_batch,  # Keep prompts for stat tracking
            "rewards": rewards_tensor,
        })
        
        prompts.extend(prompts_batch)                
    
    if not samples:
        raise RuntimeError("No samples were successfully generated")
    
    # Compute advantages exactly like Flow-GRPO
    # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
    collated_samples = {}
    for k in samples[0].keys():
        if k == "past_key_values":
            # Handle past_key_values specially - just pass through (Bagel handles batching internally)
            collated_samples[k] = samples[0][k]  # Use first sample's cache for now
        elif isinstance(samples[0][k], dict):
            # Handle nested dictionaries (like rewards)
            collated_samples[k] = {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
        elif isinstance(samples[0][k], torch.Tensor):
            # Handle tensors
            collated_samples[k] = torch.cat([s[k] for s in samples], dim=0)
        elif k == "prompts":
            # Handle prompts specially - expand to match num_images_per_prompt
            collated_samples[k] = []
            for s in samples:
                for prompt in s[k]:
                    collated_samples[k].extend([prompt] * config.sample.num_image_per_prompt)
        elif isinstance(samples[0][k], list):
            # Handle other lists - flatten into single list
            collated_samples[k] = []
            for s in samples:
                collated_samples[k].extend(s[k])
        else:
            # Handle other types - just collect into list
            collated_samples[k] = [s[k] for s in samples]
    
    samples = collated_samples
    
    # Gather rewards across processes (multi-GPU compatible) - EXACTLY like original Flow-GRPO
    gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
    gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
    
    if config.per_prompt_stat_tracking and stat_tracker is not None:
        # gather the prompts across processes - exactly like Flow-GRPO
        # Prompts are already expanded in the collation step, so use them directly
        expanded_prompts = samples["prompts"]
        
        # Convert to indices for gathering (since we can't gather strings directly)
        unique_prompts = list(set(expanded_prompts))
        prompt_to_idx = {prompt: idx for idx, prompt in enumerate(unique_prompts)}
        prompt_indices = torch.tensor([prompt_to_idx[prompt] for prompt in expanded_prompts], device=accelerator.device)
        
        # Gather indices across processes (like Flow-GRPO gathers prompt_ids)
        gathered_prompt_indices = accelerator.gather(prompt_indices).cpu().numpy()
        
        # Convert back to prompts (like Flow-GRPO decodes prompt_ids to prompts)
        gathered_prompts = [unique_prompts[idx] for idx in gathered_prompt_indices]
        
        if len(gathered_prompts) != len(gathered_rewards['avg']):
            raise RuntimeError(
                f"Prompt-reward dimension mismatch: "
                f"gathered_prompts={len(gathered_prompts)}, gathered_rewards={len(gathered_rewards['avg'])}. "
                f"Expected: {len(samples['prompts'])} prompts * {config.sample.num_image_per_prompt} images_per_prompt * {accelerator.num_processes} processes = {len(samples['prompts']) * config.sample.num_image_per_prompt * accelerator.num_processes}"
            )
        
        advantages = stat_tracker.update(gathered_prompts, gathered_rewards['avg'])
    else:
        # Use global statistics (fallback)
        advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)
    
    # DEBUG: Check advantages computation
    if accelerator.is_main_process:
        print(f"gathered_rewards['avg'] value: {gathered_rewards['avg']}")
    
    # Ungather advantages; we only need to keep the entries corresponding to the samples on this process
    advantages = torch.as_tensor(advantages, device=accelerator.device).unsqueeze(-1)  # [12] -> [12, 1] for Flow-GRPO compatibility
    
    if accelerator.is_main_process:
        print(f"accelerator.num_processes: {accelerator.num_processes}")
        print(f"accelerator.process_index: {accelerator.process_index}")
    
    samples["advantages"] = (
        advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
        .to(accelerator.device)
    )
    
    if accelerator.is_main_process:
        print(f"final advantages shape: {samples['advantages'].shape}")
        print("=" * 40)
    
    # NOTE: Repeat advantages across all training timesteps
    # The training loop expects advantages to have shape [batch_size, num_train_timesteps]
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction) if hasattr(config.train, 'timestep_fraction') else config.sample.num_steps
    samples["advantages"] = samples["advantages"].repeat(1, num_train_timesteps)
    
    
    # Store original rewards and repeat rewards along timestep dimension like Flow-GRPO
    samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
    samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.num_steps)
    
    # Log stat tracker metrics if enabled (same place as original Flow-GRPO)
    if config.per_prompt_stat_tracking and stat_tracker is not None:
        group_size, trained_prompt_num = stat_tracker.get_stats()
        
        # Calculate zero std ratio using gathered rewards (same pattern as Flow-GRPO)
        zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
        
        if accelerator.is_main_process:
            wandb.log({
                "group_size": group_size,
                "trained_prompt_num": trained_prompt_num,
                "zero_std_ratio": zero_std_ratio,
                "reward_std_mean": reward_std_mean,
            }, step=global_step)
        
        stat_tracker.clear()
    
    return samples, gathered_rewards

def train_epoch_bagel(
    bagel_model,
    samples,
    config,
    accelerator,
    optimizer,
    num_train_timesteps,
    epoch_info,
    info,
    pipeline,  # Add pipeline for scheduler and tokenizer access
    global_step,  # Add global_step parameter
    ema=None,
    trainable_parameters=None,
):
    """Train one epoch for Bagel using Flow-GRPO's proper pattern."""
    
    # Clean up processed data before shuffling (follow Flow-GRPO pattern)
    del samples["rewards"]  # Rewards already processed into gathered_rewards
    # Keep prompts for stat tracking, past_key_values for efficient training
    
    # Convert samples to batched format for training
    total_batch_size = samples["latents"].shape[0]
    
    # Add assertion like Flux to catch timestep mismatches
    _, num_timesteps = samples["timesteps"].shape
    assert num_timesteps == config.sample.num_steps, f"Timestep mismatch: expected {config.sample.num_steps}, got {num_timesteps}"
    
    # NOTE: Skip shuffling for Bagel - past_key_values cache is tied to specific prompts
    # Shuffling would break prompt-embedding correspondence, causing incorrect training
    # TODO: Consider per-sample caches or cache reconstruction to enable shuffling
    
    # Safety check: validate tensor dimensions before training
    for k, v in samples.items():
        if isinstance(v, torch.Tensor) and v.shape[0] != total_batch_size:
            raise RuntimeError(
                f"Tensor dimension mismatch: "
                f"'{k}' has batch size {v.shape[0]}, expected {total_batch_size}. "
                f"Shape: {v.shape}. This indicates a bug in sample collection."
            )
    
    # Rebatch for training (no shuffling to preserve cache-data correspondence)
    samples_batched = {}
    for k, v in samples.items():
        if k == "past_key_values":
            # past_key_values is a NaiveCache object, don't reshape it
            # Just use the same cache for all batches (Bagel handles this internally)
            samples_batched[k] = [v] * config.sample.num_batches_per_epoch
        elif isinstance(v, torch.Tensor):
            # Reshape tensors without shuffling
            samples_batched[k] = v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
        elif isinstance(v, list):
            # Reshape lists without shuffling
            batch_size = len(v) // config.sample.num_batches_per_epoch
            samples_batched[k] = [v[i:i+batch_size] for i in range(0, len(v), batch_size)]
        else:
            # For other types, just repeat
            samples_batched[k] = [v] * config.sample.num_batches_per_epoch
    
    # dict of lists -> list of dicts for easier iteration
    samples_batched = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    
    # Set model to training mode (like SD3 script)
    bagel_model.train()
    
    for i, sample in tqdm(
        list(enumerate(samples_batched)),
        desc=f"Epoch {epoch_info}: training",
        position=0,
        disable=not accelerator.is_local_main_process,
    ):
        # Use step indices for training - follow Flux pattern
        train_timesteps = [step_index for step_index in range(num_train_timesteps)]
        
        for j in tqdm(
            train_timesteps,
            desc="Timestep",
            position=1,
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):
            with accelerator.accumulate(bagel_model):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32):
                    # Compute log probability using Bagel's native approach
                    unwrapped_model = bagel_model.module if hasattr(bagel_model, 'module') else bagel_model
                    prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob_bagel_native(
                        bagel_model=unwrapped_model,
                        sample=sample,
                        timestep_idx=j,
                        config=config,
                        tokenizer=pipeline.tokenizer,
                        new_token_ids=pipeline.new_token_ids,
                        scheduler=pipeline.scheduler,  # Pass the pipeline's scheduler for consistency
                        policy_train=True,
                    )
                    
                    if config.train.beta > 0:
                        with torch.no_grad():
                            # Reference computation for KL loss
                            # Disable adapter on the entire model
                            with unwrapped_model.disable_adapter():
                                _, _, prev_sample_mean_ref, _ = compute_log_prob_bagel_native(
                                    bagel_model=unwrapped_model,
                                    sample=sample,
                                    timestep_idx=j,
                                    config=config,
                                    tokenizer=pipeline.tokenizer,
                                    new_token_ids=pipeline.new_token_ids,
                                    scheduler=pipeline.scheduler,  # Pass the pipeline's scheduler for consistency
                                    policy_train=True,
                                )

                # Compute advantages and policy loss
                advantages = torch.clamp(
                    sample["advantages"][:, j],
                    -config.train.adv_clip_max,
                    config.train.adv_clip_max,
                )
                ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - config.train.clip_range,
                    1.0 + config.train.clip_range,
                )
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                
                # Add KL loss if enabled
                if config.train.beta > 0:
                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                    kl_loss = torch.mean(kl_loss)
                    loss = policy_loss + config.train.beta * kl_loss
                else:
                    loss = policy_loss

                # Update statistics
                info["approx_kl"].append(
                    0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                )
                info["clipfrac"].append(
                    torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float())
                )
                info["policy_loss"].append(policy_loss)
                if config.train.beta > 0:
                    info["kl_loss"].append(kl_loss)
                info["loss"].append(loss)

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(bagel_model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                if accelerator.is_main_process:
                    wandb.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)
                
        # Update EMA if enabled
        if config.train.ema and ema is not None:
            ema.step(trainable_parameters, global_step)
            

    return global_step, info

def evaluate_bagel(pipeline, test_dataloader, config, accelerator, reward_fn, global_step):
    """Evaluate Bagel model performance using Flow GRPO native pattern."""
   
    logger.info("Starting evaluation...")
    
    all_rewards = defaultdict(list)
    
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        
        # ORIGINAL BAGEL: Generate evaluation images using native pipeline
        with torch.no_grad():
            # Use num_steps if eval_num_steps is not set
            eval_steps = config.sample.eval_num_steps if hasattr(config.sample, 'eval_num_steps') and config.sample.eval_num_steps is not None else config.sample.num_steps
            
            outputs = pipeline(
                prompt=prompts,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=eval_steps,
                guidance_scale=config.sample.guidance_scale,
                num_images_per_prompt=config.sample.num_image_per_prompt,
                output_type="pt",
            )
        
        images = outputs["images"]
        
        # Validate pipeline output immediately
        expected_image_count = len(prompts) * config.sample.num_image_per_prompt
        actual_image_count = images.shape[0] if hasattr(images, 'shape') else len(images)
        if actual_image_count != expected_image_count:
            raise RuntimeError(
                f"Bagel pipeline failed to generate correct number of images: "
                f"Expected {expected_image_count} images "
                f"(prompts={len(prompts)} * images_per_prompt={config.sample.num_image_per_prompt}), "
                f"but pipeline generated {actual_image_count} images. "
                f"Images shape: {images.shape if hasattr(images, 'shape') else type(images)}. "
                f"This indicates a bug in the Bagel pipeline's num_images_per_prompt handling."
            )
        
        # Compute rewards using Flow GRPO native pattern (expand metadata to match images)
        eval_metadata = []
        for prompt in prompts:
            for _ in range(config.sample.num_image_per_prompt):
                # Use the original prompt_metadata if available, otherwise create simple metadata
                if prompt_metadata:
                    eval_metadata.append(prompt_metadata[prompts.index(prompt)])
                else:
                    eval_metadata.append({"prompt": prompt})
        rewards, reward_metadata = reward_fn(images, prompts, eval_metadata, only_strict=False)
        
        # Validate reward dimensions in evaluation
        expected_eval_reward_count = len(prompts) * config.sample.num_image_per_prompt
        if 'avg' not in rewards:
            raise RuntimeError(
                f"Evaluation reward function returned invalid structure: missing 'avg' key. "
                f"Available keys: {list(rewards.keys())}"
            )
        actual_eval_reward_count = len(rewards['avg'])
        if actual_eval_reward_count != expected_eval_reward_count:
            raise RuntimeError(
                f"Evaluation reward function dimension mismatch: "
                f"Expected {expected_eval_reward_count} rewards "
                f"(prompts={len(prompts)} * images_per_prompt={config.sample.num_image_per_prompt}), "
                f"but reward_fn returned {actual_eval_reward_count} rewards. "
                f"Images shape: {images.shape if hasattr(images, 'shape') else type(images)}, "
                f"Eval metadata length: {len(eval_metadata)}, "
                f"Reward keys: {list(rewards.keys())}"
            )
        
        
        # Collect rewards exactly like original Flow-GRPO
        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    # Concatenate rewards exactly like original Flow-GRPO
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    
    if accelerator.is_main_process:
        for key, value in all_rewards.items():
            if '_strict_accuracy' not in key and '_accuracy' not in key:
                wandb.log({f"eval_{key}": np.mean(value)}, step=global_step)
        
    
    logger.info("Evaluation completed successfully")
    

def save_ckpt(save_dir, bagel_model, global_step, accelerator, ema, trainable_parameters, config, optimizer=None, epoch=None):
    """Save model checkpoint."""
                
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    os.makedirs(save_root, exist_ok=True)
    
    # Save model
    model_dir = os.path.join(save_root, "lora" if config.use_lora else "")
    accelerator.unwrap_model(bagel_model).save_pretrained(model_dir)
    
    # Save optimizer
    if optimizer:
        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.bin"))
    
    # Save EMA
    if config.train.ema and ema:
        ema.copy_ema_to(trainable_parameters, store_temp=True)
        torch.save(ema.state_dict(), os.path.join(save_root, "ema.bin"))
        ema.copy_temp_to(trainable_parameters)
    
    # Save training state
    training_state = {
        "global_step": global_step,
        "epoch": epoch or 0,
        "config": config.to_dict(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(save_root, "training_state.json"), 'w') as f:
        json.dump(training_state, f, indent=2)
    
    logger.info(f"Checkpoint saved to {save_root}")

def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint in a run directory."""
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(checkpoint_dir, item)))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Return the checkpoint with highest step number
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    return latest_checkpoint[1]

def auto_discover_checkpoint(config):
    """Auto-discover checkpoint based on run name or resume setting."""
    if config.train.resume_from_checkpoint is True:
        # Auto-discover based on run name
        if config.run_name:
            run_dir = os.path.join(config.logdir, config.run_name)
            if os.path.exists(run_dir):
                latest_checkpoint = find_latest_checkpoint(run_dir)
                if latest_checkpoint:
                    logger.info(f"Auto-discovered checkpoint: {latest_checkpoint}")
                    return latest_checkpoint
                else:
                    logger.info(f"No checkpoints found in {run_dir}")
        return None
    
    elif config.train.resume_from_checkpoint:
        # If explicit path provided, use it
        if os.path.exists(config.train.resume_from_checkpoint):
            return config.train.resume_from_checkpoint
        else:
            logger.warning(f"Checkpoint path not found: {config.train.resume_from_checkpoint}")
            return None
    
    return None

def load_ckpt(checkpoint_path, bagel_model, optimizer, accelerator, ema, config):
    """Load model checkpoint for resuming training."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model
    model_dir = os.path.join(checkpoint_path, "lora" if config.use_lora else "")
    if os.path.exists(model_dir):
        if config.use_lora:
            from peft import PeftModel
            bagel_model = PeftModel.from_pretrained(bagel_model, model_dir)
            bagel_model.set_adapter("default")
        else:
            bagel_model = bagel_model.__class__.from_pretrained(model_dir)
        logger.info("Model loaded")
    
    # Load optimizer
    optimizer_path = os.path.join(checkpoint_path, "optimizer.bin")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=accelerator.device))
        logger.info("Optimizer loaded")
    
    # Load EMA
    if config.train.ema and ema:
        ema_path = os.path.join(checkpoint_path, "ema.bin")
        if os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location=accelerator.device))
            logger.info("EMA loaded")
    
    # Load training state
    training_state = None
    state_path = os.path.join(checkpoint_path, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            training_state = json.load(f)
        logger.info("Training state loaded")
    
    return bagel_model, optimizer, training_state

def main(_):
    """Main training function."""
    
    config = FLAGS.config
    
    # Setup unique run name
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = f"bagel_flow_grpo_{unique_id}"
    else:
        config.run_name += "_" + unique_id

    
    # Setup accelerator
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction) if hasattr(config.train, 'timestep_fraction') else config.sample.num_steps
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=os.path.join(config.logdir, config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=config.num_checkpoint_limit,
        ),
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    
    # Setup logging
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
            name=config.run_name,
            config=config.to_dict(),
            resume="allow" if config.train.resume_from_checkpoint else None,
        )
    
    # Auto-discover checkpoint
    checkpoint_path = auto_discover_checkpoint(config)
    
    set_seed(config.seed, device_specific=True)

    bagel_model, vae, tokenizer, new_token_ids = load_bagel_model(config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    bagel_model.to(accelerator.device, dtype=inference_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    
    # Setup LoRA BEFORE creating pipeline (simpler and cleaner approach)
    if config.use_lora: 
        bagel_model, lora_config = setup_lora_on_bagel_model(config, bagel_model, accelerator)
        bagel_model.to(accelerator.device, dtype=inference_dtype)
    
    # Create pipeline with the (potentially LoRA-updated) model
    pipeline = create_bagel_pipeline(bagel_model, vae, tokenizer, new_token_ids, config)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        bagel_model.parameters(),
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.train.adam_weight_decay,
    )
    
    # Setup EMA
    ema = EMAModuleWrapper() if config.train.ema else None
    trainable_parameters = [p for p in bagel_model.parameters() if p.requires_grad] if config.train.ema else None
    
    # Load checkpoint if resuming
    training_state = None
    if checkpoint_path:
        bagel_model, optimizer, training_state = load_ckpt(
            checkpoint_path, bagel_model, optimizer, accelerator, ema, config
        )
    
    # Setup datasets
    if config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    
    # Create dataloaders (Flow GRPO native pattern)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sample.train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=train_dataset.collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=2,
    )
    
    # Setup reward function
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    
    # Setup stat tracker
    stat_tracker = PerPromptStatTracker() if config.per_prompt_stat_tracking else None
    
    # Prepare for training
    bagel_model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        bagel_model, optimizer, train_dataloader, test_dataloader
    )
    logger.info("=== STARTING TRAINING ===")

    # Initialize training state
    global_step = training_state.get("global_step", 0) if training_state else 0
    epoch = training_state.get("epoch", 0) if training_state else 0
    info = defaultdict(list)
    
    remaining_steps = config.train.total_steps - global_step if training_state else config.train.total_steps
    logger.info(f"{'Resuming' if training_state else 'Starting'} training for {remaining_steps} steps")
    if training_state:
        logger.info(f"From step {global_step}, epoch {epoch}")
    
    while True:
        #################### EVAL ####################
        # bagel_model.eval()
        # if epoch % config.eval_freq == 0:
        #     evaluate_bagel(
        #         pipeline, test_dataloader, config, accelerator,
        #         eval_reward_fn, global_step
        #     )
                                
        # Save checkpoint
        if accelerator.is_main_process:
            if epoch % config.save_freq == 0 and epoch > 0:
                save_ckpt(config.logdir, bagel_model, global_step, accelerator, ema, trainable_parameters, config, optimizer, epoch)
            
        #################### SAMPLING ####################
        # Reset iterator for each epoch to avoid StopIteration
        train_iter = iter(train_dataloader)
        samples, gathered_rewards = collect_samples_bagel(
            pipeline, train_iter, config, accelerator,
            reward_fn, stat_tracker, epoch, global_step
        )
        
        # Log rewards
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )
        
        #################### TRAINING ####################
        # Calculate num_train_timesteps for this epoch
        num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction) if hasattr(config.train, 'timestep_fraction') else config.sample.num_steps
        
        for inner_epoch in range(config.train.num_inner_epochs):
            bagel_model.train()
            global_step, info = train_epoch_bagel(
                bagel_model, samples, config, accelerator,
                optimizer, num_train_timesteps, f"{epoch}.{inner_epoch}", info,
                pipeline, global_step, ema, trainable_parameters
            )
        
        epoch += 1
        
        # Check if we've reached the total steps
        if global_step >= config.train.total_steps:
            break
    
    logger.info("Training completed successfully!")
    


if __name__ == "__main__":
    app.run(main)
