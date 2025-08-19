#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import torch
from typing import List, Optional, Union

def encode_prompt_bagel(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 77,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids=None,
    do_classifier_free_guidance: bool = False,
):
    """
    Encode prompt for Bagel model.
    Bagel typically uses a single text encoder (CLIP-based).
    
    Args:
        text_encoder: Bagel text encoder
        tokenizer: Bagel tokenizer
        prompt: Text prompt(s) to encode
        max_sequence_length: Maximum sequence length
        device: Device to place tensors on
        num_images_per_prompt: Number of images per prompt
        text_input_ids: Pre-computed input IDs (optional)
        do_classifier_free_guidance: Whether to prepare for classifier-free guidance
    
    Returns:
        prompt_embeds: Encoded prompt embeddings
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    # Encode with text encoder
    if hasattr(text_encoder, '__call__'):
        # Standard text encoder call
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    else:
        # Handle different text encoder interfaces
        try:
            prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        except Exception as e:
            # Try alternative calling patterns
            try:
                prompt_embeds = text_encoder(text_input_ids.to(device))
                if isinstance(prompt_embeds, tuple):
                    prompt_embeds = prompt_embeds[0]
            except Exception as e2:
                raise ValueError(f"Failed to encode prompts with text encoder: {e}, {e2}")

    dtype = text_encoder.dtype if hasattr(text_encoder, 'dtype') else prompt_embeds.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # Duplicate text embeddings for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    # Handle classifier-free guidance
    if do_classifier_free_guidance:
        # Create uncond embeddings for classifier-free guidance
        uncond_tokens = [""] * batch_size
        uncond_inputs = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
        uncond_embeds = uncond_embeds.to(dtype=dtype, device=device)
        uncond_embeds = uncond_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_embeds = uncond_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        
        # Concatenate uncond and text embeddings
        prompt_embeds = torch.cat([uncond_embeds, prompt_embeds])

    return prompt_embeds

def compute_text_embeddings_bagel(
    prompt: Union[str, List[str]], 
    text_encoder, 
    tokenizer, 
    max_sequence_length: int, 
    device,
    do_classifier_free_guidance: bool = False,
):
    """
    Compute text embeddings for Bagel model.
    
    Args:
        prompt: Text prompt(s)
        text_encoder: Bagel text encoder
        tokenizer: Bagel tokenizer
        max_sequence_length: Maximum sequence length
        device: Device to place tensors on
        do_classifier_free_guidance: Whether to prepare for classifier-free guidance
    
    Returns:
        prompt_embeds: Computed text embeddings
    """
    with torch.no_grad():
        prompt_embeds = encode_prompt_bagel(
            text_encoder, 
            tokenizer, 
            prompt, 
            max_sequence_length, 
            device,
            do_classifier_free_guidance=do_classifier_free_guidance
        )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds

def encode_prompt_bagel_batch(
    text_encoder,
    tokenizer,
    prompts: List[str],
    max_sequence_length: int = 77,
    device=None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
):
    """
    Encode a batch of prompts for Bagel model.
    
    Args:
        text_encoder: Bagel text encoder
        tokenizer: Bagel tokenizer
        prompts: List of text prompts
        max_sequence_length: Maximum sequence length
        device: Device to place tensors on
        num_images_per_prompt: Number of images per prompt
        do_classifier_free_guidance: Whether to prepare for classifier-free guidance
    
    Returns:
        prompt_embeds: Batch of encoded prompt embeddings
    """
    batch_size = len(prompts)
    
    # Tokenize all prompts
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    # Encode with text encoder
    prompt_embeds = text_encoder(text_input_ids)[0]
    
    dtype = text_encoder.dtype if hasattr(text_encoder, 'dtype') else prompt_embeds.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    _, seq_len, _ = prompt_embeds.shape
    
    # Duplicate for multiple images per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
    # Handle classifier-free guidance
    if do_classifier_free_guidance:
        # Create uncond embeddings
        uncond_tokens = [""] * batch_size
        uncond_inputs = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
        uncond_embeds = uncond_embeds.to(dtype=dtype, device=device)
        uncond_embeds = uncond_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_embeds = uncond_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        
        # Concatenate uncond and text embeddings
        prompt_embeds = torch.cat([uncond_embeds, prompt_embeds])
    
    return prompt_embeds

def compute_text_embeddings(
    prompt: Union[str, List[str]],
    text_encoders,
    tokenizers,
    max_sequence_length: int = 77,
    device=None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
):
    """
    Compute text embeddings for Bagel model (compatible with Flow-GRPO training script).
    This function matches the interface expected by the training script.
    """
    # For Bagel, we only have one text encoder and tokenizer
    text_encoder = text_encoders[0] if isinstance(text_encoders, (list, tuple)) else text_encoders
    tokenizer = tokenizers[0] if isinstance(tokenizers, (list, tuple)) else tokenizers
    
    return compute_text_embeddings_bagel(
        prompt=prompt,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        device=device,
        do_classifier_free_guidance=do_classifier_free_guidance,
    ), None  # Return None for pooled_embeds to match SD3 interface
