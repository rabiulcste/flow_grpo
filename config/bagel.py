import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def bagel_base():
    """Base Bagel configuration for Flow-GRPO training."""
    config = base.get_config()
    
    # Bagel model settings
    config.pretrained.model = "models/BAGEL-7B-MoT" #Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT to models/BAGEL-7B-MoT
    config.resolution = 1024  # Bagel default resolution
    config.max_sequence_length = 77  # Bagel text encoder sequence length
    
    # Sampling settings optimized for Bagel
    config.sample.num_steps = 50  # Bagel default inference steps
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 7.5  # Bagel default guidance scale
    config.sample.noise_level = 0.7  # Noise level for logprob computation
    
    # Bagel-specific flow matching parameters
    config.bagel_timestep_shift = 3.0  # Bagel's default timestep_shift parameter
    
    # Training settings
    config.train.learning_rate = 1e-4
    config.train.beta = 0.0  # KL penalty coefficient
    config.train.clip_range = 1e-4  # PPO clip range
    config.train.adv_clip_max = 5  # Advantage clipping maximum
    config.train.max_grad_norm = 1.0
    
    # LoRA settings for memory efficiency
    config.use_lora = True
    config.train.lora_r = 16
    config.train.lora_alpha = 32
    config.train.lora_dropout = 0.1
    
    # Batch sizes optimized for Bagel (Flow GRPO style)
    # Total images per batch = train_batch_size × num_image_per_prompt = 4 × 4 = 16 images
    config.sample.train_batch_size = 4
    config.sample.test_batch_size = 8  
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 8
    
    config.train.batch_size = 4  # Must match sample.train_batch_size
    config.train.gradient_accumulation_steps = 4  # Effective batch size = 16
    
    # Training duration (optimized for Flow GRPO)
    config.train.total_steps = 2000  # More training steps for better convergence
    config.train.num_inner_epochs = 1
    config.save_freq = 50   # Less frequent saves for faster training
    config.eval_freq = 25   # More frequent evaluation
    
    # Classifier-free guidance
    config.train.cfg = True
    
    # Default dataset and prompt function
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.prompt_fn = "general_ocr"
    
    # Default reward function
    config.reward_fn = {"aesthetic": 1.0}
    
    # Per-prompt stat tracking
    config.per_prompt_stat_tracking = True
    
    return config

def bagel_1gpu():
    """Bagel configuration for 1 GPU training."""
    config = bagel_base()
    
    # Single GPU optimized settings (memory-conscious for 1024px)
    config.resolution = 256
    # Total images per training batch = train_batch_size × num_image_per_prompt = 8 × 2 = 16 images
    config.sample.train_batch_size = 4
    config.sample.test_batch_size = 4
    config.sample.num_image_per_prompt = 2
    config.sample.num_batches_per_epoch = 4
    
    config.train.batch_size = 2  # Must match sample.train_batch_size
    config.train.gradient_accumulation_steps = 4  # Effective batch size = 8
    
    # Smaller LoRA for memory efficiency
    config.train.lora_r = 8
    config.train.lora_alpha = 16
    
    # number of sampler inference steps for collecting dataset.
    config.sample.num_steps = 10
    # number of sampler inference steps for evaluation.
    config.sample.eval_num_steps = 10

    config.train.total_steps = 500
    config.mixed_precision = "bf16"

    config.train.resume_from_checkpoint = True

    # More frequent evaluation for single GPU development
    config.eval_freq = 5
    config.save_freq = 50
    
    config.save_dir = 'logs/bagel/1gpu'
    
    return config


def bagel_4gpu():
    """Bagel configuration for 4 GPU training."""
    config = bagel_base()
    
    # 4 GPU optimized settings
    config.resolution = 256
    # Total images per GPU = train_batch_size × num_image_per_prompt = 2 × 4 = 8 images
    # Total images across 4 GPUs = 8 × 4 = 32 images per batch
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 8
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 8
    
    # Memory optimization settings
    config.mixed_precision = "bf16"  # Better than fp16 for memory
    config.gradient_checkpointing = True  # Enable gradient checkpointing
    
    # Reduce inference steps for memory
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 15

    config.train.batch_size = 2  # Must match sample.train_batch_size
    config.train.gradient_accumulation_steps = 4  # Effective batch size = 8
    
    # Smaller LoRA for memory efficiency
    config.train.lora_r = 8
    config.train.lora_alpha = 16
    config.train.resume_from_checkpoint = True

    config.train.total_steps = 500
    
    
    # More frequent evaluation for single GPU development
    config.eval_freq = 5
    config.save_freq = 5
    
    config.save_dir = 'logs/bagel/4gpu'
    
    return config

def bagel_8gpu():
    """Bagel configuration for 8 GPU training."""
    config = bagel_base()
    
    # 8 GPU optimized settings  
    # Total images per GPU = train_batch_size × num_image_per_prompt = 2 × 3 = 6 images
    # Total images across 8 GPUs = 6 × 8 = 48 images per batch
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 8
    config.sample.num_image_per_prompt = 3
    config.sample.num_batches_per_epoch = 16
    
    config.train.batch_size = 2  # Must match sample.train_batch_size
    config.train.gradient_accumulation_steps = 4  # Effective batch size = 8
    
    # Larger LoRA for better performance
    config.train.lora_r = 32
    config.train.lora_alpha = 64
    
    config.save_dir = 'logs/bagel/8gpu'
    
    return config

def bagel_ocr():
    """Bagel configuration optimized for OCR training."""
    config = bagel_base()
    
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"ocr": 1.0}
    
    # OCR-specific settings
    config.sample.num_steps = 30  # Fewer steps for faster training
    config.sample.eval_num_steps = 50
    config.train.learning_rate = 5e-5  # Lower learning rate for OCR
    config.train.timestep_fraction = 1.0  # Train on all timesteps
    
    config.save_dir = 'logs/bagel/ocr'
    
    return config

def bagel_geneval():
    """Bagel configuration optimized for Geneval training."""
    config = bagel_base()
    
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.prompt_fn = "geneval"
    config.reward_fn = {"geneval": 1.0}
    
    # Geneval-specific settings
    config.sample.num_steps = 30
    config.sample.eval_num_steps = 50
    config.train.learning_rate = 5e-5
    config.train.timestep_fraction = 1.0  # Train on all timesteps
    
    config.save_dir = 'logs/bagel/geneval'
    
    return config

def bagel_aesthetic():
    """Bagel configuration optimized for aesthetic training."""
    config = bagel_base()
    
    config.dataset = os.path.join(os.getcwd(), "dataset/aesthetic")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"aesthetic": 1.0}
    
    # Aesthetic-specific settings
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 50
    config.train.learning_rate = 1e-4
    config.train.timestep_fraction = 1.0  # Train on all timesteps
    
    config.save_dir = 'logs/bagel/aesthetic'
    
    return config

def bagel_pickscore():
    """Bagel configuration optimized for PickScore training."""
    config = bagel_base()
    
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"pickscore": 1.0}
    
    # PickScore-specific settings
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 50
    config.train.learning_rate = 1e-4
    config.train.timestep_fraction = 1.0  # Train on all timesteps
    
    config.save_dir = 'logs/bagel/pickscore'
    
    return config

def bagel_fast():
    """Bagel configuration for Flow-GRPO-Fast training."""
    config = bagel_base()
    
    # Fast training settings
    config.sample.num_steps = 10  # Fewer steps for fast training
    config.sample.eval_num_steps = 30
    config.train.learning_rate = 2e-4  # Higher learning rate for fast training
    config.train.timestep_fraction = 1.0  # Train on all timesteps
    
    # Smaller LoRA for faster training
    config.train.lora_r = 8
    config.train.lora_alpha = 16
    
    # Faster iteration batch sizes
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 4
    config.train.gradient_accumulation_steps = 2
    
    config.save_dir = 'logs/bagel/fast'
    
    return config

def bagel_multi_gpu():
    """Bagel configuration optimized for multi-GPU training."""
    config = bagel_base()
    
    # Multi-GPU optimized settings
    # Total images per batch = train_batch_size × num_image_per_prompt = 4 × 4 = 16 images per GPU
    config.sample.train_batch_size = 4
    config.sample.test_batch_size = 8
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 8
    
    config.train.batch_size = 4  # Must match sample.train_batch_size
    config.train.gradient_accumulation_steps = 2  # Effective batch size = 8
    
    # Larger LoRA for better performance
    config.train.lora_r = 32
    config.train.lora_alpha = 64
    
    config.save_dir = 'logs/bagel/multi_gpu'
    
    return config

def bagel_high_performance():
    """High-performance Bagel configuration inspired by Flow GRPO best practices."""
    config = bagel_base()
    
    # Aggressive batch sizes for maximum throughput
    # Total images per batch = train_batch_size × num_image_per_prompt = 8 × 3 = 24 images
    config.sample.train_batch_size = 8
    config.sample.test_batch_size = 16
    config.sample.num_image_per_prompt = 3
    config.sample.num_batches_per_epoch = 8
    
    config.train.batch_size = 8  # Must match sample.train_batch_size
    config.train.gradient_accumulation_steps = 1  # More frequent updates
    
    # Enhanced LoRA for better adaptation
    config.train.lora_r = 32
    config.train.lora_alpha = 64
    config.train.lora_dropout = 0.05  # Lower dropout for aggressive training
    
    # Optimized learning and evaluation
    config.train.learning_rate = 5e-5  # Higher LR for faster convergence
    config.train.total_steps = 3000
    config.save_freq = 100
    config.eval_freq = 20
    
    # Reduced inference steps for faster sampling
    config.sample.num_steps = 30  # vs 50 default
    config.sample.eval_num_steps = 40
    
    config.save_dir = 'logs/bagel/high_performance'
    
    return config

def get_config(name):
    return globals()[name]()
