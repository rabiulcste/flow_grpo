from huggingface_hub import snapshot_download

save_dir = os.environ["SAVE_DIR"]
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"
# Download model using the official BAGEL approach
snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    local_dir=save_dir,
    local_dir_use_symlinks=False,
    # resume_download=False,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
        