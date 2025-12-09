import os
from huggingface_hub import snapshot_download

# Set a long timeout just in case
os.environ["HF_HUB_DEFAULT_TIMEOUT"] = "120" 

print("Starting dataset download...")

snapshot_download(
    repo_id="Mindgard/evaded-prompt-injection-and-jailbreak-samples",
    repo_type="dataset",
    local_dir="evaded-prompt-injection-and-jailbreak-samples" # Downloads to a folder
)

print("Download complete!")