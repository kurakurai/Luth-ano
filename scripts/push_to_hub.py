from huggingface_hub import create_repo, snapshot_download, upload_folder

# Existing HF repo
source_repo_id = "MaxLSB/Mistral-Small-24B-Instruct-lora-adapter-checkpoint-3648"

# New private HF repo
new_repo_id = "lightonai/Mistral-Small-24B-Instruct-lora-adapter-French"

# Create the new private repo if it doesn't exist
create_repo(repo_id=new_repo_id, repo_type="model", private=True, exist_ok=True)

# Download the source repo to a temporary folder
local_folder = snapshot_download(repo_id=source_repo_id, repo_type="model")

# Upload the downloaded folder to the new repo
upload_folder(
    folder_path=local_folder,
    repo_id=new_repo_id,
    repo_type="model",
)
