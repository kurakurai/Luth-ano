from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import create_repo

# Define your repo and local folder
repo_id = "kurakurai/Luth-LFM2-700M-linear-0.5"
folder_path = "merged-output"

# Create repo if it doesn't exist
create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# Load the tokenizer and model from the local checkpoint
tokenizer = AutoTokenizer.from_pretrained(folder_path)
model = AutoModelForCausalLM.from_pretrained(folder_path, torch_dtype="bfloat16")

# Push both to the Hub
tokenizer.push_to_hub(repo_id)
model.push_to_hub(repo_id)
