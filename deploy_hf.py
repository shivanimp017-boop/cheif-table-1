from huggingface_hub import HfApi, create_repo
import os

TOKEN = input("Paste your Hugging Face token: ").strip()
USERNAME = input("Enter your Hugging Face username: ").strip()
REPO_NAME = "chefs-table-ai"

api = HfApi()

try:
    create_repo(repo_id=f"{USERNAME}/{REPO_NAME}", repo_type="space", space_sdk="docker", token=TOKEN, exist_ok=True)
    print("Space ready.")
except Exception as e:
    print(f"Note: {e}")

SKIP = {'.git', '__pycache__', '.env', 'users.json', 'rl_data.json'}

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for file in files:
        if file.endswith('.pyc'):
            continue
        filepath = os.path.join(root, file)
        repo_path = filepath.replace(".\\", "").replace("./", "")
        try:
            api.upload_file(path_or_fileobj=filepath, path_in_repo=repo_path,
                repo_id=f"{USERNAME}/{REPO_NAME}", repo_type="space", token=TOKEN)
            print(f"Uploaded: {repo_path}")
        except Exception as e:
            print(f"Skipped {repo_path}: {e}")

print(f"\nDone! Visit: https://huggingface.co/spaces/{USERNAME}/{REPO_NAME}")
