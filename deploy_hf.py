from huggingface_hub import HfApi, create_repo
import os

TOKEN = input("Paste your Hugging Face token: ").strip()
USERNAME = input("Enter your Hugging Face username: ").strip()
REPO_NAME = "chefs-table-ai"

api = HfApi()

# Create the space
print("Creating Space...")
try:
    create_repo(
        repo_id=f"{USERNAME}/{REPO_NAME}",
        repo_type="space",
        space_sdk="gradio",
        token=TOKEN,
        exist_ok=True
    )
    print(f"Space created: https://huggingface.co/spaces/{USERNAME}/{REPO_NAME}")
except Exception as e:
    print(f"Space may already exist: {e}")

# Upload all files
print("Uploading files...")
files_to_upload = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.env']]
    for file in files:
        if not file.endswith('.pyc') and file != 'users.json' and file != 'rl_data.json':
            filepath = os.path.join(root, file)
            files_to_upload.append(filepath)

for filepath in files_to_upload:
    try:
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filepath.replace(".\\", "").replace("./", ""),
            repo_id=f"{USERNAME}/{REPO_NAME}",
            repo_type="space",
            token=TOKEN
        )
        print(f"Uploaded: {filepath}")
    except Exception as e:
        print(f"Skipped {filepath}: {e}")

print(f"\nDone! Visit: https://huggingface.co/spaces/{USERNAME}/{REPO_NAME}")
