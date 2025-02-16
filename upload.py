from huggingface_hub import upload_file, upload_folder
import os
from dotenv import load_dotenv

load_dotenv()

args = dict(
    repo_id="datenlabor-bmz/ai-language-monitor",
    repo_type="space",
    token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
)
upload_folder(
    folder_path="dist",
    path_in_repo="/",
    **args,
)
upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    **args,
)