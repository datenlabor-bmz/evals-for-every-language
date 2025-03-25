# /// script
# dependencies = [
#   "huggingface_hub",
#   "python-dotenv",
# ]
# ///

from huggingface_hub import upload_folder
import os
from dotenv import load_dotenv

load_dotenv()

upload_folder(
    folder_path="build",
    path_in_repo="/",
    repo_id="datenlabor-bmz/ai-language-monitor",
    repo_type="space",
    token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
)
