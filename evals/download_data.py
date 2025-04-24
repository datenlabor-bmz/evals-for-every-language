# download_data.py
import requests
import tarfile
import zipfile
import io
import pandas as pd
from pathlib import Path
import sys
import huggingface_hub
from datasets import load_dataset, DatasetDict
# Import fleurs DataFrame directly from its source module
from datasets_.fleurs import fleurs

# --- Configuration ---


# Add project root to sys.path (still useful for potential future imports if needed)
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

DATA_DIR = project_root / "data"

FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
FLEURS_TARGET_DIR = DATA_DIR / "fleurs"

FLORES_PLUS_HF_ID = "openlanguagedata/flores_plus"
FLORES_TARGET_DIR = DATA_DIR / "floresp-v2.0-rc.3" / "dev_parquet" # Note: Saving as parquet

GLOTTOLOG_URL = "https://cdstar.shh.mpg.de/bitstreams/EAEA0-B44E-8CEC-EA65-0/glottolog_languoid.zip" # Assumed direct link from https://glottolog.org/meta/downloads
GLOTTOLOG_TARGET_DIR = DATA_DIR / "glottolog_languoid.csv"
GLOTTOLOG_CSV_NAME = "languoid.csv"

SCRIPTCODES_URL = "https://www.unicode.org/iso15924/iso15924-codes.html" # This is HTML, need manual download or parsing
SCRIPTCODES_TARGET_FILE = DATA_DIR / "ScriptCodes.csv"

SPBLEU_SPM_URL = "https://tinyurl.com/flores200sacrebleuspm" # Assumed direct link
SPBLEU_TARGET_DIR = DATA_DIR / "spbleu"
SPBLEU_SPM_NAME = "flores200_sacrebleu_tokenizer_spm.model"
SPBLEU_DICT_URL = "https://dl.fbaipublicfiles.com/large_objects/nllb/models/spm_200/dictionary.txt"
SPBLEU_DICT_NAME = "dictionary.txt"


# --- Helper Functions ---

def download_file(url, path: Path):
    """Downloads a file from a URL to a local path."""
    print(f"Downloading {url} to {path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {path.name}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"An error occurred while saving {path}: {e}")


def extract_tar_gz(tar_path: Path, extract_path: Path):
    """Extracts a .tar.gz file."""
    print(f"Extracting {tar_path} to {extract_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully extracted {tar_path.name}.")
        # tar_path.unlink() # Optionally remove the archive after extraction
    except tarfile.TarError as e:
        print(f"Error extracting {tar_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")


def extract_zip(zip_content: bytes, extract_path: Path, target_filename: str):
    """Extracts a specific file from zip content in memory."""
    print(f"Extracting {target_filename} from zip data to {extract_path}...")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            # Find the correct file within the zip structure
            target_zip_path = None
            for member in z.namelist():
                if member.endswith(target_filename):
                    target_zip_path = member
                    break

            if target_zip_path:
                with z.open(target_zip_path) as source, open(extract_path / target_filename, "wb") as target:
                    target.write(source.read())
                print(f"Successfully extracted {target_filename}.")
            else:
                print(f"Error: Could not find {target_filename} within the zip archive.")

    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip archive.")
    except Exception as e:
        print(f"An error occurred during zip extraction: {e}")


# --- Download Functions ---

def download_fleurs_data():
    """Downloads Fleurs audio and text data."""
    print("\n--- Downloading Fleurs Data ---")
    FLEURS_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Use the fleurs_tag column from the imported DataFrame
    fleurs_tags_list = fleurs['fleurs_tag'].tolist()

    if not fleurs_tags_list:
        print("No Fleurs tags found in imported fleurs DataFrame. Skipping Fleurs.")
        return

    print(f"Checking/Downloading Fleurs for {len(fleurs_tags_list)} languages...")

    for lang_tag in fleurs_tags_list:
        lang_dir = FLEURS_TARGET_DIR / lang_tag
        audio_dir = lang_dir / "audio"
        dev_tsv_path = lang_dir / "dev.tsv"
        dev_audio_archive_path = audio_dir / "dev.tar.gz"
        audio_extracted_marker = audio_dir / "dev" # Check if extraction likely happened

        # Download TSV
        if not dev_tsv_path.exists():
            tsv_url = f"{FLEURS_BASE_URL}/{lang_tag}/dev.tsv"
            download_file(tsv_url, dev_tsv_path)
        else:
            print(f"Found: {dev_tsv_path}")

        # Download and Extract Audio
        if not audio_extracted_marker.exists():
            if not dev_audio_archive_path.exists():
                 tar_url = f"{FLEURS_BASE_URL}/{lang_tag}/audio/dev.tar.gz"
                 download_file(tar_url, dev_audio_archive_path)

            if dev_audio_archive_path.exists():
                 extract_tar_gz(dev_audio_archive_path, audio_dir)
            else:
                print(f"Audio archive missing, cannot extract for {lang_tag}")
        else:
             print(f"Found extracted audio: {audio_extracted_marker}")

def download_flores_plus_data():
    """Downloads Flores+ data using Hugging Face datasets library."""
    print("\n--- Downloading Flores+ Data (requires HF login & accepted terms) ---")
    FLORES_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Check login status first
        token = huggingface_hub.HfFolder.get_token()
        if not token:
            print("Hugging Face token not found. Please log in using `huggingface-cli login`.")
            print("You also need to accept the terms for 'openlanguagedata/flores_plus' on the HF website.")
            return

        print(f"Attempting to download '{FLORES_PLUS_HF_ID}' (dev split)...")
        # Load only the 'dev' split
        ds = load_dataset(FLORES_PLUS_HF_ID, split='dev', verification_mode='no_checks')

        # Save as parquet files, potentially one per language if needed later
        # For simplicity now, save the whole dev split as one parquet file
        target_file = FLORES_TARGET_DIR / "dev_split.parquet"
        print(f"Saving dev split to {target_file}...")
        ds.to_parquet(target_file)
        print("Flores+ dev split downloaded and saved as parquet.")

    except huggingface_hub.utils.GatedRepoError:
        print(f"Error: Access to '{FLORES_PLUS_HF_ID}' is gated.")
        print("Please ensure you are logged in (`huggingface-cli login`) and have accepted the terms ")
        print(f"on the dataset page: https://huggingface.co/datasets/{FLORES_PLUS_HF_ID}")
    except Exception as e:
        print(f"An error occurred downloading or saving Flores+: {e}")


def download_glottolog_data():
    """Downloads and extracts Glottolog languoid CSV."""
    print("\n--- Downloading Glottolog Data ---")
    target_csv = GLOTTOLOG_TARGET_DIR / GLOTTOLOG_CSV_NAME
    if not target_csv.exists():
        print(f"Downloading Glottolog zip from {GLOTTOLOG_URL}...")
        try:
            response = requests.get(GLOTTOLOG_URL, timeout=60)
            response.raise_for_status()
            GLOTTOLOG_TARGET_DIR.mkdir(parents=True, exist_ok=True)
            extract_zip(response.content, GLOTTOLOG_TARGET_DIR, GLOTTOLOG_CSV_NAME)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Glottolog zip: {e}")
        except Exception as e:
            print(f"An error occurred processing Glottolog: {e}")
    else:
        print(f"Found: {target_csv}")


def download_scriptcodes_data():
    """Downloads ScriptCodes CSV."""
    print("\n--- Downloading ScriptCodes Data ---")
    # The URL points to an HTML page, not a direct CSV link.
    # Manual download is likely required for ScriptCodes.csv.
    print(f"Cannot automatically download from {SCRIPTCODES_URL}")
    print(f"Please manually download the ISO 15924 codes list (often available as a .txt file)")
    print("from the Unicode website or related sources and save it as:")
    print(f"{SCRIPTCODES_TARGET_FILE}")
    if SCRIPTCODES_TARGET_FILE.exists():
        print(f"Note: File already exists at {SCRIPTCODES_TARGET_FILE}")


def download_spbleu_data():
    """Downloads the SPM model and dictionary for spbleu."""
    print("\n--- Downloading spbleu SPM Model and Dictionary ---")
    SPBLEU_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Download SPM Model
    target_model_file = SPBLEU_TARGET_DIR / SPBLEU_SPM_NAME
    if not target_model_file.exists():
        print(f"Downloading SPM Model...")
        download_file(SPBLEU_SPM_URL, target_model_file)
    else:
        print(f"Found: {target_model_file}")

    # Download Dictionary
    target_dict_file = SPBLEU_TARGET_DIR / SPBLEU_DICT_NAME
    if not target_dict_file.exists():
        print(f"Downloading Dictionary...")
        download_file(SPBLEU_DICT_URL, target_dict_file)
    else:
        print(f"Found: {target_dict_file}")

# --- Conversion Function ---

def convert_flores_parquet_to_text():
    """Converts the downloaded Flores+ parquet dev split to text files."""
    print("\n--- Converting Flores+ Parquet to Text Files ---")
    parquet_file = FLORES_TARGET_DIR / "dev_split.parquet"
    text_dir = project_root / "data" / "floresp-v2.0-rc.3" / "dev" # Original expected dir

    if not parquet_file.exists():
        print(f"Parquet file not found: {parquet_file}. Skipping conversion.")
        return

    try:
        print(f"Reading parquet file: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"Read {len(df)} rows from parquet.")

        if not all(col in df.columns for col in ['iso_639_3', 'iso_15924', 'text']):
            print("Error: Parquet file missing required columns (iso_639_3, iso_15924, text).")
            return

        text_dir.mkdir(parents=True, exist_ok=True)
        print(f"Target directory for text files: {text_dir}")

        # Group by language and script to create individual files
        grouped = df.groupby(['iso_639_3', 'iso_15924'])
        count = 0
        for (lang, script), group in grouped:
            target_filename = f"dev.{lang}_{script}"
            target_path = text_dir / target_filename
            print(f"Writing {len(group)} sentences to {target_path}...")
            try:
                with open(target_path, 'w', encoding='utf-8') as f:
                    for sentence in group['text']:
                        f.write(sentence + '\n')
                count += 1
            except Exception as e:
                print(f"Error writing file {target_path}: {e}")

        print(f"Successfully wrote {count} language/script files to {text_dir}.")

    except ImportError:
        print("Error: pandas or pyarrow might be missing. Cannot read parquet.")
        print("Please install them: pip install pandas pyarrow")
    except Exception as e:
        print(f"An error occurred during parquet conversion: {e}")


# --- Main Execution ---

def main():
    """Runs all download functions and the conversion step."""
    print("Starting data download process...")
    DATA_DIR.mkdir(exist_ok=True)

    download_flores_plus_data()
    convert_flores_parquet_to_text()
    #download_fleurs_data()
    download_glottolog_data()
    download_scriptcodes_data()
    download_spbleu_data()

    print("\nData download process finished.")
    print("Please verify downloads and manually obtain ScriptCodes.csv if needed.")
    print("Note: Flores+ was downloaded as parquet, which might require changes but has been processed as well")
    print("in 'evals/datasets_/flores.py' to be read correctly.")


if __name__ == "__main__":
    main() 