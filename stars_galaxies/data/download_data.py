import zipfile
from pathlib import Path

import gdown

RAW_DIR = Path(__file__).parent / "raw"
GDRIVE_FILE_ID = "19IQDx6TzUcQYHFEg7ocvckll0VP1475j"


def download_data() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print("Data already exists locally.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "raw_data.zip"

    print("Downloading data from Google Drive...")
    gdown.download(
        id=GDRIVE_FILE_ID,
        output=str(zip_path),
        quiet=False,
    )

    print("Extracting data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    zip_path.unlink()
    print("Data downloaded and extracted.")
