import zipfile
from pathlib import Path
import subprocess
import gdown

RAW_DIR = Path(__file__).parent / "raw"
GDRIVE_FILE_ID = "19IQDx6TzUcQYHFEg7ocvckll0VP1475j"
DVC_FILE = RAW_DIR.parent / "raw.dvc"


def download_data() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print("Data already exists locally.")

        # Проверяем, нужно ли обновить DVC-метаданные
        if DVC_FILE.exists():
            print("Updating DVC metadata...")
            try:
                subprocess.run(["dvc", "add", str(RAW_DIR)], check=True)
                print("DVC metadata updated.")
            except subprocess.CalledProcessError as e:
                print(f"Error updating DVC metadata: {e}")
        return

    # Создаём директорию
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "raw_data.zip"

    # Загружаем данные
    print("Downloading data from Google Drive...")
    gdown.download(
        id=GDRIVE_FILE_ID,
        output=str(zip_path),
        quiet=False,
    )

    # Распаковываем
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    # Удаляем архив
    zip_path.unlink()
    print("Data downloaded and extracted.")

    # Добавляем данные в DVC
    print("Adding data to DVC...")
    try:
        subprocess.run(["dvc", "add", str(RAW_DIR)], check=True)
        print("Data successfully added to DVC.")
    except subprocess.CalledProcessError as e:
        print(f"Error adding data to DVC: {e}")


if __name__ == "__main__":
    download_data()
