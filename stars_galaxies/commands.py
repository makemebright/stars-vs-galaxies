import os
import sys
from pathlib import Path

import fire

# Добавляем путь к проекту, чтобы импортировать модули
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stars_galaxies.data.download_data import download_data

os.chdir(Path(__file__).parent.parent)


class CLI:
    """Command Line Interface for Stars vs Galaxies project."""

    def download(self):
        """Download raw data if not already present."""
        data_dir = project_root / "data" / "raw"
        if all(
            (data_dir / folder).exists() for folder in ["train", "test", "validation"]
        ):
            print("Data already exists, skipping download.")
        else:
            print("Downloading data...")
            download_data()
            print("Data download complete.")

    def train(self):
        """Train the model."""
        # Проверяем данные
        self.download()  # загрузка данных если нужно

        print("Starting training...")
        try:
            from stars_galaxies.training import train
        except ImportError:
            raise ImportError(
                "Could not import train_main from stars_galaxies.training.train"
            )

        train.run()

    def infer(
        self,
        onnx_path: str = None,
        data_dir: str = None,
        plots_dir: str = "plots",
        image_size: int = 64,
    ):
        """Run inference on test dataset."""
        self.download()  # убедимся, что данные есть

        if onnx_path is None:
            onnx_path = project_root / "checkpoints" / "galaxy_star_model.onnx"
        if data_dir is None:
            data_dir = project_root / "data" / "raw" / "test"

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"Test data not found at {data_dir}")

        print(
            f"Running inference using ONNX model at {onnx_path} on data in {data_dir}..."
        )

        # Импортируем локально, чтобы избежать циклических зависимостей
        from stars_galaxies.infer.infer_onnx import infer_images

        # Собираем все файлы из GALAXY и STAR
        test_paths = []
        for label in ["GALAXY", "STAR"]:
            folder = Path(data_dir) / label
            if folder.exists():
                test_paths.extend([str(p) for p in folder.glob("*") if p.is_file()])

        if not test_paths:
            raise FileNotFoundError(f"No images found in {data_dir}")

        infer_images(
            str(onnx_path), test_paths, plots_dir=plots_dir, image_size=image_size
        )
        print("Inference finished.")

    def test(self):
        """Test basic imports and GPU availability."""
        try:
            import pytorch_lightning as pl
            import torch

            print("✓ PyTorch version:", torch.__version__)
            print("✓ Lightning version:", pl.__version__)
            print("✓ CUDA available:", torch.cuda.is_available())
            return True
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
