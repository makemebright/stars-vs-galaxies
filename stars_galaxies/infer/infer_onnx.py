import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
from PIL import Image


# ---------------------------
# Класс совместимый с MLflow
# ---------------------------
class GalaxyStarONNXModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Загрузка модели ONNX из пути"""
        onnx_path = context.artifacts["onnx_model"]
        self.session = ort.InferenceSession(onnx_path)

    def preprocess_image(self, image_path: str, image_size: int = 64):
        """Преобразование изображения в формат ONNX"""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((image_size, image_size))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # batch dim
        return img_array

    def predict(self, context, model_input: dict):
        """
        model_input: {"image": path_to_image}
        """
        image_path = model_input["image"]
        input_array = self.preprocess_image(image_path)
        probs = self.session.run(None, {"input": input_array})[0].flatten()
        label = "GALAXY" if probs[0] > probs[1] else "STAR"
        return {"label": label, "probabilities": probs.tolist()}


# ---------------------------
# Утилиты
# ---------------------------
def plot_output(pred: np.ndarray, plots_dir: str, image_name: str):
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar(["Galaxy", "Star"], pred.flatten())
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    plt.title(f"Prediction for {image_name}")
    save_path = os.path.join(plots_dir, f"{image_name}_prediction.png")
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def infer_images(
    onnx_path: str,
    image_paths: List[str],
    plots_dir: str = "plots",
    image_size: int = 64,
):
    """Инференс по списку изображений"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    # Загрузка ONNX модели
    model = GalaxyStarONNXModel()
    model.session = ort.InferenceSession(onnx_path)

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Galaxy_Star_Inference")

    with mlflow.start_run():
        mlflow.log_param("onnx_model", onnx_path)
        mlflow.log_param("image_size", image_size)

        for image_path in image_paths:
            image_name = Path(image_path).name
            input_array = model.preprocess_image(image_path, image_size)
            probs = model.session.run(None, {"input": input_array})[0].flatten()
            label = "GALAXY" if probs[0] > probs[1] else "STAR"

            print(f"{image_name}: {label}, probs={probs}")

            # Логируем в MLflow
            mlflow.log_metric(f"prob_galaxy_{image_name}", float(probs[0]))
            mlflow.log_metric(f"prob_star_{image_name}", float(probs[1]))

            # Сохраняем график
            plot_path = plot_output(probs, plots_dir, Path(image_path).stem)
            mlflow.log_artifact(plot_path, artifact_path="plots")


# ---------------------------
# Основной запуск
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to test images root"
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots", help="Where to save plots"
    )
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    # Собираем все файлы из папок GALAXY и STAR
    test_paths = []
    for label in ["GALAXY", "STAR"]:
        folder = Path(args.data_dir) / label
        if folder.exists():
            test_paths.extend([str(p) for p in folder.glob("*") if p.is_file()])

    if not test_paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")

    infer_images(
        args.onnx, test_paths, plots_dir=args.plots_dir, image_size=args.image_size
    )
