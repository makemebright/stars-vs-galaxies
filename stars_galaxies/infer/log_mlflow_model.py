import os
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
from PIL import Image


class GalaxyStarONNXModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Загрузка модели ONNX из пути"""
        onnx_path = context.artifacts["onnx_model"]
        self.session = ort.InferenceSession(onnx_path)

    def preprocess_image(self, image_bytes, image_size: int = 64):
        """Преобразование байтов изображения в формат для ONNX"""
        img = Image.open(image_bytes).convert("RGB")
        img = img.resize((image_size, image_size))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # add batch dim
        return img_array

    def predict(self, context, model_input):
        """
        model_input: dict с ключом 'image' -> байты файла изображения
        Возвращает словарь: {"label": str, "probabilities": [p_galaxy, p_star]}
        """
        image_bytes = model_input["image"]
        input_array = self.preprocess_image(image_bytes)

        probs = self.session.run(None, {"input": input_array})[0].flatten()
        label = "GALAXY" if probs[0] > probs[1] else "STAR"

        return {"label": label, "probabilities": probs.tolist()}


sys.path.append(os.path.dirname(__file__))  # добавляет текущую папку

# Путь к сохранённой ONNX модели
onnx_path = Path("stars_galaxies/checkpoints/galaxy_star_model.onnx")
if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

# Артефакты модели
artifacts = {"onnx_model": str(onnx_path)}

# Логируем модель в MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Galaxy_Star_Inference")

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="mlflow_model",
        python_model=GalaxyStarONNXModel(),
        artifacts=artifacts,
        registered_model_name="GalaxyStarONNX",  # можно зарегистрировать
    )
    print("Model logged to MLflow successfully!")
