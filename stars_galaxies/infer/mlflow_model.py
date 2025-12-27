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
